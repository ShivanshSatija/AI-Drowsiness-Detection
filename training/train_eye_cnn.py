"""Train and evaluate the eye-state CNN (Stage 7).

Runs identically in Google Colab (GPU) and locally (CPU, for smoke tests).
The Colab notebook training/train_eye_cnn.ipynb is a thin driver around this
script, so there is one copy of the training logic.

Input: a folder with train.npz, val.npz and test.npz written by
training/prepare_mrl.py. Each file holds

    images    uint8  (N, size, size)   already gray + resized by src.eye_cnn.preprocess_eye_image
    labels    uint8  (N,)              0 = CLOSED, 1 = OPEN
    subjects  int32  (N,)              subject id - splits must be disjoint in this field
    glasses   uint8  (N,)              0 / 1 where the dataset provides it (255 = unknown)

Standardisation is applied at load time with src.eye_cnn.normalize_eye - the
same function the live system uses - after training-time augmentation of the
uint8 image (src.eye_cnn.augment_eye).

Outputs, all under --out:

    history.json          per-epoch train/val loss and accuracy, learning rate
    curves.png            training/validation curves
    last.pt / best.pt     checkpoints (best = highest validation accuracy)
    metrics.json          test-set accuracy, precision, recall, F1 (CLOSED as the
                          positive, safety-relevant class; OPEN and macro as well),
                          confusion matrix, breakdown by glasses
    confusion_matrix.png
    metrics.md            the same numbers as a report-ready table

and, with --export, the final self-describing model file (models/eye_cnn.pt).

Nothing here invents numbers: every metric is computed from the test split,
which contains subjects never seen in training or validation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eye_cnn import (CLASSES, AugmentConfig, EyePreprocessConfig, augment_eye,  # noqa: E402
                         build_model, count_parameters, load_model, normalize_eye, save_model)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


# --- data ----------------------------------------------------------------------

def load_split(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    for key in ("images", "labels", "subjects"):
        if key not in data:
            raise SystemExit("{} lacks the '{}' array - regenerate it with prepare_mrl.py".format(path, key))
    if "glasses" not in data:
        data["glasses"] = np.full(len(data["labels"]), 255, dtype=np.uint8)
    return data


def assert_subject_independent(splits: Dict[str, Dict[str, np.ndarray]]) -> None:
    """No subject may appear in more than one split. Fails loudly otherwise."""
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = set(splits[names[i]]["subjects"].tolist()), set(splits[names[j]]["subjects"].tolist())
            shared = a & b
            if shared:
                raise SystemExit("SUBJECT LEAKAGE between {} and {}: {}".format(
                    names[i], names[j], sorted(shared)[:10]))


class EyeDataset(Dataset):
    """uint8 images -> (1, size, size) float32 standardised tensors, with
    optional augmentation that is deterministic given (seed, epoch, index)."""

    def __init__(self, data: Dict[str, np.ndarray], augment: bool, preprocess: EyePreprocessConfig,
                 aug_config: Optional[AugmentConfig] = None, seed: int = 0) -> None:
        self.images = data["images"]
        self.labels = data["labels"].astype(np.int64)
        self.glasses = data["glasses"]
        self.augment = augment
        self.preprocess = preprocess
        self.aug_config = aug_config or AugmentConfig()
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        img = self.images[index]
        if self.augment:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            img = augment_eye(img, rng, self.aug_config)
        tensor = normalize_eye(img, self.preprocess.equalize)
        return torch.from_numpy(tensor)[None], int(self.labels[index])


# --- metrics -------------------------------------------------------------------

def confusion(labels: np.ndarray, preds: np.ndarray, n: int = 2) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)          # rows = true, cols = predicted
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def prf(cm: np.ndarray, positive: int) -> Tuple[float, float, float]:
    tp = cm[positive, positive]
    fp = cm[:, positive].sum() - tp
    fn = cm[positive, :].sum() - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def summarise(labels: np.ndarray, preds: np.ndarray) -> Dict:
    cm = confusion(labels, preds)
    p0, r0, f0 = prf(cm, 0)
    p1, r1, f1 = prf(cm, 1)
    return {
        "n": int(len(labels)),
        "accuracy": float((labels == preds).mean()) if len(labels) else 0.0,
        "closed": {"precision": p0, "recall": r0, "f1": f0, "support": int(cm[0].sum())},
        "open": {"precision": p1, "recall": r1, "f1": f1, "support": int(cm[1].sum())},
        "macro_f1": float((f0 + f1) / 2),
        "confusion_matrix": {"rows": "true", "cols": "predicted", "classes": list(CLASSES),
                             "matrix": cm.tolist()},
    }


@torch.no_grad()
def predict_all(model, loader, device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_labels, all_preds, losses = [], [], []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int64)
    preds = np.concatenate(all_preds) if all_preds else np.zeros(0, dtype=np.int64)
    return labels, preds, (sum(losses) / len(labels) if len(labels) else float("nan"))


# --- plots ---------------------------------------------------------------------

def plot_curves(history: List[Dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [100 * h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [100 * h["val_acc"] for h in history], label="validation")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("Eye-state CNN - subject-independent validation")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{:,}".format(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_metrics_md(metrics: Dict, path: Path) -> None:
    t = metrics["test"]
    lines = [
        "# Eye-state CNN - test set ({} images, {} unseen subjects)".format(t["n"], metrics["test_subjects"]),
        "",
        "| Metric | CLOSED | OPEN |",
        "|---|---|---|",
        "| Precision | {:.4f} | {:.4f} |".format(t["closed"]["precision"], t["open"]["precision"]),
        "| Recall | {:.4f} | {:.4f} |".format(t["closed"]["recall"], t["open"]["recall"]),
        "| F1 | {:.4f} | {:.4f} |".format(t["closed"]["f1"], t["open"]["f1"]),
        "| Support | {:,} | {:,} |".format(t["closed"]["support"], t["open"]["support"]),
        "",
        "Accuracy **{:.4f}**, macro F1 {:.4f}.".format(t["accuracy"], t["macro_f1"]),
        "",
        "Confusion matrix (rows = true, cols = predicted; order CLOSED, OPEN):",
        "",
        "```",
        "{}".format(np.array(t["confusion_matrix"]["matrix"])),
        "```",
        "",
    ]
    if metrics.get("by_glasses"):
        lines += ["| Glasses | n | Accuracy | CLOSED recall | OPEN recall |", "|---|---|---|---|---|"]
        for name, m in metrics["by_glasses"].items():
            lines.append("| {} | {:,} | {:.4f} | {:.4f} | {:.4f} |".format(
                name, m["n"], m["accuracy"], m["closed"]["recall"], m["open"]["recall"]))
        lines.append("")
    lines += ["Training: {} epochs, best validation accuracy {:.4f} at epoch {}, seed {}, {} parameters, "
              "device {}, {:.1f} min.".format(
                  metrics["epochs_run"], metrics["best_val_acc"], metrics["best_epoch"], metrics["seed"],
                  metrics["parameters"], metrics["device"], metrics["train_minutes"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- training ------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the eye-state CNN.")
    parser.add_argument("--data", type=Path, required=True, help="Folder with train.npz, val.npz, test.npz")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default runs/<timestamp>)")
    parser.add_argument("--export", type=Path, default=None, help="Write the final model here (e.g. models/eye_cnn.pt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--no-augment", action="store_true", help="Ablation: train without augmentation")
    parser.add_argument("--equalize", action="store_true", help="Ablation: CLAHE before standardisation")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N images per split (smoke tests)")
    parser.add_argument("--resume", type=Path, default=None, help="Continue from a last.pt checkpoint")
    parser.add_argument("--patience", type=int, default=8, help="Stop when val accuracy has not improved for N epochs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate <out>/best.pt on the test split and write metrics / export")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = args.out or (PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    preprocess = EyePreprocessConfig(equalize=args.equalize)

    # --- data -------------------------------------------------------------------
    splits = {name: load_split(args.data / "{}.npz".format(name)) for name in ("train", "val", "test")}
    if args.limit:
        splits = {k: {kk: vv[:args.limit] for kk, vv in v.items()} for k, v in splits.items()}
    assert_subject_independent(splits)
    size = int(splits["train"]["images"].shape[1])
    if size != preprocess.size:
        preprocess.size = size
    print("[train] device {} | image size {} | augmentation {} | CLAHE {}".format(
        device, size, "off" if args.no_augment else "on", "on" if args.equalize else "off"))
    for name, d in splits.items():
        labels = d["labels"]
        print("[train] {:<5} {:>7,} images  {:>3} subjects  closed {:>6,} ({:.1%})  open {:>6,}  glasses {}".format(
            name, len(labels), len(np.unique(d["subjects"])), int((labels == 0).sum()),
            float((labels == 0).mean()) if len(labels) else 0.0, int((labels == 1).sum()),
            "n/a" if (d["glasses"] == 255).all() else "{:.1%}".format(float((d["glasses"] == 1).mean()))))

    train_ds = EyeDataset(splits["train"], augment=not args.no_augment, preprocess=preprocess, seed=args.seed)
    val_ds = EyeDataset(splits["val"], augment=False, preprocess=preprocess)
    test_ds = EyeDataset(splits["test"], augment=False, preprocess=preprocess)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=pin, drop_last=False, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=args.workers, pin_memory=pin)

    # --- model ------------------------------------------------------------------
    model = build_model(width=args.width, dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    counts = np.bincount(splits["train"]["labels"], minlength=2).astype(np.float64)
    class_weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=max(1, args.epochs), steps_per_epoch=max(1, len(train_loader)))
    print("[train] EyeStateCNN width {} -> {:,} parameters; class weights {}".format(
        args.width, n_params, [round(float(w), 3) for w in class_weights.cpu()]))

    history: List[Dict] = []
    start_epoch, best_val_acc, best_epoch = 1, -1.0, 0
    if args.eval_only and args.resume is None and (out / "last.pt").exists():
        args.resume = out / "last.pt"          # recover history and best-epoch bookkeeping
    if args.resume and args.resume.exists():
        ckpt = torch.load(str(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        best_val_acc, best_epoch = ckpt["best_val_acc"], ckpt["best_epoch"]
        print("[train] resumed from {} at epoch {}".format(args.resume, start_epoch))
    if args.eval_only:
        if not (out / "best.pt").exists():
            raise SystemExit("--eval-only needs {} - nothing to evaluate".format(out / "best.pt"))
        start_epoch = args.epochs + 1          # skip the training loop entirely
        print("[train] --eval-only: evaluating {} (best epoch {}, val acc {:.4f}, {} epochs recorded)".format(
            out / "best.pt", best_epoch, best_val_acc, len(history)))

    # --- loop -------------------------------------------------------------------
    t_start = time.perf_counter()
    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0
        t_epoch = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * len(y)
            running_correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += len(y)
        train_loss, train_acc = running_loss / max(1, seen), running_correct / max(1, seen)

        val_labels, val_preds, val_loss = predict_all(model, val_loader, device)
        val_acc = float((val_labels == val_preds).mean()) if len(val_labels) else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0],
                        "seconds": time.perf_counter() - t_epoch})
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch, epochs_without_improvement = val_acc, epoch, 0
        else:
            epochs_without_improvement += 1
        print("[train] epoch {:>3}/{}  train loss {:.4f} acc {:.4f} | val loss {:.4f} acc {:.4f} | "
              "lr {:.2e} | {:.0f}s{}".format(epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
                                             history[-1]["lr"], history[-1]["seconds"], "  *best*" if improved else ""))

        state = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "history": history,
                 "best_val_acc": best_val_acc, "best_epoch": best_epoch, "args": vars(args) | {"data": str(args.data),
                 "out": str(out), "export": str(args.export), "resume": str(args.resume)}}
        torch.save(state, str(out / "last.pt"))
        if improved:
            save_model(model, out / "best.pt", preprocess,
                       {"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss, "seed": args.seed})
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_curves(history, out / "curves.png")
        if epochs_without_improvement >= args.patience:
            print("[train] early stop: no validation improvement for {} epochs".format(args.patience))
            break
    train_minutes = ((time.perf_counter() - t_start) / 60.0 if not args.eval_only
                     else sum(h.get("seconds", 0.0) for h in history) / 60.0)

    # --- final evaluation on unseen subjects -------------------------------------
    best_model, _ = load_model(out / "best.pt", device=device)
    test_labels, test_preds, test_loss = predict_all(best_model, test_loader, device)
    metrics: Dict = {
        "test": summarise(test_labels, test_preds),
        "test_loss": test_loss,
        "test_subjects": int(len(np.unique(splits["test"]["subjects"]))),
        "val_best": {"accuracy": best_val_acc, "epoch": best_epoch},
        "best_val_acc": best_val_acc, "best_epoch": best_epoch,
        "epochs_run": len(history), "seed": args.seed, "parameters": n_params, "device": device,
        "train_minutes": train_minutes, "augmentation": not args.no_augment, "equalize": args.equalize,
        "image_size": size, "class_order": list(CLASSES),
        "splits": {name: {"images": int(len(d["labels"])), "subjects": int(len(np.unique(d["subjects"])))}
                   for name, d in splits.items()},
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    glasses = splits["test"]["glasses"]
    if not (glasses == 255).all():
        metrics["by_glasses"] = {}
        for value, name in ((0, "no glasses"), (1, "glasses")):
            mask = glasses == value
            if mask.any():
                metrics["by_glasses"][name] = summarise(test_labels[mask], test_preds[mask])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion(np.array(metrics["test"]["confusion_matrix"]["matrix"]), out / "confusion_matrix.png",
                   "Test set - {} unseen subjects".format(metrics["test_subjects"]))
    write_metrics_md(metrics, out / "metrics.md")

    t = metrics["test"]
    print("[eval ] TEST ({:,} images, {} unseen subjects): accuracy {:.4f} | CLOSED precision {:.4f} recall {:.4f} "
          "F1 {:.4f} | OPEN precision {:.4f} recall {:.4f} F1 {:.4f} | macro F1 {:.4f}".format(
              t["n"], metrics["test_subjects"], t["accuracy"], t["closed"]["precision"], t["closed"]["recall"],
              t["closed"]["f1"], t["open"]["precision"], t["open"]["recall"], t["open"]["f1"], t["macro_f1"]))
    print("[eval ] confusion matrix (rows true CLOSED, OPEN; cols predicted):\n{}".format(
        np.array(t["confusion_matrix"]["matrix"])))
    for name, m in metrics.get("by_glasses", {}).items():
        print("[eval ] {:<11} n {:>6,}  accuracy {:.4f}  CLOSED recall {:.4f}  OPEN recall {:.4f}".format(
            name, m["n"], m["accuracy"], m["closed"]["recall"], m["open"]["recall"]))
    print("[eval ] wrote {}".format(", ".join(p.name for p in sorted(out.iterdir()))))

    if args.export:
        save_model(best_model, args.export, preprocess, {
            "test_accuracy": t["accuracy"], "closed_f1": t["closed"]["f1"], "open_f1": t["open"]["f1"],
            "test_images": t["n"], "test_subjects": metrics["test_subjects"], "best_epoch": best_epoch,
            "epochs_run": len(history), "seed": args.seed, "augmentation": not args.no_augment,
            "data": str(args.data), "run": str(out)})
        print("[eval ] exported final model -> {}".format(args.export))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
