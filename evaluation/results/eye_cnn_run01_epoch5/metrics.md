# Eye-state CNN - test set (13107 images, 8 unseen subjects)

| Metric | CLOSED | OPEN |
|---|---|---|
| Precision | 0.9663 | 0.9118 |
| Recall | 0.9057 | 0.9687 |
| F1 | 0.9350 | 0.9394 |
| Support | 6,531 | 6,576 |

Accuracy **0.9373**, macro F1 0.9372.

Confusion matrix (rows = true, cols = predicted; order CLOSED, OPEN):

```
[[5915  616]
 [ 206 6370]]
```

| Glasses | n | Accuracy | CLOSED recall | OPEN recall |
|---|---|---|---|---|
| no glasses | 8,549 | 0.9641 | 0.9691 | 0.9581 |
| glasses | 4,558 | 0.8870 | 0.7473 | 0.9840 |

Training: 5 epochs, best validation accuracy 0.9703 at epoch 5, seed 0, 582562 parameters, device cpu, 34.3 min.
