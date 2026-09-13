/*
 * drowsiness_alarm.ino  -  Stage 11: physical alarm on an ESP32-WROOM-32
 * ======================================================================
 *
 * The laptop runs the whole AI (camera, MediaPipe, CNN, temporal state
 * machine, Stage 10 alerts). This board runs NONE of it. It receives the
 * drowsiness state over USB serial and drives one buzzer. That is all.
 *
 *     Python (src/hardware.py)  --USB serial 115200 8N1-->  ESP32  -->  buzzer
 *
 * Wiring (see hardware/README.md for the diagram)
 * ----------------------------------------------
 *     buzzer  +   ->  GPIO 23  (BUZZER_PIN)
 *     buzzer  -   ->  GND
 *     on-board LED  GPIO 2 mirrors the buzzer so the pattern is visible too
 *
 * Buzzer type
 * -----------
 *     BUZZER_PASSIVE 0  (default) ACTIVE buzzer: has its own oscillator, sounds
 *                       when the pin is HIGH. Just switch it on and off.
 *     BUZZER_PASSIVE 1  PASSIVE buzzer / piezo disc: needs a tone. Driven with
 *                       the ESP32 LEDC peripheral at TONE_HZ.
 *     If your buzzer only clicks instead of beeping, it is passive: set 1.
 *
 * Serial protocol (ASCII lines, '\n' terminated, '\r' ignored, case-insensitive)
 * -------------------------------------------------------------------------
 *     laptop -> ESP32                 ESP32 -> laptop
 *     HELLO    laptop program present  OK HELLO       arms the 3 s link watchdog
 *     ALERT    driver alert            OK ALERT       buzzer silent
 *     MILD     mild drowsiness         OK MILD        1 short chirp every 2 s
 *     DROWSY   drowsy                  OK DROWSY      fast beeping 150 ms on / 150 ms off
 *     CLEAR    silence now             OK CLEAR       buzzer silent until the next MILD/DROWSY
 *              (driver dismissed the alert, or the program is shutting down)
 *     PING     link check              PONG <uptime_ms>
 *     STATUS   ask                     STATUS <state> buzzer=<0|1> age_ms=<ms since last command>
 *     TEST     local demo              OK TEST  then plays MILD 3 s, DROWSY 3 s, silence
 *     anything else                    ERR unknown <text>
 *     on boot                          READY drowsiness_alarm v1 pin=23 passive=0 watchdog_ms=3000
 *     link watchdog trips              LINK LOST   (buzzer forced silent, LED blinks slowly)
 *
 * Every state command is idempotent: the laptop re-sends the current state
 * once a second as a heartbeat. After HELLO, if nothing arrives for
 * LINK_TIMEOUT_MS the board assumes the AI is no longer running and goes
 * silent (fail-silent:
 * a stuck alarm after a crash or unplug would be a false alarm, and the
 * laptop's own Stage 10 alerts are still the primary channel). The LED
 * blinks slowly to show the link is down.
 *
 * Nothing here blocks: the buzzer patterns are a millis() state machine, so
 * a new command takes effect within one loop() iteration (< 1 ms).
 */

#include <Arduino.h>

// ---- configuration -------------------------------------------------------
#define BUZZER_PIN        23        // safe general-purpose output on WROOM-32
#define LED_PIN           2         // on-board blue LED on most WROOM-32 dev boards
#define BUZZER_PASSIVE    0         // 0 = active buzzer (on/off), 1 = passive buzzer (tone)
#define TONE_HZ           2700      // passive buzzer tone (most piezos are loudest near 2.5-3 kHz)
#define TONE_HZ_ALT       2200      // second tone for the DROWSY pattern (passive only)
#define BAUD              115200
#define LINK_TIMEOUT_MS   3000UL    // no command for this long -> buzzer off, "LINK LOST"

// MILD pattern: one chirp every MILD_PERIOD_MS
#define MILD_ON_MS        100UL
#define MILD_PERIOD_MS    2000UL
// DROWSY pattern: continuous on/off beeping
#define DROWSY_ON_MS      150UL
#define DROWSY_OFF_MS     150UL

// ---- state ---------------------------------------------------------------
enum State { ST_ALERT, ST_MILD, ST_DROWSY, ST_CLEAR, ST_LOST };
static const char* STATE_NAMES[] = {"ALERT", "MILD", "DROWSY", "CLEAR", "LOST"};

State         state          = ST_ALERT;
bool          buzzerOn       = false;
unsigned long lastCommandMs  = 0;
bool          watchdogArmed  = false;    // set by HELLO: the laptop program is present
unsigned long patternStartMs = 0;
String        lineBuf;

// TEST sequence (runs locally, no laptop needed)
bool          testRunning    = false;
unsigned long testStartMs    = 0;

// ---- buzzer driver -------------------------------------------------------
#if BUZZER_PASSIVE
  #if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    // Arduino-ESP32 core 3.x API
    static void buzzerInit()            { ledcAttach(BUZZER_PIN, TONE_HZ, 10); ledcWrite(BUZZER_PIN, 0); }
    static void buzzerTone(uint32_t hz) { ledcWriteTone(BUZZER_PIN, hz); }
    static void buzzerOff()             { ledcWriteTone(BUZZER_PIN, 0); ledcWrite(BUZZER_PIN, 0); }
  #else
    // Arduino-ESP32 core 2.x API
    #define LEDC_CH 0
    static void buzzerInit()            { ledcSetup(LEDC_CH, TONE_HZ, 10); ledcAttachPin(BUZZER_PIN, LEDC_CH); ledcWrite(LEDC_CH, 0); }
    static void buzzerTone(uint32_t hz) { ledcWriteTone(LEDC_CH, hz); }
    static void buzzerOff()             { ledcWriteTone(LEDC_CH, 0); ledcWrite(LEDC_CH, 0); }
  #endif
#else
  static void buzzerInit()            { pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN, LOW); }
  static void buzzerTone(uint32_t hz) { (void)hz; digitalWrite(BUZZER_PIN, HIGH); }
  static void buzzerOff()             { digitalWrite(BUZZER_PIN, LOW); }
#endif

static void setBuzzer(bool on, uint32_t hz = TONE_HZ) {
  if (on) buzzerTone(hz); else buzzerOff();
  buzzerOn = on;
  digitalWrite(LED_PIN, on ? HIGH : LOW);
}

// ---- commands ------------------------------------------------------------
static void enterState(State s) {
  if (s != state) patternStartMs = millis();
  state = s;
  if (s == ST_ALERT || s == ST_CLEAR || s == ST_LOST) setBuzzer(false);
}

static void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;
  String cmd = line;
  cmd.toUpperCase();

  lastCommandMs = millis();
  if (state == ST_LOST) { state = ST_ALERT; patternStartMs = millis(); }   // link is back

  if      (cmd == "HELLO")  { watchdogArmed = true; Serial.println("OK HELLO"); }
  else if (cmd == "ALERT")  { testRunning = false; enterState(ST_ALERT);  Serial.println("OK ALERT"); }
  else if (cmd == "MILD")   { testRunning = false; enterState(ST_MILD);   Serial.println("OK MILD"); }
  else if (cmd == "DROWSY") { testRunning = false; enterState(ST_DROWSY); Serial.println("OK DROWSY"); }
  else if (cmd == "CLEAR")  { testRunning = false; enterState(ST_CLEAR);  Serial.println("OK CLEAR"); }
  else if (cmd == "PING")   { Serial.print("PONG "); Serial.println(millis()); }
  else if (cmd == "STATUS") {
    Serial.print("STATUS "); Serial.print(STATE_NAMES[state]);
    Serial.print(" buzzer="); Serial.print(buzzerOn ? 1 : 0);
    Serial.print(" age_ms="); Serial.println(millis() - lastCommandMs);
  }
  else if (cmd == "TEST")   { testRunning = true; testStartMs = millis(); enterState(ST_MILD); Serial.println("OK TEST"); }
  else                      { Serial.print("ERR unknown "); Serial.println(line); }
}

// ---- patterns (non-blocking) ---------------------------------------------
static void runPattern(unsigned long now) {
  unsigned long t = now - patternStartMs;
  switch (state) {
    case ST_MILD: {                                     // chirp for MILD_ON_MS at the start of every period
      bool on = (t % MILD_PERIOD_MS) < MILD_ON_MS;
      if (on != buzzerOn) setBuzzer(on, TONE_HZ);
      break;
    }
    case ST_DROWSY: {                                   // on/off beeping, alternating tone on a passive buzzer
      unsigned long period = DROWSY_ON_MS + DROWSY_OFF_MS;
      bool on = (t % period) < DROWSY_ON_MS;
      uint32_t hz = ((t / period) % 2 == 0) ? TONE_HZ : TONE_HZ_ALT;
      if (on != buzzerOn) setBuzzer(on, hz);
      break;
    }
    case ST_LOST: {                                     // silent, LED blinks slowly: 100 ms every 1.5 s
      digitalWrite(LED_PIN, (t % 1500UL) < 100UL ? HIGH : LOW);
      break;
    }
    default:                                            // ALERT / CLEAR: silent
      if (buzzerOn) setBuzzer(false);
      break;
  }
}

static void runTestSequence(unsigned long now) {
  if (!testRunning) return;
  unsigned long t = now - testStartMs;
  if (t < 3000UL)       { if (state != ST_MILD)   enterState(ST_MILD); }
  else if (t < 6000UL)  { if (state != ST_DROWSY) enterState(ST_DROWSY); }
  else                  { enterState(ST_ALERT); testRunning = false; Serial.println("TEST DONE"); }
  lastCommandMs = now;                                  // a local test must not trip the link watchdog
}

// ---- Arduino entry points ------------------------------------------------
void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  buzzerInit();
  Serial.begin(BAUD);
  delay(300);                                           // let the USB-serial bridge settle
  lastCommandMs  = millis();
  patternStartMs = millis();
  Serial.print("READY drowsiness_alarm v1 pin="); Serial.print(BUZZER_PIN);
  Serial.print(" passive="); Serial.print(BUZZER_PASSIVE);
  Serial.print(" watchdog_ms="); Serial.println(LINK_TIMEOUT_MS);
}

void loop() {
  // 1. read complete lines from the laptop
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n')       { handleLine(lineBuf); lineBuf = ""; }
    else if (c != '\r')  { if (lineBuf.length() < 64) lineBuf += c; }
  }

  unsigned long now = millis();

  // 2. link watchdog: armed by HELLO (so Serial Monitor typing is never cut off), fail-silent
  if (watchdogArmed && !testRunning && state != ST_LOST && (now - lastCommandMs) > LINK_TIMEOUT_MS) {
    enterState(ST_LOST);
    Serial.println("LINK LOST");
  }

  // 3. drive the buzzer
  runTestSequence(now);
  runPattern(now);
}
