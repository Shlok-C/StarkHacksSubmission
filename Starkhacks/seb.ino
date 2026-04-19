#include <AccelStepper.h>

// ======================================================
// CALIBRATION & GEAR RATIOS (TUNE THESE!)
// ======================================================
// How many motor steps equal 1 degree of joint movement?
const float TMC_STEPS_PER_DEGREE = 50.0;  // Example value, please tune
const float L298_STEPS_PER_DEGREE = 20.0; // Example value, please tune

// ======================================================
// ENCODER PINS
// ======================================================
const int ENC1_CLK = 22;
const int ENC1_DT  = 23;
const int ENC1_SW  = 24;   // button used to request calibration

const int ENC2_CLK = 27;
const int ENC2_DT  = 26;

const int ENC3_CLK = 28;
const int ENC3_DT  = 29;

// ======================================================
// LIMIT SWITCH PINS
// ======================================================
const int LIM1_PIN = 35;   // TMC2208 home switch
const int LIM2_PIN = 37;   // L298N home switch
const int LIM3_PIN = 41;   // just reported for now

// ======================================================
// TMC2208 PINS
// ======================================================
const int TMC_STEP_PIN = 2;
const int TMC_DIR_PIN  = 3;
const int TMC_EN_PIN   = 4;

// ======================================================
// L298N PINS
// ======================================================
const int L298_ENA = 5;    // PWM
const int L298_ENB = 6;    // PWM
const int L298_IN1 = 31;
const int L298_IN2 = 32;
const int L298_IN3 = 33;
const int L298_IN4 = 34;

// ======================================================
// SERIAL MESSAGE STATE CODES
// ======================================================
int control_mode = 0;

enum SystemState {
  STATE_NORMAL,
  STATE_WAITING_FOR_ACK,
  STATE_CALIBRATING
};

SystemState systemState = STATE_NORMAL;

// ======================================================
// ENCODER COUNTS
// ======================================================
long enc1Count = 0;
long enc2Count = 0;
long enc3Count = 0;

int lastEnc1CLK;
int lastEnc2CLK;
int lastEnc3CLK;

// ======================================================
// BUTTON DEBOUNCE & TIMING
// ======================================================
bool lastButtonState = HIGH;
bool stableButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_DELAY = 50;

unsigned long lastPrintTime = 0;
const unsigned long PRINT_INTERVAL_MS = 20;

String rxLine = "";

// ======================================================
// TMC2208 SETUP
// ======================================================
AccelStepper tmcStepper(AccelStepper::DRIVER, TMC_STEP_PIN, TMC_DIR_PIN);

const long TMC_STEPS_PER_CLICK = 20;
const float TMC_MOVE_SPEED = 1200.0;
const float TMC_ACCEL      = 800.0;
long tmcTargetPosition = 0;

const float TMC_HOME_SPEED = 400.0;
const long  TMC_BACKOFF_STEPS = 800;
const unsigned long TMC_HOME_TIMEOUT_MS = 15000;
const bool TMC_HOME_DIR_POSITIVE = false;

// ======================================================
// L298N SETUP
// ======================================================
const int L298_PWM_VALUE = 255;
const int L298_JOG_STEPS_PER_CLICK = 30;
const int L298_JOG_STEP_DELAY_MS = 6;

const int L298_HOME_STEP_DELAY_MS = 6;
const long L298_BACKOFF_HALFSTEPS = 600;
const unsigned long L298_HOME_TIMEOUT_MS = 15000;
const bool L298_HOME_DIR_FORWARD = false;

int l298SeqIndex = 0;
long l298Position = 0;
long l298TargetPosition = 0;
unsigned long lastL298StepTime = 0;

const uint8_t L298_SEQ[8][4] = {
  {1,0,0,0}, {1,0,1,0}, {0,0,1,0}, {0,1,1,0},
  {0,1,0,0}, {0,1,0,1}, {0,0,0,1}, {1,0,0,1}
};

// ======================================================
// FORWARD DECLARATIONS
// ======================================================
void readEncoder1(); void readEncoder2(); void readEncoder3();
void handleEnc1Button(); void handleIncomingSerial();
void sendStatusLine(); bool limitPressed(int pin);
void runCalibrationSequence(); bool homeTMCAxis(); bool homeL298Axis();
void enableL298Motor(); void disableL298Motor();
void setL298Outputs(int a, int b, int c, int d);
void stepL298(bool forward, int steps, int stepDelayMs);
void streamDuringCalibration();

// ======================================================
// SETUP
// ======================================================
void setup() {
  Serial.begin(115200);

  pinMode(ENC1_CLK, INPUT_PULLUP); pinMode(ENC1_DT, INPUT_PULLUP); pinMode(ENC1_SW, INPUT_PULLUP);
  pinMode(ENC2_CLK, INPUT_PULLUP); pinMode(ENC2_DT, INPUT_PULLUP);
  pinMode(ENC3_CLK, INPUT_PULLUP); pinMode(ENC3_DT, INPUT_PULLUP);

  lastEnc1CLK = digitalRead(ENC1_CLK); lastEnc2CLK = digitalRead(ENC2_CLK); lastEnc3CLK = digitalRead(ENC3_CLK);

  pinMode(LIM1_PIN, INPUT_PULLUP); pinMode(LIM2_PIN, INPUT_PULLUP); pinMode(LIM3_PIN, INPUT_PULLUP);

  pinMode(TMC_EN_PIN, OUTPUT); digitalWrite(TMC_EN_PIN, LOW);
  tmcStepper.setMaxSpeed(TMC_MOVE_SPEED); tmcStepper.setAcceleration(TMC_ACCEL); tmcStepper.setCurrentPosition(0);

  pinMode(L298_ENA, OUTPUT); pinMode(L298_ENB, OUTPUT);
  pinMode(L298_IN1, OUTPUT); pinMode(L298_IN2, OUTPUT); pinMode(L298_IN3, OUTPUT); pinMode(L298_IN4, OUTPUT);

  disableL298Motor();
}

// ======================================================
// MAIN LOOP
// ======================================================
void loop() {
  handleIncomingSerial();

  readEncoder1(); readEncoder2(); readEncoder3();

  if (systemState == STATE_NORMAL) {
    handleEnc1Button();

    // Move TMC2208 towards target
    tmcStepper.moveTo(tmcTargetPosition);
    tmcStepper.run();

    // Move L298N towards target (non-blocking)
    if (l298Position != l298TargetPosition) {
      if (millis() - lastL298StepTime >= L298_JOG_STEP_DELAY_MS) {
        lastL298StepTime = millis();
        bool forward = (l298TargetPosition > l298Position);
        enableL298Motor();
        stepL298(forward, 1, 0); // Move 1 step, with 0 delay (non-blocking)
      }
    }
  }

  if (systemState == STATE_NORMAL || systemState == STATE_WAITING_FOR_ACK) {
    if (millis() - lastPrintTime >= PRINT_INTERVAL_MS) {
      lastPrintTime = millis();
      sendStatusLine();
    }
  }

  if (systemState == STATE_CALIBRATING) {
    runCalibrationSequence();
    systemState = STATE_NORMAL;
    control_mode = 0;
  }
}

// ======================================================
// SERIAL PARSER
// ======================================================
void handleIncomingSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (rxLine.length() > 0) {
        rxLine.trim();

        // 1. Homing Acknowledgement
        if (rxLine == "2" && systemState == STATE_WAITING_FOR_ACK) {
          control_mode = 2;
          systemState = STATE_CALIBRATING;
        }
        
        // 2. Python Angle Command: A,base_deg,elbow_deg
        else if (rxLine.startsWith("A,")) {
          int firstComma = rxLine.indexOf(',');
          int secondComma = rxLine.indexOf(',', firstComma + 1);
          
          if (firstComma != -1 && secondComma != -1) {
            float targetBase = rxLine.substring(firstComma + 1, secondComma).toFloat();
            float targetElbow = rxLine.substring(secondComma + 1).toFloat();

            // Convert degrees to steps
            tmcTargetPosition = (long)(targetBase * TMC_STEPS_PER_DEGREE);
            l298TargetPosition = (long)(targetElbow * L298_STEPS_PER_DEGREE);
          }
        }

        rxLine = "";
      }
    } else {
      rxLine += c;
    }
  }
}

// ======================================================
// OTHER FUNCTIONS (Unchanged logic, just compacted)
// ======================================================
void sendStatusLine() {
  Serial.print(control_mode); Serial.print(",");
  Serial.print(enc1Count); Serial.print(","); Serial.print(enc2Count); Serial.print(","); Serial.print(enc3Count); Serial.print(",");
  Serial.print(digitalRead(LIM1_PIN)); Serial.print(","); Serial.print(digitalRead(LIM2_PIN)); Serial.print(","); Serial.println(digitalRead(LIM3_PIN));
}

void handleEnc1Button() {
  int reading = digitalRead(ENC1_SW);
  if (reading != lastButtonState) lastDebounceTime = millis();
  if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
    if (reading != stableButtonState) {
      stableButtonState = reading;
      if (stableButtonState == LOW) { control_mode = 1; systemState = STATE_WAITING_FOR_ACK; }
    }
  }
  lastButtonState = reading;
}

void readEncoder1() { /* Omitted for brevity: keep your original logic here */ }
void readEncoder2() { /* Omitted for brevity: keep your original logic here */ }
void readEncoder3() { /* Omitted for brevity: keep your original logic here */ }
bool limitPressed(int pin) { return digitalRead(pin) == LOW; }

void runCalibrationSequence() {
  control_mode = 2;
  homeTMCAxis(); homeL298Axis();
  disableL298Motor();
  tmcStepper.setCurrentPosition(0); l298Position = 0; tmcTargetPosition = 0; l298TargetPosition = 0;
}

bool homeTMCAxis() {
  unsigned long startTime = millis();
  tmcStepper.setSpeed(TMC_HOME_DIR_POSITIVE ? TMC_HOME_SPEED : -TMC_HOME_SPEED);
  while (!limitPressed(LIM1_PIN)) {
    tmcStepper.runSpeed(); streamDuringCalibration();
    if (millis() - startTime > TMC_HOME_TIMEOUT_MS) return false;
  }
  long targetPos = tmcStepper.currentPosition() + (TMC_HOME_DIR_POSITIVE ? -TMC_BACKOFF_STEPS : TMC_BACKOFF_STEPS);
  tmcStepper.moveTo(targetPos);
  while (tmcStepper.distanceToGo() != 0) { tmcStepper.run(); streamDuringCalibration(); }
  return true;
}

bool homeL298Axis() {
  unsigned long startTime = millis();
  enableL298Motor();
  while (!limitPressed(LIM2_PIN)) {
    stepL298(L298_HOME_DIR_FORWARD, 1, L298_HOME_STEP_DELAY_MS); streamDuringCalibration();
    if (millis() - startTime > L298_HOME_TIMEOUT_MS) return false;
  }
  stepL298(!L298_HOME_DIR_FORWARD, L298_BACKOFF_HALFSTEPS, L298_HOME_STEP_DELAY_MS);
  return true;
}

void streamDuringCalibration() {
  if (millis() - lastPrintTime >= PRINT_INTERVAL_MS) {
    lastPrintTime = millis(); control_mode = 2; sendStatusLine();
  }
  handleIncomingSerial();
}

void enableL298Motor() { analogWrite(L298_ENA, L298_PWM_VALUE); analogWrite(L298_ENB, L298_PWM_VALUE); }
void disableL298Motor() { analogWrite(L298_ENA, 0); analogWrite(L298_ENB, 0); setL298Outputs(LOW,LOW,LOW,LOW); }
void setL298Outputs(int a, int b, int c, int d) { digitalWrite(L298_IN1, a); digitalWrite(L298_IN2, b); digitalWrite(L298_IN3, c); digitalWrite(L298_IN4, d); }
void stepL298(bool forward, int steps, int stepDelayMs) {
  for (int i = 0; i < steps; i++) {
    if (forward) { l298SeqIndex++; if (l298SeqIndex > 7) l298SeqIndex = 0; l298Position++; } 
    else { l298SeqIndex--; if (l298SeqIndex < 0) l298SeqIndex = 7; l298Position--; }
    setL298Outputs(L298_SEQ[l298SeqIndex][0], L298_SEQ[l298SeqIndex][1], L298_SEQ[l298SeqIndex][2], L298_SEQ[l298SeqIndex][3]);
    if (stepDelayMs > 0) delay(stepDelayMs);
  }
}
