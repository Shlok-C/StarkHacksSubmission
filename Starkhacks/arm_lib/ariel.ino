#include <AccelStepper.h>

// =========================
// Encoder pins
// =========================
const int ENC1_CLK = 22;
const int ENC1_DT  = 23;
const int ENC1_SW  = 24;

const int ENC2_CLK = 25;
const int ENC2_DT  = 26;
const int ENC2_SW  = 27;

const int ENC3_CLK = 28;
const int ENC3_DT  = 29;
const int ENC3_SW  = 30;

// =========================
// TMC2208 pins
// =========================
const int TMC_STEP_PIN = 2;
const int TMC_DIR_PIN  = 3;
const int TMC_EN_PIN   = 4;

// =========================
// L298N pins
// =========================
const int L298_ENA = 5;   // PWM
const int L298_ENB = 6;   // PWM
const int L298_IN1 = 31;
const int L298_IN2 = 32;
const int L298_IN3 = 33;
const int L298_IN4 = 34;

// =========================
// Tuning
// =========================
const int TMC_STEPS_PER_CLICK = 20;
const int L298_STEPS_PER_CLICK = 30;

const int L298_PWM_VALUE = 255;
const int L298_STEP_DELAY = 2;

const unsigned long PRINT_INTERVAL_MS = 120;
const unsigned long BUTTON_DEBOUNCE_MS = 30;
const unsigned long L298_IDLE_TIMEOUT_MS = 500;   // turn off after idle

// =========================
// TMC2208 via AccelStepper
// =========================
AccelStepper tmcStepper(AccelStepper::DRIVER, TMC_STEP_PIN, TMC_DIR_PIN);

// =========================
// Encoder state
// =========================
long enc1Count = 0;
long enc2Count = 0;
long enc3Count = 0;

int lastEnc1CLK;
int lastEnc2CLK;
int lastEnc3CLK;

// button states
bool btn1Stable = HIGH;
bool btn2Stable = HIGH;
bool btn3Stable = HIGH;

bool btn1LastRead = HIGH;
bool btn2LastRead = HIGH;
bool btn3LastRead = HIGH;

unsigned long btn1LastChange = 0;
unsigned long btn2LastChange = 0;
unsigned long btn3LastChange = 0;

// =========================
// Motor position tracking
// =========================
long tmcTargetPosition = 0;
long l298MotorPosition = 0;
int l298SeqIndex = 0;

// L298 idle management
unsigned long lastL298MoveTime = 0;
bool l298Enabled = false;

// Half-step sequence for L298N bipolar stepper
const uint8_t L298_SEQ[8][4] = {
  {1,0,0,0},
  {1,0,1,0},
  {0,0,1,0},
  {0,1,1,0},
  {0,1,0,0},
  {0,1,0,1},
  {0,0,0,1},
  {1,0,0,1}
};

unsigned long lastPrintTime = 0;

// =========================
// Setup
// =========================
void setup() {
  Serial.begin(115200);

  // Encoder inputs
  pinMode(ENC1_CLK, INPUT_PULLUP);
  pinMode(ENC1_DT,  INPUT_PULLUP);
  pinMode(ENC1_SW,  INPUT_PULLUP);

  pinMode(ENC2_CLK, INPUT_PULLUP);
  pinMode(ENC2_DT,  INPUT_PULLUP);
  pinMode(ENC2_SW,  INPUT_PULLUP);

  pinMode(ENC3_CLK, INPUT_PULLUP);
  pinMode(ENC3_DT,  INPUT_PULLUP);
  pinMode(ENC3_SW,  INPUT_PULLUP);

  lastEnc1CLK = digitalRead(ENC1_CLK);
  lastEnc2CLK = digitalRead(ENC2_CLK);
  lastEnc3CLK = digitalRead(ENC3_CLK);

  // TMC2208
  pinMode(TMC_EN_PIN, OUTPUT);
  digitalWrite(TMC_EN_PIN, LOW);  // enable driver

  tmcStepper.setMaxSpeed(1200.0);
  tmcStepper.setAcceleration(800.0);
  tmcStepper.setCurrentPosition(0);

  // L298N
  pinMode(L298_ENA, OUTPUT);
  pinMode(L298_ENB, OUTPUT);
  pinMode(L298_IN1, OUTPUT);
  pinMode(L298_IN2, OUTPUT);
  pinMode(L298_IN3, OUTPUT);
  pinMode(L298_IN4, OUTPUT);

  disableL298Motor();   // start with L298 off

  Serial.println("System started");
  Serial.println("ENC1 -> TMC2208, ENC2 -> monitor only, ENC3 -> L298N");
}

// =========================
// Main loop
// =========================
void loop() {
  readEncoder1();
  readEncoder2();
  readEncoder3();

  updateButton(ENC1_SW, btn1LastRead, btn1Stable, btn1LastChange);
  updateButton(ENC2_SW, btn2LastRead, btn2Stable, btn2LastChange);
  updateButton(ENC3_SW, btn3LastRead, btn3Stable, btn3LastChange);

  // Keep TMC motor moving toward latest encoder-set target
  tmcStepper.run();

  // Shut off L298N after idle timeout
  if (l298Enabled && (millis() - lastL298MoveTime > L298_IDLE_TIMEOUT_MS)) {
    disableL298Motor();
  }

  if (millis() - lastPrintTime >= PRINT_INTERVAL_MS) {
    lastPrintTime = millis();
    printStatus();
  }
}

// =========================
// Encoder readers
// =========================
void readEncoder1() {
  int currentCLK = digitalRead(ENC1_CLK);

  if (currentCLK != lastEnc1CLK && currentCLK == HIGH) {
    if (digitalRead(ENC1_DT) != currentCLK) {
      enc1Count++;
      tmcTargetPosition += TMC_STEPS_PER_CLICK;
    } else {
      enc1Count--;
      tmcTargetPosition -= TMC_STEPS_PER_CLICK;
    }

    tmcStepper.moveTo(tmcTargetPosition);
  }

  lastEnc1CLK = currentCLK;
}

void readEncoder2() {
  int currentCLK = digitalRead(ENC2_CLK);

  if (currentCLK != lastEnc2CLK && currentCLK == HIGH) {
    if (digitalRead(ENC2_DT) != currentCLK) {
      enc2Count++;
    } else {
      enc2Count--;
    }
  }

  lastEnc2CLK = currentCLK;
}

void readEncoder3() {
  int currentCLK = digitalRead(ENC3_CLK);

  if (currentCLK != lastEnc3CLK && currentCLK == HIGH) {
    enableL298Motor();
    lastL298MoveTime = millis();

    if (digitalRead(ENC3_DT) != currentCLK) {
      enc3Count++;
      stepL298(true, L298_STEPS_PER_CLICK);
    } else {
      enc3Count--;
      stepL298(false, L298_STEPS_PER_CLICK);
    }
  }

  lastEnc3CLK = currentCLK;
}

// =========================
// Button debounce
// =========================
void updateButton(int pin, bool &lastRead, bool &stableState, unsigned long &lastChangeTime) {
  bool reading = digitalRead(pin);

  if (reading != lastRead) {
    lastChangeTime = millis();
    lastRead = reading;
  }

  if ((millis() - lastChangeTime) > BUTTON_DEBOUNCE_MS) {
    stableState = reading;
  }
}

// =========================
// L298N enable/disable
// =========================
void enableL298Motor() {
  if (!l298Enabled) {
    analogWrite(L298_ENA, L298_PWM_VALUE);
    analogWrite(L298_ENB, L298_PWM_VALUE);
    l298Enabled = true;
  }
}

void disableL298Motor() {
  analogWrite(L298_ENA, 0);
  analogWrite(L298_ENB, 0);

  digitalWrite(L298_IN1, LOW);
  digitalWrite(L298_IN2, LOW);
  digitalWrite(L298_IN3, LOW);
  digitalWrite(L298_IN4, LOW);

  l298Enabled = false;
}

// =========================
// L298N stepper control
// =========================
void stepL298(bool forward, int steps) {
  for (int i = 0; i < steps; i++) {
    if (forward) {
      l298SeqIndex++;
      if (l298SeqIndex > 7) l298SeqIndex = 0;
      l298MotorPosition++;
    } else {
      l298SeqIndex--;
      if (l298SeqIndex < 0) l298SeqIndex = 7;
      l298MotorPosition--;
    }

    setL298Outputs(
      L298_SEQ[l298SeqIndex][0],
      L298_SEQ[l298SeqIndex][1],
      L298_SEQ[l298SeqIndex][2],
      L298_SEQ[l298SeqIndex][3]
    );

    delay(L298_STEP_DELAY);
  }
}

void setL298Outputs(int a, int b, int c, int d) {
  digitalWrite(L298_IN1, a);
  digitalWrite(L298_IN2, b);
  digitalWrite(L298_IN3, c);
  digitalWrite(L298_IN4, d);
}

// =========================
// Serial print
// =========================
void printStatus() {
  Serial.print("E1=");
  Serial.print(enc1Count);
  Serial.print(" BTN1=");
  Serial.print(btn1Stable == LOW ? "PRESSED" : "RELEASED");
  Serial.print(" TMC_target=");
  Serial.print(tmcTargetPosition);
  Serial.print(" TMC_pos=");
  Serial.print(tmcStepper.currentPosition());

  Serial.print(" | E2=");
  Serial.print(enc2Count);
  Serial.print(" BTN2=");
  Serial.print(btn2Stable == LOW ? "PRESSED" : "RELEASED");

  Serial.print(" | E3=");
  Serial.print(enc3Count);
  Serial.print(" BTN3=");
  Serial.print(btn3Stable == LOW ? "PRESSED" : "RELEASED");
  Serial.print(" L298_pos=");
  Serial.print(l298MotorPosition);
  Serial.print(" L298_enabled=");
  Serial.println(l298Enabled ? "YES" : "NO");
}
