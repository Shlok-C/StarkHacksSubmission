#include <AccelStepper.h>




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
// ANGLE TRACKING CONSTANTS
// ======================================================
// TMC2208: 200 full steps/rev * 8x microstepping (default MS1=LOW, MS2=LOW on most breakouts)
// Change multiplier if your MS1/MS2 pins are wired differently:
//   8x  -> 1600   (MS1=LOW,  MS2=LOW)
//   2x  -> 400    (MS1=HIGH, MS2=LOW)
//   4x  -> 800    (MS1=LOW,  MS2=HIGH)
//   16x -> 3200   (MS1=HIGH, MS2=HIGH)
const float TMC_STEPS_PER_REV = 1600.0;


// L298N half-step: standard 200-step motor * 2 = 400
// If using 28BYJ-48 with gearbox, use 4096.0 instead
const float L298_HALFSTEPS_PER_REV = 400.0;




// ======================================================
// SERIAL MESSAGE STATE CODES
// ======================================================
// 0 = normal/manual jogging
// 1 = homing requested / waiting for Pi ack
// 2 = homing in progress
int control_mode = 0;




// ======================================================
// STATE MACHINE
// ======================================================
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


int lastEnc2CLK;
int lastEnc3CLK;




// ======================================================
// BUTTON DEBOUNCE
// ======================================================
bool lastButtonState    = HIGH;
bool stableButtonState  = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_DELAY = 50;




// ======================================================
// STREAMING TIMING
// ======================================================
unsigned long lastPrintTime = 0;
const unsigned long PRINT_INTERVAL_MS = 20;




// ======================================================
// SERIAL INPUT BUFFER
// ======================================================
String rxLine = "";


// ======================================================
// PI POSITION CONTROL
// Set by "A,<base_deg>,<elbow_deg>" from Pi.
// piControlActive disables encoder jogging on both axes.
// ======================================================
bool piControlActive  = false;
long tmcTargetFromPi  = 0;
long l298TargetFromPi = 0;




// ======================================================
// TMC2208 VIA ACCELSTEPPER
// ======================================================
AccelStepper tmcStepper(AccelStepper::DRIVER, TMC_STEP_PIN, TMC_DIR_PIN);


// Jog tuning
const long  TMC_STEPS_PER_CLICK = 20;
const float TMC_MOVE_SPEED      = 1200.0;
const float TMC_ACCEL           = 800.0;
long tmcTargetPosition = 0;


// Homing tuning
const float        TMC_HOME_SPEED       = 400.0;
const long         TMC_BACKOFF_STEPS    = 1900;
const unsigned long TMC_HOME_TIMEOUT_MS = 1500000;
const bool         TMC_HOME_DIR_POSITIVE = true;




// ======================================================
// L298N TUNING
// ======================================================
const int L298_PWM_VALUE = 255;


// Jog tuning
const int L298_JOG_STEPS_PER_CLICK = 30;
const int L298_JOG_STEP_DELAY_MS   = 6;


// Homing tuning
const int          L298_HOME_STEP_DELAY_MS  = 6;
const long         L298_BACKOFF_HALFSTEPS   = 1200;
const unsigned long L298_HOME_TIMEOUT_MS    = 15000;
const bool         L298_HOME_DIR_FORWARD    = true;


// L298 half-step sequence
int  l298SeqIndex = 0;
long l298Position = 0;


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




// ======================================================
// FORWARD DECLARATIONS
// ======================================================
void readEncoder1();
void readEncoder2();
void readEncoder3();
void handleEnc1Button();
void handleIncomingSerial();
void sendStatusLine();
bool limitPressed(int pin);


void runCalibrationSequence();
bool homeTMCAxis();
bool homeL298Axis();

float degToTMCSteps(float deg);
float degToL298Steps(float deg);


void enableL298Motor();
void disableL298Motor();
void setL298Outputs(int a, int b, int c, int d);
void stepL298(bool forward, int steps, int stepDelayMs);
void streamDuringCalibration();


float getTMCAngle();
float getL298Angle();




// ======================================================
// SETUP
// ======================================================
void setup() {
  Serial.begin(115200);


  // Encoders
  pinMode(ENC1_CLK, INPUT_PULLUP);
  pinMode(ENC1_DT,  INPUT_PULLUP);
  pinMode(ENC1_SW,  INPUT_PULLUP);


  pinMode(ENC2_CLK, INPUT_PULLUP);
  pinMode(ENC2_DT,  INPUT_PULLUP);


  pinMode(ENC3_CLK, INPUT_PULLUP);
  pinMode(ENC3_DT,  INPUT_PULLUP);


  lastEnc2CLK = digitalRead(ENC2_CLK);
  lastEnc3CLK = digitalRead(ENC3_CLK);


  // Limit switches
  pinMode(LIM1_PIN, INPUT_PULLUP);
  pinMode(LIM2_PIN, INPUT_PULLUP);
  pinMode(LIM3_PIN, INPUT_PULLUP);


  // TMC2208
  pinMode(TMC_EN_PIN, OUTPUT);
  digitalWrite(TMC_EN_PIN, LOW);   // LOW = enabled on TMC2208


  tmcStepper.setMaxSpeed(TMC_MOVE_SPEED);
  tmcStepper.setAcceleration(TMC_ACCEL);
  tmcStepper.setCurrentPosition(0);


  // L298N
  pinMode(L298_ENA, OUTPUT);
  pinMode(L298_ENB, OUTPUT);
  pinMode(L298_IN1, OUTPUT);
  pinMode(L298_IN2, OUTPUT);
  pinMode(L298_IN3, OUTPUT);
  pinMode(L298_IN4, OUTPUT);


  disableL298Motor();
}




// ======================================================
// MAIN LOOP
// ======================================================
void loop() {
  handleIncomingSerial();


  // Always update encoder counts
  readEncoder1();
  readEncoder2();
  readEncoder3();


  // Default state: manual jog or Pi position control
  if (systemState == STATE_NORMAL) {
    if (!piControlActive) {
      handleEnc1Button();
      tmcStepper.moveTo(tmcTargetPosition);
    }
    tmcStepper.run();  // always run — moveTo() was already set by command handler when piControlActive
  }


  // Normal streaming or waiting for Pi ack
  if (systemState == STATE_NORMAL || systemState == STATE_WAITING_FOR_ACK) {
    if (millis() - lastPrintTime >= PRINT_INTERVAL_MS) {
      lastPrintTime = millis();
      sendStatusLine();
    }
  }


  // Once Pi sends back "2", start homing both motors
  if (systemState == STATE_CALIBRATING) {
    runCalibrationSequence();
    systemState = STATE_NORMAL;
    control_mode = 0;
  }
}




// ======================================================
// ANGLE GETTERS
// Returns degrees from the homed zero position.
// Positive = forward/CW from home, negative = reverse.
// Values can exceed 360 for multi-turn axes — divide by
// 360 yourself if you need turn count.
// ======================================================
float getTMCAngle() {
  return ((float)tmcStepper.currentPosition() / TMC_STEPS_PER_REV) * 360.0;
}


float getL298Angle() {
  return ((float)l298Position / L298_HALFSTEPS_PER_REV) * 360.0;
}




// ======================================================
// SERIAL STATUS MESSAGE
// Format: mode,enc1,enc2,enc3,lim1,lim2,lim3,tmc_deg,l298_deg
// ======================================================
void sendStatusLine() {
  int lim1State = digitalRead(LIM1_PIN);
  int lim2State = digitalRead(LIM2_PIN);
  int lim3State = digitalRead(LIM3_PIN);


  Serial.print(control_mode);      Serial.print(",");
  Serial.print(enc1Count);         Serial.print(",");
  Serial.print(enc2Count);         Serial.print(",");
  Serial.print(enc3Count);         Serial.print(",");
  Serial.print(lim1State);         Serial.print(",");
  Serial.print(lim2State);         Serial.print(",");
  Serial.print(lim3State);         Serial.print(",");
  Serial.print(getTMCAngle(),  2); Serial.print(",");
  Serial.println(getL298Angle(), 2);
}




// ======================================================
// HANDLE SERIAL COMMANDS FROM PI
// Expected:
//   "2"  -> acknowledge request and start calibration
// ======================================================
void handleIncomingSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();


    if (c == '\n' || c == '\r') {
      if (rxLine.length() > 0) {
        rxLine.trim();


        if (rxLine == "2" && systemState == STATE_WAITING_FOR_ACK) {
          // Encoder button homing ack from Pi
          control_mode = 2;
          systemState  = STATE_CALIBRATING;

        } else if (rxLine == "H") {
          // Direct home command from Pi (no button press needed)
          piControlActive = false;
          control_mode    = 2;
          systemState     = STATE_CALIBRATING;

        } else if (rxLine.startsWith("A,")) {
          // Angle command: "A,<base_deg>,<elbow_deg>"
          int comma2 = rxLine.indexOf(',', 2);
          if (comma2 > 0) {
            float base_deg  = rxLine.substring(2, comma2).toFloat();
            float elbow_deg = rxLine.substring(comma2 + 1).toFloat();

            // TMC2208 (base): non-blocking, AccelStepper runs it in loop
            long tmcTarget = (long)round(degToTMCSteps(base_deg));
            tmcStepper.moveTo(tmcTarget);
            tmcTargetFromPi = tmcTarget;

            // L298N (elbow): blocking step to target
            long l298Target = (long)round(degToL298Steps(elbow_deg));
            long delta = l298Target - l298Position;
            if (delta != 0) {
              enableL298Motor();
              stepL298(delta > 0, (int)abs(delta), L298_JOG_STEP_DELAY_MS);
            }

            piControlActive = true;
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
// ENC1 BUTTON
// Press once -> request homing
// ======================================================
void handleEnc1Button() {
  int reading = digitalRead(ENC1_SW);


  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }


  if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
    if (reading != stableButtonState) {
      stableButtonState = reading;


      // Detect press (HIGH -> LOW)
      if (stableButtonState == LOW) {
        control_mode = 1;
        systemState  = STATE_WAITING_FOR_ACK;
      }
    }
  }


  lastButtonState = reading;
}




// ======================================================
// ENCODER READERS
// ======================================================
void readEncoder1() {
  static int lastState = 0;


  int clk = digitalRead(ENC1_CLK);
  int dt  = digitalRead(ENC1_DT);
  int currentState = (clk << 1) | dt;


  if (currentState != lastState) {
    // Valid CW transitions: 00->01, 01->11, 11->10, 10->00
    if ((lastState == 0 && currentState == 1) ||
        (lastState == 1 && currentState == 3) ||
        (lastState == 3 && currentState == 2) ||
        (lastState == 2 && currentState == 0)) {
      enc1Count++;
      if (systemState == STATE_NORMAL) {
        tmcTargetPosition += TMC_STEPS_PER_CLICK;
      }
    }
    // Valid CCW transitions: 00->10, 10->11, 11->01, 01->00
    else if ((lastState == 0 && currentState == 2) ||
             (lastState == 2 && currentState == 3) ||
             (lastState == 3 && currentState == 1) ||
             (lastState == 1 && currentState == 0)) {
      enc1Count--;
      if (systemState == STATE_NORMAL) {
        tmcTargetPosition -= TMC_STEPS_PER_CLICK;
      }
    }


    lastState = currentState;
  }
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
    if (digitalRead(ENC3_DT) != currentCLK) {
      enc3Count++;


      if (systemState == STATE_NORMAL) {
        enableL298Motor();
        stepL298(true, L298_JOG_STEPS_PER_CLICK, L298_JOG_STEP_DELAY_MS);
      }
    } else {
      enc3Count--;


      if (systemState == STATE_NORMAL) {
        enableL298Motor();
        stepL298(false, L298_JOG_STEPS_PER_CLICK, L298_JOG_STEP_DELAY_MS);
      }
    }
  }


  lastEnc3CLK = currentCLK;
}




// ======================================================
// LIMIT SWITCH HELPER
// INPUT_PULLUP: pressed = LOW, released = HIGH
// ======================================================
bool limitPressed(int pin) {
  return digitalRead(pin) == LOW;
}




// ======================================================
// CALIBRATION MASTER ROUTINE
// While calibrating, Mega continuously sends control_mode=2.
// After completion, both motor positions are zeroed so that
// getTMCAngle() and getL298Angle() read 0.0 at home.
// ======================================================
void runCalibrationSequence() {
  control_mode = 2;


  bool tmcOk  = homeTMCAxis();
  bool l298Ok = homeL298Axis();


  disableL298Motor();


  // Zero both position trackers at the backed-off home position
  tmcStepper.setCurrentPosition(0);
  tmcTargetPosition = 0;
  l298Position      = 0;


  (void)tmcOk;
  (void)l298Ok;
}




// ======================================================
// TMC2208 HOMING
// 1) Drive toward switch at home speed
// 2) On trigger, back off TMC_BACKOFF_STEPS
// 3) That position becomes step 0 / angle 0
// ======================================================
bool homeTMCAxis() {
  unsigned long startTime = millis();


  if (TMC_HOME_DIR_POSITIVE) {
    tmcStepper.setSpeed(TMC_HOME_SPEED);
  } else {
    tmcStepper.setSpeed(-TMC_HOME_SPEED);
  }


  while (!limitPressed(LIM1_PIN)) {
    tmcStepper.runSpeed();
    streamDuringCalibration();


    if (millis() - startTime > TMC_HOME_TIMEOUT_MS) {
      return false;
    }
  }


  long currentPos = tmcStepper.currentPosition();
  long targetPos;


  if (TMC_HOME_DIR_POSITIVE) {
    targetPos = currentPos - TMC_BACKOFF_STEPS;
  } else {
    targetPos = currentPos + TMC_BACKOFF_STEPS;
  }


  tmcStepper.moveTo(targetPos);
  while (tmcStepper.distanceToGo() != 0) {
    tmcStepper.run();
    streamDuringCalibration();
  }


  tmcTargetPosition = tmcStepper.currentPosition();
  return true;
}




// ======================================================
// L298N HOMING
// 1) Drive toward switch
// 2) On trigger, back off L298_BACKOFF_HALFSTEPS
// 3) That position becomes step 0 / angle 0
// ======================================================
bool homeL298Axis() {
  unsigned long startTime = millis();


  enableL298Motor();


  while (!limitPressed(LIM3_PIN)) {
    stepL298(L298_HOME_DIR_FORWARD, 1, L298_HOME_STEP_DELAY_MS);
    streamDuringCalibration();


    if (millis() - startTime > L298_HOME_TIMEOUT_MS) {
      return false;
    }
  }


  stepL298(!L298_HOME_DIR_FORWARD, L298_BACKOFF_HALFSTEPS, L298_HOME_STEP_DELAY_MS);
  return true;
}




// ======================================================
// STREAM STATUS WHILE CALIBRATING
// Keeps sending control_mode=2 with live encoder/angle data
// ======================================================
void streamDuringCalibration() {
  if (millis() - lastPrintTime >= PRINT_INTERVAL_MS) {
    lastPrintTime = millis();
    control_mode  = 2;
    sendStatusLine();
  }


  handleIncomingSerial();
}




// ======================================================
// L298N CONTROL
// ======================================================
void enableL298Motor() {
  analogWrite(L298_ENA, L298_PWM_VALUE);
  analogWrite(L298_ENB, L298_PWM_VALUE);
}


void disableL298Motor() {
  analogWrite(L298_ENA, 0);
  analogWrite(L298_ENB, 0);


  digitalWrite(L298_IN1, LOW);
  digitalWrite(L298_IN2, LOW);
  digitalWrite(L298_IN3, LOW);
  digitalWrite(L298_IN4, LOW);
}


void setL298Outputs(int a, int b, int c, int d) {
  digitalWrite(L298_IN1, a);
  digitalWrite(L298_IN2, b);
  digitalWrite(L298_IN3, c);
  digitalWrite(L298_IN4, d);
}


void stepL298(bool forward, int steps, int stepDelayMs) {
  for (int i = 0; i < steps; i++) {
    if (forward) {
      l298SeqIndex++;
      if (l298SeqIndex > 7) l298SeqIndex = 0;
      l298Position++;
    } else {
      l298SeqIndex--;
      if (l298SeqIndex < 0) l298SeqIndex = 7;
      l298Position--;
    }


    setL298Outputs(
      L298_SEQ[l298SeqIndex][0],
      L298_SEQ[l298SeqIndex][1],
      L298_SEQ[l298SeqIndex][2],
      L298_SEQ[l298SeqIndex][3]
    );


    delay(stepDelayMs);
  }
}