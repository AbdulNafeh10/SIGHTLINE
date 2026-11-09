#include <Stepper.h>
#include <LiquidCrystal.h>

// Stepper motor
const int stepsPerRevolution = 2048;
Stepper motor(stepsPerRevolution, 8, 10, 9, 11);

// LCD: RS, E, D4, D5, D6, D7
LiquidCrystal lcd(22, 23, 24, 25, 26, 27);

int currentAngle = 0;

void showStatus(String status) {
  lcd.setCursor(0, 0);
  lcd.print("                ");
  lcd.setCursor(0, 0);
  lcd.print(status.substring(0, 16));
}

void showAngle() {
  char line[17];

  snprintf(
    line,
    sizeof(line),
    "ANGLE: %03d DEG",
    currentAngle
  );

  lcd.setCursor(0, 1);
  lcd.print("                ");
  lcd.setCursor(0, 1);
  lcd.print(line);
}

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(100);

  motor.setSpeed(15);
  lcd.begin(16, 2);

  showStatus("SIGHTLINE READY");
  showAngle();

  Serial.println("SIGHTLINE READY");
}

void loop() {
  if (Serial.available() == 0) return;

  String command = Serial.readStringUntil('\n');
  command.trim();

  if (command.length() == 0) return;

  // Receive LCD status updates from Python
  if (command.startsWith("STATUS:")) {
    String status = command.substring(7);

    showStatus(status);
    Serial.println("STATUS OK");
    return;
  }

  // Handle motor rotation
  int angle = command.toInt();

  if (angle < -180 || angle > 180) {
    Serial.println("ERROR: Angle out of range");
    return;
  }

  showStatus("ROTATING...");

  long steps = (long)angle * stepsPerRevolution / 360;
  motor.step(steps);

  currentAngle = (currentAngle + angle + 360) % 360;

  showAngle();
  showStatus("SIGHTLINE READY");

  Serial.print("DONE: ");
  Serial.println(angle);
}