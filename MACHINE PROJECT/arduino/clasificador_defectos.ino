// arduino_sketch.ino
#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27,16,2);
Servo servo;

const int pinIRentrada = 6;
const int pinIRexpulsion = 7;
const int pinFaja = 12;
const int pinServo = 9;

void setup(){
  Serial.begin(115200);
  pinMode(pinIRentrada, INPUT_PULLUP);   // si sensor da LOW cuando detecta
  pinMode(pinIRexpulsion, INPUT_PULLUP);
  pinMode(pinFaja, OUTPUT);
  digitalWrite(pinFaja, HIGH); // faja ON
  servo.attach(pinServo);
  servo.write(90); // neutral
  lcd.init(); lcd.backlight();
  lcd.setCursor(0,0); lcd.print("Sistema ON");
  delay(800);
  lcd.clear();
}

void loop(){
  // Si IR entrada detecta objeto (assume LOW when present)
  if (digitalRead(pinIRentrada) == LOW) {
    // debouncing
    delay(80);
    if (digitalRead(pinIRentrada) != LOW) return;

    // mandar S y esperar clasificacion
    Serial.println("S");
    lcd.clear(); lcd.setCursor(0,0); lcd.print("Objeto detectado");
    lcd.setCursor(0,1); lcd.print("Esperando PC...");

    unsigned long start = millis();
    String respuesta = "";
    const unsigned long TIMEOUT = 4000; // ms

    while (millis() - start < TIMEOUT) {
      if (Serial.available()>0) {
        respuesta = Serial.readStringUntil('\n');
        respuesta.trim();
        break;
      }
    }
    if (respuesta.length() == 0) {
      // timeout => dejar pasar
      lcd.clear(); lcd.setCursor(0,0); lcd.print("Timeout -> OK");
      digitalWrite(pinFaja, HIGH);
      delay(300);
      lcd.clear();
      return;
    }

    if (respuesta.equalsIgnoreCase("DEFECTO") || respuesta.equalsIgnoreCase("BAD")) {
      lcd.clear(); lcd.setCursor(0,0); lcd.print("DEFECTUOSO!");
      // detener faja
      digitalWrite(pinFaja, LOW);
      // esperar al sensor de expulsion o un tiempo fijo
      unsigned long waitStart = millis();
      const unsigned long MAXWAIT = 4000; // ms
      bool arrived = false;
      while (millis() - waitStart < MAXWAIT) {
        if (digitalRead(pinIRexpulsion) == LOW) { arrived = true; break; }
        delay(10);
      }
      // activar servo 360° (giro temporal) o servo 180° (angulo)
      // si servo es 360, girar por tiempo; si es 180, set angle
      servo.write(0); // empuja (ajusta según tu mecánica)
      delay(700);     // tiempo de empuje
      servo.write(90); // reposo
      delay(300);
      // reanudar faja
      digitalWrite(pinFaja, HIGH);
      lcd.clear(); lcd.setCursor(0,0); lcd.print("Expulsado");
      delay(500);
      lcd.clear();
    } else {
      // OK
      lcd.clear(); lcd.setCursor(0,0); lcd.print("OK - Continua");
      digitalWrite(pinFaja, HIGH);
      delay(200);
      lcd.clear();
    }
  }
  // Loop rapido
  delay(50);
}
