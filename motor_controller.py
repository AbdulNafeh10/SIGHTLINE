"""SIGHTLINE Arduino serial control: stepper rotation and LCD messages.

Compatible with Sightline_Controller firmware:
  45                  -> rotate 45 degrees; reply 'DONE: 45'
  STATUS:ANALYZING... -> update LCD; reply 'STATUS OK'

Positions are commanded estimates; there is no angular encoder.
"""

import os
import time

import serial

PORT = os.environ.get("SIGHTLINE_PORT", "COM10")
BAUD_RATE = 9600


def connect(port=PORT):
    """Open Arduino's serial port and allow its automatic reset to finish."""
    arduino = serial.Serial(port, BAUD_RATE, timeout=1, write_timeout=5)
    try:
        time.sleep(2)
        # Bound boot draining even if a faulty device continuously emits text.
        deadline = time.monotonic() + 5
        while arduino.in_waiting:
            if time.monotonic() >= deadline:
                raise TimeoutError("Arduino boot output did not settle")
            line = arduino.readline().decode(errors="replace").strip()
            if line:
                print(f"Arduino: {line}")
    except BaseException:
        arduino.close()
        raise
    return arduino


def _wait_for_reply(arduino, accepted, timeout):
    """Read serial replies until the expected acknowledgment arrives."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = arduino.readline().decode(errors="replace").strip()
        if not line:
            continue
        print(f"Arduino: {line}")
        if line.startswith("ERROR:"):
            raise RuntimeError(line)
        if accepted(line):
            return line
    raise TimeoutError("No matching Arduino reply received. Check USB/COM port and firmware.")


def rotate(arduino, angle):
    """Rotate by signed degrees, valid range -180 through +180."""
    if isinstance(angle, bool) or not isinstance(angle, int) or not -180 <= angle <= 180:
        raise ValueError("Rotation must be an integer from -180 to 180 degrees")

    arduino.write(f"{angle}\n".encode("ascii"))
    arduino.flush()
    expected = f"DONE: {angle}"
    _wait_for_reply(arduino, lambda reply: reply == expected, timeout=20)
    return angle


def rotate_to(arduino, current_angle, target_angle):
    """Rotate via the shortest arc to an absolute *commanded* angle in [0, 360)."""
    if not isinstance(current_angle, int) or not isinstance(target_angle, int):
        raise TypeError("Angles must be integers")
    current_angle %= 360
    target_angle %= 360
    delta = (target_angle - current_angle + 180) % 360 - 180
    if delta != 0:
        rotate(arduino, delta)
    return target_angle


def set_status(arduino, message):
    """Show an ASCII status message on LCD row 1 (max 16 characters)."""
    if not isinstance(message, str) or not 1 <= len(message) <= 16 or not message.isascii():
        raise ValueError("LCD status must contain 1-16 ASCII characters")
    if "\n" in message or "\r" in message:
        raise ValueError("LCD status may not contain line breaks")

    arduino.write(f"STATUS:{message}\n".encode("ascii"))
    arduino.flush()
    _wait_for_reply(arduino, lambda reply: reply == "STATUS OK", timeout=5)
    print(f"LCD: {message}")
