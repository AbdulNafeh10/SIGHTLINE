
import cv2
import time


class Camera:
    def __init__(self, index=3):
        # Plugcam virtual camera (DirectShow)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open Plugcam camera")

        # Request 1440p resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

        print("SIGHTLINE Plugcam connected")

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(f"Camera resolution: {int(width)} x {int(height)}")

        # Allow Plugcam to initialize
        print("Waiting for camera to initialize...")
        time.sleep(5)

    def capture(self, filename):
        frame = None

        # Read several frames to get the latest image
        for _ in range(15):
            success, current_frame = self.cap.read()

            if success and current_frame is not None:
                frame = current_frame

            time.sleep(0.1)

        if frame is None:
            raise RuntimeError("Failed to capture image")

        if not cv2.imwrite(filename, frame):
            raise RuntimeError("Failed to save image")

        print(f"Saved: {filename}")

    def close(self):
        self.cap.release()
        print("Camera disconnected")


if __name__ == "__main__":
    camera = Camera(index=3)

    try:
        camera.capture("test_camera.jpg")
    finally:
        camera.close()
