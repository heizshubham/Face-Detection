import time
import datetime
import logging
from threading import Thread

import cv2 as cv
from deepface import DeepFace


class FaceDetection:
    def __init__(self, source_image_path, source_name):
        self.source_image = cv.imread(source_image_path)
        self.source_name = source_name
        self.face_match = False
        self.face_data = None
        self.check_flag = True
        self.cap = cv.VideoCapture(0)
        self.logger = logging.getLogger("FaceDetection")
        logging.basicConfig(level=logging.INFO)

    def check_face(self, frame):
        """ Check for the face and compare it with the source image. """
        try:
            result = DeepFace.verify(frame, self.source_image)
            self.face_match = result['verified']
            self.face_data = result["facial_areas"]
        except Exception as e:
            self.logger.error(f"Error in face verification: {e}")
            self.face_match = False
        finally:
            self.check_flag = True

    def run(self):
        timer = time.time()
        image_counter = 0
        fps = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.check_flag:
                self.logger.info(f"Thread Started! {image_counter}")
                Thread(target=self.check_face, args=(frame.copy(),)).start()
                self.check_flag = False

            if self.face_match:
                frame = self.draw_face_match(frame)
            else:
                frame = cv.putText(
                    img=frame,
                    org=(0, 690),
                    text="No Match",
                    color=(0, 255, 0),
                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                    thickness=3,
                    fontScale=1.5,
                )
                frame = cv.rectangle(frame, (0, 698), (225, 698), color=(0, 255, 0), thickness=2)

            frame = self.add_fps_and_timestamp(frame, fps)
            frame = cv.resize(frame, (920, 640))
            cv.imshow("video", frame)

            image_counter += 1
            now = time.time()
            if now - timer > 1:
                fps = image_counter / (now - timer)
                image_counter = 0
                timer = time.time()

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv.destroyAllWindows()

    def draw_face_match(self, frame):
        frame = cv.putText(
            img=frame,
            org=(0, 690),
            text="Face Match",
            color=(0, 0, 255),
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            thickness=3,
            fontScale=1.5,
        )

        frame = cv.rectangle(frame, (0, 698), (300, 698), color=(0, 0, 255), thickness=2)

        x = "img1"
        start_point = (self.face_data[x]['x'], self.face_data[x]['y'])
        end_point = (self.face_data[x]['x'] + self.face_data[x]['w'], self.face_data[x]['y'] + self.face_data[x]['h'])

        # Draw a rectangle around the face in the frame
        frame = cv.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=3)

        # Add person name to frame
        frame = cv.putText(
            img=frame,
            org=(end_point[0] - 100, end_point[1] + 20),
            text=self.source_name,
            color=(0, 0, 255),
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            thickness=2,
            fontScale=0.6,
        )

        frame = cv.putText(
            img=frame,
            org=(end_point[0] - 100, end_point[1] + 40),
            text=f"x:{start_point[0]},y:{start_point[1]}",
            color=(0, 0, 255),
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            thickness=1,
            fontScale=0.6,
        )
        return frame

    def add_fps_and_timestamp(self, frame, fps):
        frame = cv.putText(
            img=frame,
            org=(0, 20),
            text=f"FPS: {fps:.2f}",
            color=(0, 0, 255),
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            thickness=1,
            fontScale=0.6,
        )

        frame = cv.putText(
            img=frame,
            org=(0, 715),
            text=f"{datetime.datetime.utcnow()}",
            color=(0, 0, 255),
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            thickness=1,
            fontScale=0.6,
        )
        return frame


if __name__ == "__main__":
    detector = FaceDetection("source.png", "Will Smith")
    detector.run()
