import threading
import tkinter

import cv2
import weaviate

from database.setup import setup_all
from database.minio_store import create_client as minio_create_client
from facialRecognition.faceProcessing import FaceProcessing
from facialRecognition.faceTracker import FaceTracker



def cv2_loop(face_tracker: FaceTracker, stop_event: threading.Event, root: tkinter.Tk) -> None:
    WINDOW_NAME = "Face Tracker (Press Shift+Q to close)"
    capture = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, face_tracker.mouse_callback)

    try:
        while not stop_event.is_set():
            returned, frame = capture.read()
            if not returned:
                break
            frame = face_tracker.update_frame(frame)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("Q"):
                stop_event.set()
                break
    finally:
        face_tracker.stop()
        capture.release()
        cv2.destroyAllWindows()
        root.after(0, root.quit)


if __name__ == "__main__":
    fp = FaceProcessing()
    wv_client = weaviate.connect_to_local()
    mn_client = minio_create_client()
    setup_all(wv_client)

    root = tkinter.Tk()
    root.withdraw()

    face_tracker = FaceTracker(fp, root, wv_client, mn_client)
    stop_event = threading.Event()
    cv_thread = threading.Thread(target=cv2_loop, args=(face_tracker, stop_event, root), daemon=True)
    cv_thread.start()

    try:
        root.mainloop()
    finally:
        stop_event.set()
        wv_client.close()