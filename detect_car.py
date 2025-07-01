import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import argparse


class MovingAverage:
    def __init__(self, window_size=30, threshold=0.5):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold  

    def update(self, current_detection):
        self.window.append(current_detection)
        positives = sum(self.window)  
        return positives >= (self.threshold * len(self.window))


def detect_car(video_path):
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Не удалось открыть видео(")
        return
    
    frame_num = 0
    tracker = MovingAverage()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = frame.copy()

        cnt = np.array([(0, 1050), (0, 1512), (1575, 1512), (1670, 350), (1250, 370)])

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        img_cropped = masked_img[350:, :1670]

        
        if frame_num % 3 == 0:
            infer = model(img_cropped, verbose=False)
            result = 0
            for c in infer[0].boxes.cls:
                if c in classes:
                    result = 1
                    break
            text = "Car detected" if tracker.update(result) else "No car"


        cv2.putText(
            frame, 
            text, 
            (20, 120), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            4, 
            (0, 0, 255), 
            10, 
            cv2.LINE_AA
        )

        frame = cv2.resize(frame, (1344, 756))
        cv2.imshow("car", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="cvtest.avi", help="Video path")
    args = parser.parse_args()
    model = YOLO('yolo11n.pt')
    classes = [2., 3., 5., 7.]
    detect_car(args.path)