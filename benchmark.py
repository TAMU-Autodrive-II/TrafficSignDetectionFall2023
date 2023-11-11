from ultralytics import YOLO
import os
import time

if __name__ == '__main__':
    weightsPath = 'C:/Users/harry/PycharmProjects/TrafficSignDetection/runs/detect/train7/weights/best.pt'
    model = YOLO(weightsPath)

    imagesDir = 'C:/Users/harry/Desktop/LISA road signs.v1i.yolov8/test/images/'
    filecount = 0

    start_time = time.time()

    for filename in os.listdir(imagesDir):
        fullPath = os.path.join(imagesDir, filename)
        results = model(fullPath)
        filecount += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total time taken: {elapsed_time} seconds for {filecount} images")
    print(f"{elapsed_time / filecount } seconds/image")
    print(f"{filecount / elapsed_time } images/second")
