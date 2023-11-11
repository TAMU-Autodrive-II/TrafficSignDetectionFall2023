from ultralytics import YOLO
from PIL import Image
import os
from google.cloud import vision


#
# This script will label a bunch of images using a model, so you can see it in action.
#

def extract_text_from_bbox(image, bbox):
    roi = image.crop(bbox)
    roi.save(f'C:/Users/harry/Desktop/results/result{bbox}.png')

    return 'hi'


if __name__ == '__main__':

    weightsPath = 'C:/Users/harry/PycharmProjects/TrafficSignDetection/runs/detect/train7/weights/best.pt'
    model = YOLO(weightsPath)

    imagesDir = 'C:/Users/harry/Desktop/LISA road signs.v1i.yolov8/test/images/'
    for filename in os.listdir(imagesDir):
        fullPath = os.path.join(imagesDir, filename)
        results = model(fullPath)
        for result in results:
            im = Image.open(fullPath)
            for i, c in enumerate(result.boxes.cls):
                class_name = model.names[int(c)]

                if 'speed' in class_name.lower():
                    bounding_box = result.boxes.xyxy[i]  # get box coordinates in (top, left, bottom, right) format
                    left, top, right, bottom = bounding_box
                    bb_fixed = (int(left), int(top), int(right), int(bottom))
                    text = extract_text_from_bbox(im, bb_fixed)
                    print(f"Traffic sign text: '{text}'")