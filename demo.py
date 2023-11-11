from ultralytics import YOLO
from PIL import Image
import os
#
# This script will label a bunch of images using a model, so you can see it in action.
#
if __name__ == '__main__':

    weightsPath = 'C:/Users/harry/PycharmProjects/TrafficSignDetection/runs/detect/train7/weights/best.pt'
    model = YOLO(weightsPath)
    
    imagesDir = 'C:/Users/harry/Desktop/LISA road signs.v1i.yolov8/test/images/'
    for filename in os.listdir(imagesDir):
        fullPath = os.path.join(imagesDir, filename)
        results = model(fullPath)
        for result in results:
            im_array = result.plot()  # plot a BGR numpy array of predictions
            im_annotated = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            saveAt = os.path.join('C:/Users/harry/Desktop/LISA road signs.v1i.yolov8/result/labledimages/', filename)
            im_annotated.save(saveAt)
