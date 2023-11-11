# input camera data, can use lidar but probably don't need to
# output type of sign, speed limit if applicable
# use another model to recognize digits
# try to keep model lightweight
# YOLOv8 prolly best because its state of the art
# dataset has 40 classes, feel free to trim unnecessary ones

#
# This script will train a new model from scratch.
#
from ultralytics import YOLO

if __name__ == '__main__':
    # create a model
    model = YOLO('yolov8n.yaml', task='detect')

    # Train the model
    dataDir = 'C:/Users/harry/Desktop/LISA road signs.v1i.yolov8/'
    # see arguments here: https://docs.ultralytics.com/modes/train/#arguments
    results = model.train(data=dataDir + 'data.yaml', save_period=1, device='cuda')
    model.export(path=dataDir + 'result/weights.pt')