# input camera data, can use lidar but probably don't need to
# output type of sign, speed limit if applicable
# use another model to recognize digits
# try to keep model lightweight
# YOLOv8 prolly best because its state of the art
# dataset has 40 classes, feel free to trim unnecessary ones

from ultralytics import YOLO

#
# This script will resume training an already existing model.
#
if __name__ == '__main__':
    # load model
    weightsPath = 'C:/Users/harry/PycharmProjects/TrafficSignDetection/runs/detect/train7/weights/epoch99.pt'
    model = YOLO(weightsPath)

    # resume training the model
    dataDir = 'C:/Users/harry/Desktop/LISA road signs.v1i.yolov8/'
    # see arguments here: https://docs.ultralytics.com/modes/train/#arguments
    results = model.train(data=dataDir + 'data.yaml', epochs=1, resume=True, save_period=1, device='cuda')
    model.export(dataDir + 'result/weights.pt')