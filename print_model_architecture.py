from ultralytics import YOLO
from torchinfo import summary

model = YOLO("yolov8s.pt").model  # structure interne (pas ton best.pt)
summary(model, input_size=(1, 3, 640, 640))