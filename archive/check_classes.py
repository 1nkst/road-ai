from ultralytics import YOLO
model = YOLO("models/best.pt")
print("Class names:", model.names)
