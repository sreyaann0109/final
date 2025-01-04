from ultralytics import YOLO

# Load a pretrained YOLOv8s model
model = YOLO('yolov8s.pt')

# Train and validate the mode
train_results = model.train(data='data.yaml', imgsz=640, batch=32, epochs=10, project='runs', name='train', save=True)