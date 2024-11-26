from ultralytics import YOLO


model = YOLO("yolo11m-seg.pt")

model.train(data = "C://Users//sedtt//brain_tumor//brain_tumor_v8//data.yaml", imgsz = 640, device=0,
    batch = 8, epochs = 10, workers = 0)

   