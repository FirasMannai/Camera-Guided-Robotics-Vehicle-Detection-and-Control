from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=True,  # Change this to True
    nms=True      # Change this to True for end-to-end model
)
print("? Export done.")