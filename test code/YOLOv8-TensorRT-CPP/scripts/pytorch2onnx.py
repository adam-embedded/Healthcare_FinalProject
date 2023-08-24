from ultralytics import YOLO

# TODO: Specify which model you want to convert
# Model can be downloaded from https://github.com/ultralytics/ultralytics
model = YOLO("~/development/yolov8m.pt")
model.fuse()
model.info(verbose=False)  # Print model information
model.export(format="engine", opset=12) # Export the model to onnx using opset 12