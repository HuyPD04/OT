from ultralytics import YOLO
import shutil
from pathlib import Path

def export(model_name: str, output_path: str = "src/resources/weights", format: str = "onnx"):
    Path(output_path).mkdir(exist_ok=True)

    model = YOLO(model_name)
    model.export(format=format)
    name = model_name.replace(model_name.split(".")[-1], format)
    shutil.move(name, Path(output_path) / name)

if __name__ == "__main__":
    export(model_name="yolo11s.pt")