from ultralytics import YOLO
import shutil
from pathlib import Path

def export(model_name: str, output_path: str = "src/resources/weights/", format: str = "onnx"):
    Path(output_path).mkdir(exist_ok=True)
    model = YOLO(model_name)
    export = model.export(format=format, int8=True)
    shutil.move(export, Path(output_path))

if __name__ == "__main__":
    export(model_name="plate.pt")