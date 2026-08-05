from __future__ import annotations

import os  
from pathlib import Path
import yaml
from types import SimpleNamespace
import json

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "src" / "resources" / "configs"

class ConfigLoader:

    @staticmethod
    def load(name: str):
        path = CONFIG_DIR / name

        with path.open("r", encoding="utf-8") as f:
            raw_text = f.read()
            expanded_text = os.path.expandvars(raw_text)
            data_dict = yaml.safe_load(expanded_text)
            data_obj = json.loads(json.dumps(data_dict), object_hook=lambda d: SimpleNamespace(**d))
            return data_obj

class Config:   
    def __init__(self):
        self.camera = ConfigLoader.load("camera.yaml")
        self.detection = ConfigLoader.load("detection.yaml")
        self.plate = ConfigLoader.load("plate.yaml")
