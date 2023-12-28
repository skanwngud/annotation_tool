import cv2
import base64

import uvicorn
import socket
import numpy as np
import base64

from PIL import Image
from io import BytesIO

from collections import defaultdict

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Optional, List, Tuple
from ultralytics import YOLO

APP = FastAPI()

IP = socket.gethostbyname(socket.gethostname())

model_list = {
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "extra": "yolov8x.pt",
}


class Input(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[Union[List[int], int]] = None
    model: str
    base_color: Optional[Union[List[int], Tuple[int]]] = None


@APP.get("/")
async def init():
    return {IP: "detect"}


@APP.post("/detect")
async def detect(inp: Input):
    """
    Args:
        inp (Input): 
        {
            "images": [
                {
                    "name": str,
                    "image": base64,
                    "shape": Union[List[int], Tuple[int]]
                }
            ],
            "types": "detect",
            "classes": List[int],
            "model": str,
            "base_color": None
        }

    Returns:
        _type_: _description_
    """
    model = YOLO(model_list[inp.model])

    results = defaultdict(list)

    for img_info in inp.images:
        image = bytes(img_info["image"], "utf-8")
        width, height, channel = img_info["shape"]
        
        decoded_img = base64.b64decode(image)
        bytes_img = BytesIO(decoded_img)
        image = Image.open(bytes_img)
        src = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        
        res = model(src, classes=inp.classes)[0].boxes.data.cpu().numpy().tolist()

        results[img_info["name"]].append(res)

    return results


if __name__ == "__main__":
    uvicorn.run("main:APP", host=IP)
