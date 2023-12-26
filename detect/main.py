import uvicorn
import socket
import numpy as np
import base64

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Optional, List, Tuple
from ultralytics import YOLO

APP = FastAPI()

IP = socket.gethostbyname(socket.gethostname())

model_list = {
    "small": YOLO("yolov8s.pt"),
    "medium": YOLO("yolov8m.pt"),
    "large": YOLO("yolov8l.pt"),
    "extra": YOLO("yolov8x.pt"),
}


class Input(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[Union[List[int], int]] = None
    model: str
    base_color = Optional[Union[List[int], Tuple[int]]] = None


@APP.get("/")
async def init():
    return {IP: "detect"}


@APP.post("/detect")
async def detect(inp: Input):
    model = model_list[inp.model]

    results = {"bbox": []}

    for img_info in inp.images:
        image = bytes(img_info["image"], "utf-8")
        width = img_info["width"]
        height = img_info["height"]

        img_data = base64.b64decode(image)
        data_bytes = np.fromstring(img_data, dtype=np.uint8)
        img = data_bytes.reshape((360, 640, 3))
        res = model(img, classes=inp.classes)[0].boxes.data.cpu().numpy().tolist()

        results["bbox"].append(res)

    return results


if __name__ == "__main__":
    uvicorn.run("main:APP", host=IP)
