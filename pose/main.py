import uvicorn
import socket
import numpy as np
import base64
import datetime

from loguru import logger

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Optional, List, Tuple
from ultralytics import YOLO


today = datetime.datetime.today()
year = today.year
month = today.month
day = today.day

logger.add(f"logs/{year}{str(month).zfill(2)}{str(day).zfill(2)}_pose_log.log", rotation="00:00")


IP = socket.gethostbyname(socket.gethostname())
APP = FastAPI()

model_list = {
    "small": "yolov8s-pose.pt",
    "medium": "yolov8m-pose.pt",
    "large": "yolov8l-pose.pt",
    "extra": "yolov8x-pose.pt",
}


class Query(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[Union[List[int], int]] = None
    model: str
    base_color: Optional[Union[List[int], Tuple[int]]] = None


@APP.get("/")
async def init():
    return {IP: "pose"}


@APP.post("/pose")
async def pose(inp: Query):
    model = YOLO(f"models/{model_list[inp.model]}")

    results = {"bbox": [], "kpts": []}

    for img_info in inp.images:
        image = bytes(img_info["image"], "utf-8")
        width = img_info["width"]
        height = img_info["height"]

        img_data = base64.b64decode(image)
        data_bytes = np.fromstring(img_data, dtype=np.uint8)
        img = data_bytes.reshape((height, width, 3))

        res = model(img, classes=inp.classes)[0]
        bbox = res.boxes.data
        kpts = res.keypoints.data

        results["bbox"].append(bbox.cpu().numpy().tolist())
        results["kpts"].append(kpts.cpu().numpy().tolist())

    return results


if __name__ == "__main__":
    uvicorn.run("main:APP", host=IP)
