import cv2
import base64
import datetime

import uvicorn
import socket
import numpy as np

from PIL import Image
from io import BytesIO

from loguru import logger

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from ultralytics import YOLO


today = datetime.datetime.today()
year = today.year
month = today.month
day = today.day

logger.add(f"logs/{year}{str(month).zfill(2)}{str(day).zfill(2)}_segment_log.log", rotation="00:00")


IP = socket.gethostbyname(socket.gethostname())
app = FastAPI()

model_list = {
    "small": "yolov8s-seg.pt",
    "medium": "yolov8m-seg.pt",
    "large": "yolov8l-seg.pt",
    "extra": "yolov8x-seg.pt",
}


class Input(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[List[int]] = None
    model: str
    base_color: Optional[tuple] = None
    conf: Optional[float] = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(exc.body)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body})
    )


@app.post("/segmentation")
async def segmentation(inp: Input):
    model = YOLO(f"models/{model_list[inp.model]}")

    results = {}

    for img_info in inp.images:
        image = bytes(img_info["image"], "utf-8")

        decoded_img = base64.b64decode(image)
        bytes_img = BytesIO(decoded_img)
        image = Image.open(bytes_img)
        src = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        
        res = model(src, classes=inp.classes, conf=inp.conf)[0]
        bbox = res.boxes.data.cpu().numpy().tolist()
        masks = [seg.tolist() for seg in res.masks.xy]

        results.update({img_info["name"]: {"bbox": bbox, "masks": masks}})

    return results


if __name__ == "__main__":
    uvicorn.run("main:app", host=IP)
