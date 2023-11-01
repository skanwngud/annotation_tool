import uvicorn
import socket
import numpy as np
import base64

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Optional, List
from ultralytics import YOLO

app = FastAPI()

IP = socket.gethostbyname(socket.gethostname())

model_list = {
    "small": YOLO("yolov8s.pt"),
    "medium": YOLO("yolov8m.pt"),
    "large": YOLO("yolov8l.pt"),
    "extra": YOLO("yolov8x.pt")
}

class Input(BaseModel):
    images: Union[bytes, List[bytes]]
    types: str
    classes: Optional[Union[int], int] = None
    model: str


@app.get("/")
async def init():
    return {IP: "detect"}


@app.post("/detect")
async def detect(inp: Input):
    model = model_list[inp.model]
    images = inp.images if isinstance(inp.images, list) else [inp.images]
    
    results = {
        "bbox": []
    }
    
    for bytes_string in images:
        img_data = base64.b64decode(bytes_string)
        data_bytes = np.fromstring(img_data, dtype=np.uint8)
        img = data_bytes.reshape((360, 640, 3))
        res = model(img, classes=inp.classes)[0].boxes.data.cpu().numpy().tolist()
        
        results["bbox"].append(res)
        
    return results


if __name__ == "__main__":
    uvicorn.run("main:app", host=IP)