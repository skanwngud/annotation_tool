import uvicorn
import socket
import numpy as np
import cv2
import base64

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Optional, List, Tuple

APP = FastAPI()

IP = socket.gethostbyname(socket.gethostname())


class Input(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[Union[List[int], int]] = None
    model: Optional[str] = None
    base_color: Union[List[int], Tuple[int]]


@APP.get("/")
async def init():
    return {IP: "clustering"}

@APP.post("/clustering")
async def clustering(inp: Input):
    base_color = inp.base_color
    
    results = {"regions": []}
    
    for img_info in inp.images:
        image = bytes(img_info["image"], "utf-8")
        width = img_info["width"]
        height = img_info["height"]
        
        img_data = base64.b64decode(image)
        data_bytes = np.fromstring(img_data)
        img = data_bytes.reshape((height, width, 3))
        
    return results


if __name__ == "__main__":
    uvicorn.run("main:APP", host=IP)