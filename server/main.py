import uvicorn
import socket
import base64
import cv2
import numpy as np
import requests
import json
import os

from ast import literal_eval
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Optional, List

from utils import check, scan_ip


DET, SEG, POS, CLU = None, None, None, None
IP = socket.gethostbyname(socket.gethostname())
APP = FastAPI()


ip_list = scan_ip(IP)
for val in ip_list:
    ip = list(val.keys())[0]
    task = list(val.values())[0]

    if task == "detect":
        DET = ip

    elif task == "segmentation":
        SEG = ip

    elif task == "pose":
        POS = ip

    elif task == "clustering":
        CLU = ip


class Input(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[Union[List[int], int]] = None
    model: str


@APP.post("/inference")
async def inference(inp: Input):
    model_type, classes = check(inp)

    for img_info in inp.images:
        name = img_info["name"]
        image = bytes(img_info["image"], "utf-8")
        width = img_info["width"]
        height = img_info["height"]

        img_data = base64.b64decode(image)
        data_bytes = np.fromstring(img_data, dtype=np.uint8)
        img = data_bytes.reshape((height, width, 3))

        cv2.imwrite(f"./{os.path.basename(name)}", img)

    results = {
        "images": inp.images,
        "types": model_type,
        "classes": classes,
        "model": inp.model,
    }

    if model_type == "detect":
        resp = requests.post(url=f"http://{DET}:8000/detect", data=json.dumps(results))

        resp = literal_eval(resp.content.decode("utf-8"))
        results.update(resp)

    elif model_type == "pose":
        resp = requests.post(url=f"http://{POS}:8000/pose", data=json.dumps(results))

        resp = literal_eval(resp.content.decode("utf-8"))
        results.update(resp)

    elif model_type == "segmentation":
        resp = requests.post(
            url=f"http://{SEG}:8000/segmentation", data=json.dumps(results)
        )

        resp = literal_eval(resp.content.decode("utf-8"))
        results.update(resp)

    elif model_type == "clustering":
        resp = requests.post(
            url=f"http://{CLU}:8000/clustering", data=json.dumps(results)
        )

        resp = literal_eval(resp.content.decode("utf-8"))
        results.update(resp)

    return resp


if __name__ == "__main__":
    uvicorn.run("main:APP", host=IP)
