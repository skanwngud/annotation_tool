import uvicorn
import socket
import base64
import cv2
import numpy as np
import requests
import json
import os

from PIL import Image
from io import BytesIO

from ast import literal_eval
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Union, Optional, List, Tuple

from loguru import logger

from utils import check, scan_ip, get_servers


IP = socket.gethostbyname(socket.gethostname())
logger.info(f"server's IP is {IP}")

templates = Jinja2Templates(directory="server/templates") # html 파일 렌더링을 위한 jinja2 템플릿 초기화
APP = FastAPI()
APP.mount("/server/static", StaticFiles(directory="server/static"), name="static")  # 정적파일 제공 설정

UPLOAD_FOLDER = "server/static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ip_list = scan_ip(IP, 8000)
DET, SEG, POS, CLU = get_servers(ip_list)

logger.info(f"""
---------------------Server Information---------------------
            Detection Server is {DET}
            Segmentation Server is {SEG}
            PoseEstimation Server is {POS}
            Clustering Server is {CLU}
------------------------------------------------------------
            """)

class Input(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[Union[List[int], int]] = None
    model: Optional[str] = None
    base_color: Optional[Union[List[int], Tuple[int]]] = None
    conf: Optional[float] = None


@APP.get("/")
async def read_home(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

import shutil

@APP.post("/upload")
async def upload(file: UploadFile = File(...), task: str = Form(...), size: str = Form(...), classes: str = Form(...)):
    # return "hello"
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}


@APP.post("/detect")
async def detect(inp: Input):
    model_type, classes = check(inp, "detect")
    
    for img_info in inp.images:
        image = bytes(img_info["image"], "utf-8")
        
        decoded_img = base64.b64decode(image)
        bytes_img = BytesIO(decoded_img)
        image = Image.open(bytes_img)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        
    payload = {
        "images": inp.images,
        "types": model_type,
        "classes": classes,
        "model": inp.model
    }
        
    resp = requests.post(
        url = f"http://{DET}:8000/detect",
        json = payload,
        headers = {"Content-Type": "application/json"}
    )
    
    resp = literal_eval(resp.content.decode("utf-8"))
    
    with open("./detect_result.json", "w") as f:
        json.dump(resp, f, indent=4)
        
    return resp


@APP.post("/inference")
async def inference(inp: Input):
    model_type, classes = check(inp)

    for img_info in inp.images:
        name = img_info["name"]
        image = bytes(img_info["image"], "utf-8")
        width, height, channel = img_info["shape"]

        decoded_img = base64.b64decode(image)
        bytes_img = BytesIO(decoded_img)
        image = Image.open(bytes_img)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"./{os.path.basename(name)}", image)

    payload = {
        "images": inp.images,
        "types": model_type,
        "classes": classes,
        "model": inp.model,
    }

    if model_type == "detect":
        resp = requests.post(url=f"http://{DET}:8000/detect", json=payload)

    elif model_type == "pose":
        resp = requests.post(url=f"http://{POS}:8000/pose", json=payload)

    elif model_type == "segmentation":
        resp = requests.post(url=f"http://{SEG}:8000/segmentation", json=payload)

    elif model_type == "clustering":
        resp = requests.post(url=f"http://{CLU}:8000/clustering", json=payload)

    resp = literal_eval(resp.content.decode("utf-8"))
    
    with open("./result.json", "w") as f:
        json.dump(resp, f, indent=4)

    return resp


if __name__ == "__main__":
    uvicorn.run(APP, host=IP)
