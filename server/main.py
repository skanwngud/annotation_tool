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

templates = Jinja2Templates(directory="templates") # html 파일 렌더링을 위한 jinja2 템플릿 초기화
APP = FastAPI()

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

APP.mount("/static", StaticFiles(directory="static"), name="static")  # 정적파일 제공 설정

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
    classes: Optional[List[int]] = None
    model: Optional[str] = None
    base_color: Optional[Union[List[int], Tuple[int]]] = None
    conf: Optional[float] = None


@APP.get("/")
async def read_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@APP.post("/inference")
async def inference(
    request: Request,
    files: List[UploadFile] = File(...),
    task: str = Form(None),
    size: Optional[str] = Form(None),
    classes: List[int] = Form(None),
    conf: float = Form(None),
    color: Optional[str] = Form(None)
):
    payload = {
        "images": [],
        "types": task,
        "conf": conf
    }
    
    if size is not None:
        payload.update({"model": size})
    
    if classes is not None:
        payload.update({"classes": classes})
    
    if color is not None:
        payload.update({"base_color": color})
        
    for file in files:
        file_name = file.filename
        
        image = base64.b64encode(file.file.read()).decode("utf-8")
        
        decoded_img = base64.b64decode(image)
        bytes_img = BytesIO(decoded_img)
        img = Image.open(bytes_img)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, file_name), img)
        
        payload["images"].append(
            {
                "name": file_name,
                "image": image
            }
        )
        
    if task == "detection":
        resp = requests.post(
            url = f"http://{DET}:8000/detect",
            data = json.dumps(payload),
            headers = {"Content-Type": "application/json"}
        )
        
        resp = literal_eval(resp.content.decode("utf-8"))
        
    with open("./results.json", "w") as f:
        json.dump(resp, f, indent=4)
        
    return templates.TemplateResponse("go_back.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(APP, host=IP)
