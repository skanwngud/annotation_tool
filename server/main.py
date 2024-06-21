import uvicorn
import socket
import requests
import os

from ast import literal_eval
from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Union, Optional, List, Tuple

from loguru import logger


TASKS = {
    "detection": "http://172.20.0.12:8000/detect",
    "pose_estimation": "http://172.20.0.13:8000/pose",
    "segmentation": "http://172.20.0.14:8000/segmentation",
    "clustering": "http://172.20.0.15:8000/clustering"
}


IP = socket.gethostbyname(socket.gethostname())
logger.info(f"server's IP is {IP}")

app = FastAPI()

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.mount("/static", StaticFiles(directory="static"), name="static")  # 정적파일 제공 설정

class Input(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[List[int]] = None
    model: Optional[str] = None
    base_color: Optional[Union[List[int], Tuple[int]]] = None
    conf: Optional[float] = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(exc.body)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body})
    )

@app.post("/inference")
async def inference(
    request: Request,
    inp: Input
):
    task = inp.types
    size = inp.model
    classes = inp.classes
    color = inp.base_color
    conf = inp.conf
    
    logger.info(f"""
    ---the requested data is----
    task: {task}
    size: {size}
    classes: {classes}
    base color: {color}
    confidence: {conf}
    ----------------------------
    """)
    
    resp = requests.post(
        url=TASKS[task],
        data=inp.model_dump_json(),
        headers={"Content-Type": "application/json"}
    )
    
    results = {"status_code": resp.status_code, "results": literal_eval(resp.content.decode("utf-8"))}
    
    return results


if __name__ == "__main__":
    uvicorn.run(app, host=IP)
