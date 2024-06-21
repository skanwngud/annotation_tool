import uvicorn
import socket
import json
import requests
import base64

from ast import literal_eval

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse

from pydantic import BaseModel
from typing import Union, Optional, List, Tuple

from loguru import logger

from datetime import datetime


today = datetime.today()
logger.add(f"logs/{today.year}{str(today.month).zfill(2)}{str(today.day).zfill(2)}_front_log.log", rotation="00:00")

IP = socket.gethostbyname(socket.gethostname())

templates = Jinja2Templates(directory="templates")  # 파일 렌더링을 위한 템플릿 파일 위치
app = FastAPI()

logger.info(f"Server Started!! in {IP}")


class Input(BaseModel):
    images: List[dict]
    types: str
    classes: Optional[List[int]] = None
    model: Optional[str] = None
    base_color: Optional[Union[List[int], Tuple[int]]] = None
    conf: Optional[float] = None

    
@app.get("/")
async def read_options():
    with open("./templates/select_options.html", "r") as file:
        return HTMLResponse(content=file.read())


@app.get("/get-options")
async def get_options(value: str):
    with open("classes.json", "r") as file:
        options_data = json.load(file)
        
    options = options_data.get(value, [])
    return JSONResponse(content=options)


@app.post("/inference")
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
        
        payload["images"].append(
            {
                "name": file_name,
                "image": image
            }
        )
        
    resp = requests.post(
        url="http://172.20.0.11:8000/inference",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    
    resp = literal_eval(resp.content.decode("utf-8"))
    
    logger.info(resp)
    
    return templates.TemplateResponse("go_back.html", {"request": request, "response": resp.get("results")})


if __name__ == "__main__":
    uvicorn.run(app=app, host=IP, port=8000)