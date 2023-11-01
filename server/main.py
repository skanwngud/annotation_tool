import uvicorn
import socket
import base64
import cv2
import numpy as np

from ast import literal_eval
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List, Optional

from utils import check, scan_ip, get_servers

IP = socket.gethostbyname(socket.gethostname())

app = FastAPI()

ip_list = scan_ip(IP)
DET, SEG, POS, CLU = get_servers(ip_list)

class Input(BaseModel):
    images: Union[bytes, List[bytes]]
    types: str
    classes: Optional[Union[List[str], str]] = None
    images_info: dict

@app.post("/inference")
async def inference(inp: Input):
    images = inp.images if isinstance(inp.images, list) else [inp.images]
    model_type, classes = check(inp)
    
    for idx, bytes_String in enumerate(images):
        img_dta = base64.b64decode()
        data_bytes = np.fromstring(img_data, dtype=np.uint8)
        img = data_bytes.reshape((360, 640, 3))
        
        cv2.imwrite(f"{idx}.idx", img)
        
    res = {
        "image": images,
        "type": types,
        "classes": classes
    }
    
    if types == "detect":
        rsp = request.post(
            url=f"http://{DET}:8000/detect",
            data=res
        )
        rsp = rsp.content.decode("utf-8")
        res.update(literal_eval(rsp))
    
    return res

if __name__ == "__main__":
    uvicorn.run("main:app", host=IP)