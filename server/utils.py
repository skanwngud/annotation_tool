import requests

from ast import literal_eval
from pydantic import BaseModel

def check(inp: BaseModel):
    types = inp.types
    classes = inp.classes
    
    assert types in ["detect", "segmentation", "clustering", "pose"], \
        "Please, check your inference type. Inference types are detect, segnmentation, clustering, pose."
        
    if types in ["detect", "segmentation"]:
        assert classes is not None, "Please, select classes for detect at least one."
        
    return types, classes


def scan_ip(host_ip: str):
    ip_range = [".".join([*host_ip.split(".")[:3], str(i)]) for i in range(1, 5)]
    port = 8000
    
    ips = []
    for ip in ip_range:
        try:
            res = requests.get(f"http://{ip}:{port}")
            ips.append(literal_eval(res.content.decode("utf-8")))
            
        except requests.exceptions.ConnectionError:
            continue
        
    return ips


def get_servers(ip_list):
    detect, segmentation, pose, clustering = None, None, None, None
    for val in ip_list:
        ip = list(val.keys())[0]
        task = list(val.values())[0]
        
        if task == "detect":
            det = ip
            
        elif task == "segmentation":
            segmentation = ip
            
        elif task == "pose":
            pose = ip
            
        elif task == "clustering":
            clustering = ip
            
    return detect, segmentation, pose, clustering