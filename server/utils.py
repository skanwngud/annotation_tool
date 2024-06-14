import requests

from loguru import logger

from ast import literal_eval
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

import time


def check(inp: BaseModel, task: str):
    types = inp.types
    classes = inp.classes

    assert types.lower() == task.lower(), f"You're trying request to wrong task. request type is {types}, but task is {task}"

    assert types in [
        "detect",
        "segmentation",
        "clustering",
        "pose",
    ], "Please, check your inference type. Inference types are detect, segnmentation, clustering, pose."

    if types in ["detect", "segmentation"]:
        assert classes is not None, "Please, select classes for detect at least one."

    return types, classes


def scan_ip(host_ip: str, port: int):
    logger.info("----------IP Scan start----------")
    ip_range = [
        f"http://{'.'.join([*host_ip.split('.')[:3], str(i)])}:{port}"
        for i in range(1, 255)
    ]

    def get_response(url):
        try:
            return requests.get(url)
        except Exception:
            pass

    time.sleep(3)

    with ThreadPoolExecutor(max_workers=len(ip_range)) as pool:
        response = list(pool.map(get_response, ip_range))

    return [
        literal_eval(resp.content.decode("utf-8"))
        for resp in response
        if resp is not None
    ]


def get_servers(ip_list):
    detect, segmentation, pose, clustering = None, None, None, None
    for val in ip_list:
        ip = list(val.keys())[0]
        task = list(val.values())[0]

        if task == "detect":
            detect = ip

        elif task == "segmentation":
            segmentation = ip

        elif task == "pose":
            pose = ip

        elif task == "clustering":
            clustering = ip

    return detect, segmentation, pose, clustering
