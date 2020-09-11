# coding:utf-8

import uvicorn
import torch
import sys
from com.model import ConvNet
from predict import Predict
from fastapi import FastAPI
from pydantic import BaseModel
from config import LABELS
import logging

app = FastAPI(title="Captcha Prediction")


class PredictModel(BaseModel):
    image: str


@app.post("/predict")
async def predict(item: PredictModel):
    try:
        item = item.dict()
        shape, image = pred.image(item['image'])
        result, score = pred.predict(image)
        # logging.info(msg=item['image'])
        return {'code': '1', 'msg': 'success', 'com': {'result': result, 'score': score}}
    except:
        e_type, e_value, e_traceback = sys.exc_info()
        # logging.info()


if __name__ == '__main__':
    log_config = logging.basicConfig(level=logging.INFO, format='%(asctime)s\t[%(levelname)s]\t%(message)s')
    model = ConvNet()
    model.load_state_dict(torch.load('dahuo_20200907.pkl', map_location='cpu'))
    pred = Predict(model, LABELS)
    uvicorn.run(app=app, host='127.0.0.1', port=9000, log_level='debug')
