import logging
import os
from typing import List, Optional
from urllib import request
import xgboost as xgb
import json

import uvicorn
import pandas as pd
from src.predict.model import load_model
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi_sqlalchemy import DBSessionMiddleware, db
from pydantic import BaseModel

from api.models import Order
from api.models import Order as ModelOrder
from api.schema import Order as SchemaOrder
from api.exceptions import APIException, ModelParamException
from api import feature_transformer

load_dotenv(".env")

app = FastAPI()

app.add_middleware(DBSessionMiddleware, db_url=os.environ["DATABASE_URL"])

try:
    model = load_model()
    logging.info('Model load XD')
except KeyError:
    logging.error('KeyError: missing model')


class Order(BaseModel):
    order_id: int
    store_id: int
    to_user_distance: float
    to_user_elevation: float
    total_earning: int
    taken: Optional[int] = None


class OrderList(BaseModel):
    data: List[Order]


@app.get("/health")
async def root():
    return {"message": "It's Ok"}


@app.post("/calculate/")
def calculate_score(orders: OrderList):
    response = []
    for x in orders.data:
        try:
            # Create Dataframe
            df = pd.DataFrame(
                {
                    'total_earning': x.total_earning,
                    'to_user_distance': x.to_user_distance,
                    'to_user_elevation': x.to_user_elevation
                }, index=[0]
            )
            data_m = xgb.DMatrix(df)
            df['taken'] = model.predict(data_m)
            df['taken_score'] = df['taken']
            df['taken'] = (df['taken'] > .5).astype(int)
            data = df.to_json(orient='records')[1:-1].replace('},{', '} {')
            db_Order = ModelOrder(order_id=x.order_id, request=data)
            db.session.add(db_Order)
            db.session.commit()
            response.append(data)
        except (APIException, ModelParamException) as e:
            return {'error': str(e)}, 400

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
