import logging
import os
from typing import Optional
from urllib import request
import xgboost as xgb

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


@app.get("/health")
async def root():
    return {"message": "It's Ok"}


@app.post("/calculate/")
def calculate_score(orders: Order):
    try:
        # Create Datagrame
        df = pd.DataFrame({
            'total_earning': orders.total_earning,
            'to_user_distance': orders.to_user_distance,
            'to_user_elevation': orders.to_user_elevation
        }, index=[0])

        print(df)
        data_m = xgb.DMatrix(df)
        df['taken'] = model.predict(data_m)
        df['taken_score'] =  df['taken']
        df['taken'] = (df['taken'] > .5).astype(int)
        print(df)
        data = df.to_json(orient='records')[1:-1].replace('},{', '} {')
        print(data)

    except (APIException, ModelParamException) as e:
        return {'error': str(e)}, 400
    db_Order = ModelOrder(order_id=orders.order_id, request=data)
    db.session.add(db_Order)
    db.session.commit()
    return data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
