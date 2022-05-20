from pydantic import BaseModel


class Order(BaseModel):
    order_id: int
    request: str

    class Config:
        orm_mode = True
