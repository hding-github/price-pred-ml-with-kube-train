from fastapi import FastAPI
import train_api
from typing import Union

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/train/{item_id}")
def read_item(item_id: int, data: Union[str, None] = None, model: Union[str, None] = None):
    strResults = train_api.input(data, model)
    return {"item_id": item_id, "q": strResults}

if __name__ == "__main__":
    print("****")