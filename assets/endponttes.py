from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

# 定义数据模型
class Item(BaseModel):
    any_data: dict

@app.post("/test")
async def test_endpoint(item: Item):
    return JSONResponse(status_code=200, content={"message": "Received!", "yourData": item.any_data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
