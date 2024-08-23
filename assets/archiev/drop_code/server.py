from fastapi import FastAPI, Request
import coordination_center_api
import uvicorn

app = FastAPI()

@app.post("/taskexecution/")
async def run_script(request: Request):
    config_json = await request.json()
    coordination_center_api.run_task_from_api(config_json)  # 调用新函数
    return {"output": "Task completed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
