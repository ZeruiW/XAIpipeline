# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functionaltool.cloudstorage import down_cloud

app = FastAPI()


class RequestModel(BaseModel):
    dataset_id: str
    model_name: str
    perturbation_func: str
    severity: str

@app.post("/download_results/")
async def download_results(request: RequestModel):
    try:
        # 构建云存储目录路径
        cloud_directory = f"evaluation_results/{request.dataset_id}_{request.perturbation_func}_{request.severity}/{request.model_name}/prediction_changes"
        local_directory = f"downloads/{request.dataset_id}/{request.model_name}/prediction_changes"
        
        down_cloud(cloud_directory, local_directory)
        
        return {"message": "Files downloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
