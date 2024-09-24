import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from uuid import uuid4
import asyncio
import logging
import pandas as pd

from models import AttentionExtractor
from utils import process_video, visualize_saliency, create_heatmap_video

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 存储视频处理状态和结果的字典
video_processing = {}

# 加载 Kinetics-400 标签
kinetics_labels = pd.read_csv("xaiserver/staa/kinetics_400_labels.csv", index_col="id")

@app.post("/upload_video/")
async def upload_video_endpoint(file: UploadFile = File(...)):
    video_id = str(uuid4())
    temp_dir = Path(f"temp_output/{video_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = temp_dir / file.filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    video_processing[video_id] = {"status": "processing"}
    
    # 异步处理视频
    asyncio.create_task(process_video_task(video_id, str(video_path)))
    
    return JSONResponse(content={
        "video_id": video_id,
        "message": "Video uploaded and processing started"
    })

async def process_video_task(video_id: str, video_path: str):
    try:
        logger.info(f"Starting processing for video {video_id}")
        extractor = AttentionExtractor('facebook/timesformer-base-finetuned-k400')
        output_dir = Path(f"temp_output/{video_id}/output")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        predicted_label = process_video(video_path, str(output_dir), extractor)
        logger.info(f"Video {video_id} processed. Predicted label: {predicted_label}")
        
        # 获取预测标签对应的动作名称
        predicted_action = kinetics_labels.loc[predicted_label, "name"]
        
        video_name = Path(video_path).stem
        logger.info(f"Visualizing saliency for video {video_id}")
        visualize_saliency(video_name, results_dir=str(output_dir))
        
        logger.info(f"Creating heatmap video for {video_id}")
        heatmap_video_path = create_heatmap_video(str(output_dir), video_name)
        
        # 检查所有必要的文件是否都已创建
        heatmap_video = next(output_dir.glob("*_heatmap.mp4"), None)
        json_data = output_dir / "results.json"
        visualization = next(output_dir.glob("*_temporal_saliency_visualization.png"), None)
        
        if not heatmap_video:
            logger.error(f"Heatmap video not found for {video_id}")
        if not json_data.exists():
            logger.error(f"JSON data not found for {video_id}")
        if not visualization:
            logger.error(f"Visualization not found for {video_id}")
        
        if not all([heatmap_video, json_data.exists(), visualization]):
            raise Exception("Not all required files were created")
        
        video_processing[video_id] = {
            "status": "completed",
            "predicted_label": int(predicted_label),
            "predicted_action": predicted_action,
            "heatmap_video_url": f"/download/{video_id}/heatmap_video",
            "json_data_url": f"/download/{video_id}/json_data",
            "visualization_url": f"/download/{video_id}/visualization"
        }
        logger.info(f"Processing completed for video {video_id}")
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        video_processing[video_id] = {"status": "failed", "error": str(e)}

@app.get("/video_status/{video_id}")
async def get_video_status(video_id: str):
    if video_id not in video_processing:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return JSONResponse(content=video_processing[video_id])

@app.get("/download/{video_id}/{file_type}")
async def download_result(video_id: str, file_type: str):
    if video_id not in video_processing or video_processing[video_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Result not found or processing not completed")
    
    output_dir = Path(f"temp_output/{video_id}/output")
    
    if file_type == "heatmap_video":
        file_path = next(output_dir.glob("*_heatmap.mp4"), None)
    elif file_type == "json_data":
        file_path = output_dir / "results.json"
    elif file_type == "visualization":
        file_path = next(output_dir.glob("*_temporal_saliency_visualization.png"), None)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if not file_path or not file_path.exists():
        logger.error(f"File not found: {file_type} for video {video_id}")
        raise HTTPException(status_code=404, detail=f"File not found: {file_type}")
    
    return FileResponse(file_path, filename=file_path.name)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6313)
