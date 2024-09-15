from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import io
import numpy as np
import base64
from pytorch_grad_cam import (
    GradCAM, HiResCAM, GradCAMPlusPlus, XGradCAM, LayerCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
import cam_resnet  # Assuming this module exists and contains the necessary functions

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ImageNet class index
with open("index/imagenet_class_index.json", "r") as f:
    imagenet_class_index = json.load(f)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True).to(device).eval()

# Prepare transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class XAIRequest(BaseModel):
    dataset_id: str
    algorithms: List[str]

@app.post("/cam_xai")
async def run_xai(request: XAIRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_xai_process, request.dataset_id, request.algorithms)
        return {"message": "XAI processing for dataset has started successfully."}
    except Exception as e:
        logger.error(f"Error in run_xai endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_xai_process(dataset_id: str, algorithm_names: List[str]):
    try:
        local_dataset_path = f"/path/to/datasets/{dataset_id}"  # Adjust this path as needed
        dataset_dirs = [local_dataset_path]
        selected_algorithms = [cam_resnet.CAM_ALGORITHMS_MAPPING[name] for name in algorithm_names]
        cam_resnet.xai_run(dataset_dirs, selected_algorithms)
        logger.info(f"XAI processing completed for dataset {dataset_id}")
    except Exception as e:
        logger.error(f"Error in run_xai_process: {e}")
        raise

@app.post("/predict_image")
async def predict_image(image: UploadFile = File(...), model_name: str = Form(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
        
        _, predicted_idx = torch.max(output, 1)
        predicted_label = imagenet_class_index[str(predicted_idx.item())][1]

        return {"predicted_label": predicted_label}
    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply_cam")
async def apply_cam(image: UploadFile = File(...), method_name: str = Form(...), model_name: str = Form(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        target_layers = [model.layer4[-1]]
        cam_algorithm = cam_resnet.CAM_ALGORITHMS_MAPPING[method_name]
        cam = cam_algorithm(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        
        grayscale_cam = cam(input_tensor=img_tensor)
        rgb_img = np.float32(img.resize((224, 224))) / 255
        cam_image = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
        
        cam_image_pil = Image.fromarray((cam_image * 255).astype(np.uint8))
        buffered = io.BytesIO()
        cam_image_pil.save(buffered, format="PNG")
        cam_image_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {"cam_image": cam_image_base64}
    except Exception as e:
        logger.error(f"Error in apply_cam: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)