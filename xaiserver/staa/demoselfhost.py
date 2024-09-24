import os
import shutil
from pathlib import Path
from uuid import uuid4
import logging
import pandas as pd
import gradio as gr
import time

from models import AttentionExtractor
from utils import process_video, visualize_saliency, create_heatmap_video

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Kinetics-400 labels
kinetics_labels = pd.read_csv("app/kinetics_400_labels.csv", index_col="id")

def process_video_task(video_path: str):
    try:
        video_id = str(uuid4())
        logger.info(f"Starting processing for video {video_id}")
        extractor = AttentionExtractor('facebook/timesformer-base-finetuned-k400')
        output_dir = Path(f"temp_output/{video_id}/output")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        predicted_label = process_video(video_path, str(output_dir), extractor)
        logger.info(f"Video {video_id} processed. Predicted label: {predicted_label}")
        
        predicted_action = kinetics_labels.loc[predicted_label, "name"]
        
        video_name = Path(video_path).stem
        logger.info(f"Visualizing saliency for video {video_id}")
        visualize_saliency(video_name, results_dir=str(output_dir))
        
        logger.info(f"Creating heatmap video for {video_id}")
        heatmap_video_path = create_heatmap_video(str(output_dir), video_name)
        
        heatmap_video = next(output_dir.glob("*_heatmap.mp4"), None)
        json_data = output_dir / "results.json"
        visualization = next(output_dir.glob("*_temporal_saliency_visualization.png"), None)
        
        if not all([heatmap_video, json_data.exists(), visualization]):
            raise Exception("Not all required files were created")
        
        logger.info(f"Processing completed for video {video_id}")
        return {
            "status": "completed",
            "predicted_label": int(predicted_label),
            "predicted_action": predicted_action,
            "heatmap_video_path": str(heatmap_video),
            "json_data_path": str(json_data),
            "visualization_path": str(visualization)
        }
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {"status": "failed", "error": str(e)}

def gradio_process_video(video_file):
    yield "Video uploaded. Starting processing..."

    # Process the video
    result = process_video_task(video_file.name)

    if result["status"] == "completed":
        yield f"Processing completed. Predicted action: {result['predicted_action']} (Label: {result['predicted_label']})"
        return (
            result['heatmap_video_path'],
            result['json_data_path'],
            result['visualization_path']
        )
    else:
        yield f"Processing failed: {result.get('error', 'Unknown error')}"
        return None, None, None

iface = gr.Interface(
    fn=gradio_process_video,
    inputs=gr.File(label="Upload Video"),
    outputs=[
        gr.Textbox(label="Processing Status"),
        gr.Video(label="Heatmap Video"),
        gr.File(label="JSON Data"),
        gr.Image(label="Visualization")
    ],
    title="Video Attention Analysis",
    description="Upload a video to analyze attention using TimesFormer model.",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=6314)