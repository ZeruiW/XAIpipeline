import gradio as gr
import requests
import json
from PIL import Image
import io
import pandas as pd

# Base URL for the API endpoints
#BASE_URL = "http://xaiport.ddns.net"  # Replace with your actual API server address
BASE_URL = "http://localhost"
def api_request(endpoint, method="POST", data=None, files=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

def save_and_run_pipeline(config):
    response = api_request("/run_pipeline", data={"config": config})
    return response.get("message", "Error occurred")

def fetch_results(config):
    response = api_request("/fetch_results", data={"config": config})
    if "error" in response:
        return None, None
    csv_data = pd.read_csv(io.StringIO(response["csv_data"]))
    plot_image = Image.open(io.BytesIO(response["plot_image"]))
    return csv_data, plot_image

def handle_dataset_selection(dataset_name):
    response = api_request(f"/dataset_preview/{dataset_name}", method="GET")
    return pd.DataFrame(response["data"]) if "error" not in response else response["error"]

def model_inference(dataset_name, model_type, index):
    response = api_request("/model_inference", data={
        "dataset_name": dataset_name,
        "model_type": model_type,
        "index": index
    })
    if "error" in response:
        return None, None, None
    return pd.DataFrame(response["sample_data"]), response["sample_label"], response["prediction"]

def explain_instance(dataset_name, model_type, index):
    response = api_request("/explain_instance", data={
        "dataset_name": dataset_name,
        "model_type": model_type,
        "index": index
    })
    return response if "error" not in response else response["error"]

def evaluate_global_shap_values(dataset_name, model_type, sample_size):
    response = api_request("/global_shap_values", data={
        "dataset_name": dataset_name,
        "model_type": model_type,
        "sample_size": sample_size
    })
    if "error" in response:
        return None, None
    return response["global_shap_dict"], response["stability_score"]

def predict(image, model_name):
    files = {"image": ("image.jpg", image, "image/jpeg")}
    response = api_request("/predict_image", data={"model_name": model_name}, files=files)
    return response["predicted_label"] if "error" not in response else response["error"]

def apply_cam(image, method_name, model_name):
    files = {"image": ("image.jpg", image, "image/jpeg")}
    response = api_request("/apply_cam", data={
        "method_name": method_name,
        "model_name": model_name
    }, files=files)
    if "error" in response:
        return None
    return Image.open(io.BytesIO(response["cam_image"]))
#text
import gradio as gr
import requests
import json
from PIL import Image
import io
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import math

def get_tree_depth(data):
    if 'children' not in data or not data['children']:
        return 0
    return 1 + max(get_tree_depth(child) for child in data['children'])

def build_radial_tree(data):
    G = nx.Graph()
    node_x, node_y = [], []
    node_text, node_colors = [], []
    edge_x, edge_y = [], []
    
    max_depth = get_tree_depth(data)
    
    def add_node(item, parent=None, level=0, start_angle=0, end_angle=2*math.pi):
        G.add_node(item['name'])
        angle = (start_angle + end_angle) / 2
        r = level * (8 / max_depth)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        node_x.append(x)
        node_y.append(y)
        node_text.append(item['name'])
        node_colors.append(max_depth - level)
        if parent:
            parent_x, parent_y = G.nodes[parent]['x'], G.nodes[parent]['y']
            G.add_edge(parent, item['name'])
            edge_x.extend([x * 0.9, parent_x * 1.1, None])
            edge_y.extend([y * 0.9, parent_y * 1.1, None])
        G.nodes[item['name']]['x'] = x
        G.nodes[item['name']]['y'] = y
        if 'children' in item and item['children']:
            angle_step = (end_angle - start_angle) / len(item['children'])
            for i, child in enumerate(item['children']):
                child_start = start_angle + i * angle_step
                child_end = child_start + angle_step * 0.9
                add_node(child, item['name'], level+1, child_start, child_end)
    
    add_node(data)
    
    return G, node_x, node_y, node_text, edge_x, edge_y, node_colors



    max_range = max(max(abs(min(node_x)), abs(max(node_x))), max(abs(min(node_y)), abs(max(node_y))))
    fig.update_xaxes(range=[-max_range-1, max_range+1])
    fig.update_yaxes(range=[-max_range-1, max_range+1])

    return fig

# def process_dataset(dataset_choice, custom_dataset):
#     if dataset_choice == "MSR":
#         with open("xaiserver/e2e-process/flower_json/msr.json", "r") as f:
#             data = json.load(f)
#         radial_tree_chart = generate_radial_tree_chart(data)
        
#         response = api_request("/load_msr_data", method="GET")
#         if "error" in response:
#             return gr.Plot.update(visible=False), gr.Image.update(visible=False), f"Error loading MSR data: {response['error']}"
        
#         similarity_plot = Image.open(io.BytesIO(response["similarity_plot"]))
#         return radial_tree_chart, similarity_plot, None
#     elif dataset_choice == "Custom" and custom_dataset:
#         try:
#             custom_data = json.loads(custom_dataset.decode('utf-8'))
#         except json.JSONDecodeError:
#             return gr.Plot.update(visible=False), gr.Image.update(visible=False), "Invalid JSON format in custom dataset"
        
#         radial_tree_chart = generate_radial_tree_chart(custom_data)
        
#         files = {"dataset": ("custom_dataset.json", custom_dataset, "application/json")}
#         response = api_request("/process_custom_dataset", files=files)
#         if "error" in response:
#             return gr.Plot.update(visible=False), gr.Image.update(visible=False), f"Error processing custom dataset: {response['error']}"
        
#         similarity_plot = Image.open(io.BytesIO(response["similarity_plot"]))
#         return radial_tree_chart, similarity_plot, None
#     else:
#         return gr.Plot.update(visible=False), gr.Image.update(visible=False), "Please select a dataset or upload a custom one."

# def process_dataset(dataset_choice, custom_dataset):
#     if dataset_choice == "MSR":
#         response = api_request("/load_msr_data", method="GET")
#     elif dataset_choice == "Custom" and custom_dataset:
#         files = {"dataset": ("custom_dataset.json", custom_dataset, "application/json")}
#         response = api_request("/process_custom_dataset", files=files)
#     else:
#         return "Please select a dataset or upload a custom one.", None

#     if "error" in response:
#         return response["error"], None

#     return response["plotly_chart"], Image.open(io.BytesIO(response["similarity_plot"]))

import json
from PIL import Image
import plotly.graph_objects as go
import networkx as nx
import math

def get_tree_depth(data):
    if 'children' not in data or not data['children']:
        return 0
    return 1 + max(get_tree_depth(child) for child in data['children'])



def generate_radial_tree_chart(data):
    G, node_x, node_y, node_text, edge_x, edge_y, node_colors = build_radial_tree(data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color='black'),
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_colors,
            size=8,
            colorbar=dict(thickness=15, title='Tree Level'),
            line_width=2)))

    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        width=1000,
        height=1000,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    max_range = max(max(abs(min(node_x)), abs(max(node_x))), max(abs(min(node_y)), abs(max(node_y))))
    fig.update_xaxes(range=[-max_range-1, max_range+1])
    fig.update_yaxes(range=[-max_range-1, max_range+1])

    return fig

def load_msr_json():
    with open("xaiserver/e2e-process/flower_json/msr.json", "r") as f:
        return json.load(f)

def process_dataset(dataset_choice, custom_dataset):
    if dataset_choice == "MSR":
        data = load_msr_json()
        radial_tree_chart = generate_radial_tree_chart(data)
        
        similarity_plot_path = 'xaiserver/e2e-process/final_diagram/similarity_matrix.png'
        similarity_plot = Image.open(similarity_plot_path)
        
        return radial_tree_chart, similarity_plot, None
    elif dataset_choice == "Custom" and custom_dataset:
        try:
            custom_data = json.loads(custom_dataset.decode('utf-8'))
            radial_tree_chart = generate_radial_tree_chart(custom_data)
            
            # 这里需要实现自定义数据集的相似度矩阵生成逻辑
            # 暂时返回一个空图像
            similarity_plot = Image.new('RGB', (100, 100), color = 'white')
            
            return radial_tree_chart, similarity_plot, None
        except json.JSONDecodeError:
            return gr.Plot.update(visible=False), gr.Image.update(visible=False), "Invalid JSON format in custom dataset"
    else:
        return gr.Plot.update(visible=False), gr.Image.update(visible=False), "Please select a dataset or upload a custom one."




# Video

import shutil
from pathlib import Path
import moviepy.editor as mp

# ... existing code ...

VIDEO_SERVER_URL = "http://0.0.0.0:6313"
UPLOAD_DIR = Path("uploaded_videos")  # 指定上传文件保存的目录
RESULT_DIR = Path("results")

def ensure_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)

def upload_video(video_file):
    ensure_dir(UPLOAD_DIR)
    video_path = UPLOAD_DIR / Path(video_file.name).name
    shutil.copy(video_file.name, video_path)  # 复制文件到指定目录
    
    url = f"{VIDEO_SERVER_URL}/upload_video/"
    files = {"file": (video_path.name, open(video_path, "rb"), "video/mp4")}
    response = requests.post(url, files=files)
    return response.json()

def check_video_status(video_id):
    url = f"{VIDEO_SERVER_URL}/video_status/{video_id}"
    response = requests.get(url)
    return response.json()

def download_result(video_id, file_type):
    url = f"{VIDEO_SERVER_URL}/download/{video_id}/{file_type}"
    response = requests.get(url)
    if response.status_code == 200:
        file_name = RESULT_DIR / f"{file_type}_{video_id}.{file_type.split('_')[-1]}"
        with open(file_name, "wb") as f:
            f.write(response.content)
        return str(file_name)
    else:
        return None

def convert_to_gif(video_path):
    gif_path = video_path.with_suffix('.gif')
    video = mp.VideoFileClip(str(video_path))
    video.write_gif(str(gif_path), fps=10)  # Adjust fps as needed
    return str(gif_path)

def process_video(video_file):
    # Upload video
    upload_result = upload_video(video_file)
    video_id = upload_result["video_id"]
    status_message = f"Video uploaded. Video ID: {video_id}"
    yield status_message, None, None, None, None

    # Check processing status
    max_retries = 60  # 5 minutes max wait time (5 * 60 seconds)
    for _ in range(max_retries):
        status = check_video_status(video_id)
        status_message += f"\nProcessing status: {status['status']}"
        yield status_message, None, None, None, None
        
        if status["status"] == "completed":
            break
        elif status["status"] == "failed":
            error_message = status.get('error', 'Unknown error')
            status_message += f"\nProcessing failed: {error_message}"
            yield status_message, None, None, None, None
            return
        time.sleep(5)
    
    if status["status"] != "completed":
        status_message += "\nProcessing timed out. Please try again later."
        yield status_message, None, None, None, None
        return

    # Download results
    heatmap_video = download_result(video_id, "heatmap_video")
    json_data = download_result(video_id, "json_data")
    visualization = download_result(video_id, "visualization")

    # Convert heatmap video to GIF
    if heatmap_video:
        heatmap_gif = convert_to_gif(Path(heatmap_video))
    else:
        heatmap_gif = None

    status_message += f"\nProcessing completed. Results downloaded."
    yield status_message, heatmap_gif, json_data, visualization, heatmap_video




# Create the main Gradio app with tabs
with gr.Blocks() as app:
    gr.Markdown("# XAIport")
    gr.Markdown("Welcome to XAIport! This application allows you to manage and configure various tasks related to Explainable AI (XAI). Use the tabs below to navigate through different functionalities.")

    with gr.Tabs():
        with gr.TabItem("Task pipeline"):
            json_input = gr.Textbox(lines=20, label="Fill Task Configuration", value="")
            output_text = gr.Textbox(label="Status")
            save_button = gr.Button("Submit Task")
            save_button.click(fn=save_and_run_pipeline, inputs=json_input, outputs=output_text)

            results_button = gr.Button("Fetch Results")
            results_dataframe = gr.DataFrame(label="Results")
            results_plot = gr.Image(label="Results Plot")
            results_button.click(fn=fetch_results, inputs=json_input, outputs=[results_dataframe, results_plot])

        with gr.TabItem("Tabular"):
            gr.Markdown("## Select a Dataset")
            dataset_dropdown = gr.Dropdown(
                choices=["COMPAS", "IoT", "Product"],
                label="Select Dataset",
                value="COMPAS"
            )
            dataset_display = gr.DataFrame(label="Dataset Preview")
            dataset_dropdown.change(fn=handle_dataset_selection, inputs=dataset_dropdown, outputs=dataset_display)

            gr.Markdown("## Select a Model")
            model_dropdown = gr.Dropdown(
                choices=["FTTransformer", "TabNet", "TabTransformer"],
                label="Select Model",
                value="TabTransformer"
            )

            gr.Markdown("## Enter Dataset Index for Inference")
            index_input = gr.Textbox(label="Dataset Index", value="0")

            infer_button = gr.Button("Inference")
            sample_data_display = gr.DataFrame(label="Sample Data")
            sample_label_display = gr.Textbox(label="Actual Label")
            prediction_display = gr.JSON(label="Prediction Result")

            infer_button.click(
                fn=model_inference,
                inputs=[dataset_dropdown, model_dropdown, index_input],
                outputs=[sample_data_display, sample_label_display, prediction_display]
            )

            gr.Markdown("## XAI Methods")
            xai_method_dropdown = gr.Dropdown(
                choices=["SHAP"],
                label="Select XAI Method",
                value="SHAP"
            )

            explain_button = gr.Button("Explain Instance")
            instance_shap_values_display = gr.JSON(label="Feature SHAP Values")

            explain_button.click(
                fn=explain_instance,
                inputs=[dataset_dropdown, model_dropdown, index_input],
                outputs=instance_shap_values_display
            )

            gr.Markdown("## Global Explanation and Stability Score")
            sample_size_input = gr.Textbox(label="Sample Size", value="100")
            global_explain_button = gr.Button("Global Explanation and Stability Score")
            global_shap_values_display = gr.JSON(label="Global Feature Importance")
            stability_score_display = gr.Textbox(label="Stability Score")

            global_explain_button.click(
                fn=evaluate_global_shap_values,
                inputs=[dataset_dropdown, model_dropdown, sample_size_input],
                outputs=[global_shap_values_display, stability_score_display]
            )

        # with gr.TabItem("Text"):
        #     gr.Markdown("This tab displays the Common Weakness Enumeration hierarchy as a radial tree chart and similarity matrix.")

        #     dataset_choice = gr.Radio(["MSR", "Custom"], label="Choose Dataset", value="MSR")
        #     custom_dataset = gr.File(label="Upload Custom Dataset (if applicable)")
        #     plot_button = gr.Button("Generate Charts")
        #     plotly_chart_display = gr.Plot(label="Tree Chart")
        #     similarity_plot_display = gr.Image(label="Similarity Matrix")

        #     plot_button.click(
        #         fn=process_dataset,
        #         inputs=[dataset_choice, custom_dataset],
        #         outputs=[plotly_chart_display, similarity_plot_display]
        #     )
        with gr.TabItem("Text"):
            gr.Markdown("## Code Vulnerability Explainable AI Analysis")
            gr.Markdown("This tab displays the Common Weakness Enumeration (CWE) hierarchy as a radial tree chart and a similarity matrix.")

            dataset_choice = gr.Radio(["MSR", "Custom"], label="Choose Dataset", value="MSR")
            custom_dataset = gr.File(label="Upload Custom Dataset (JSON format, if applicable)")
            process_button = gr.Button("Generate Analysis")
            radial_tree_chart = gr.Plot(label="CWE Hierarchy Tree Chart")
            similarity_matrix = gr.Image(label="CWE Similarity Matrix")
            status_message = gr.Textbox(label="Status", visible=False)

            process_button.click(
                fn=process_dataset,
                inputs=[dataset_choice, custom_dataset],
                outputs=[radial_tree_chart, similarity_matrix, status_message]
            )



    # ... other tabs ...


        with gr.TabItem("Vision"):
            gr.Markdown("## Upload an Image for Classification and GradCAM Explanation")
            image_input = gr.Image(type="numpy", label="Upload Image")
            model_dropdown = gr.Dropdown(
                choices=["resnet50", "nvidia/mit-b0", "microsoft/swin-large-patch4-window12-384-in22k", 
                         "microsoft/cvt-13", "google/vit-large-patch32-384", "facebook/convnext-tiny-224"],
                label="Select Model",
                value="resnet50"
            )
            cam_algorithm = gr.Dropdown(choices=["GradCAM", "HiResCAM", "GradCAMPlusPlus", "XGradCAM", "LayerCAM", "ALL"], label="Select CAM Algorithm")
            classify_button = gr.Button("Classify Image")
            prediction_display = gr.Textbox(label="Predicted Label")
            gradcam_button = gr.Button("Apply CAM")
            cam_display = gr.Image(label="GradCAM")

            classify_button.click(
                fn=predict,
                inputs=[image_input, model_dropdown],
                outputs=prediction_display
            )

            gradcam_button.click(
                fn=apply_cam,
                inputs=[image_input, cam_algorithm, model_dropdown],
                outputs=cam_display
            )

        
        with gr.TabItem("Stream"):
            gr.Markdown("## Video Attention Analysis")
            gr.Markdown("Upload a video to analyze attention using TimesFormer model.")
            
            video_input = gr.File(label="Upload Video")
            process_button = gr.Button("Process Video")
            status_output = gr.Textbox(label="Processing Status")
            heatmap_output = gr.Image(label="Heatmap Video (GIF)")
            json_output = gr.File(label="JSON Data")
            visualization_output = gr.Image(label="Visualization")
            download_video_output = gr.File(label="Download Heatmap Video (MP4)")

            process_button.click(
                fn=process_video,
                inputs=video_input,
                outputs=[status_output, heatmap_output, json_output, visualization_output, download_video_output]
            )

# Launch Gradio app
if __name__ == "__main__":
    #app.launch(server_name="0.0.0.0", server_port=7860)
    ensure_dir(UPLOAD_DIR)
    ensure_dir(RESULT_DIR)
    app.launch(share=True)