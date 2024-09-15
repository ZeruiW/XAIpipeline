import gradio as gr
import requests
import json
from PIL import Image
import io
import pandas as pd

# Base URL for the API endpoints
BASE_URL = "http://xaiport.ddns.net"  # Replace with your actual API server address

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

def process_dataset(dataset_choice, custom_dataset):
    if dataset_choice == "MSR":
        response = api_request("/load_msr_data", method="GET")
    elif dataset_choice == "Custom" and custom_dataset:
        files = {"dataset": ("custom_dataset.json", custom_dataset, "application/json")}
        response = api_request("/process_custom_dataset", files=files)
    else:
        return "Please select a dataset or upload a custom one.", None

    if "error" in response:
        return response["error"], None

    return response["plotly_chart"], Image.open(io.BytesIO(response["similarity_plot"]))

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

        with gr.TabItem("Text"):
            gr.Markdown("This tab displays the Common Weakness Enumeration hierarchy as a radial tree chart and similarity matrix.")

            dataset_choice = gr.Radio(["MSR", "Custom"], label="Choose Dataset", value="MSR")
            custom_dataset = gr.File(label="Upload Custom Dataset (if applicable)")
            plot_button = gr.Button("Generate Charts")
            plotly_chart_display = gr.Plot(label="Tree Chart")
            similarity_plot_display = gr.Image(label="Similarity Matrix")

            plot_button.click(
                fn=process_dataset,
                inputs=[dataset_choice, custom_dataset],
                outputs=[plotly_chart_display, similarity_plot_display]
            )

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
            gr.Markdown("This tab will contain functionalities for stream data.")

# Launch Gradio app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)