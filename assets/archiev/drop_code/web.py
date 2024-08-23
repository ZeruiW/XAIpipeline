import gradio as gr
import json

def create_config(upload_url, local_zip_path, perturbation_url, perturbation_type, severity, model_url, model_name, xai_url, dataset_id, algorithms, eval_url, eval_metric):
    config = {
        "upload_config": {
            "server_url": upload_url,
            "datasets": {
                "t1": {
                    "local_zip_path": local_zip_path
                }
            }
        },
        "perturbation_config": {
            "server_url": perturbation_url,
            "datasets": {
                "t1": {
                    "perturbation_type": perturbation_type,
                    "severity": severity
                }
            }
        },
        "model_config": {
            "base_url": model_url,
            "models": {
                "t1": {
                    "model_name": model_name,
                    "perturbation_type": perturbation_type,
                    "severity": severity
                }
            }
        },
        "xai_config": {
            "base_url": xai_url,
            "datasets": {
                "t1": {
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "algorithms": algorithms.split(", ")
                }
            }
        },
        "evaluation_config": {
            "base_url": eval_url,
            "datasets": {
                "t1": {
                    "evaluation_metric": eval_metric,
                    "model_name": model_name,
                    "perturbation_func": perturbation_type,
                    "severity": severity,
                    "xai_method": "cam_xai",
                    "algorithms": algorithms.split(", ")
                }
            }
        }
    }
    return json.dumps(config, indent=2)

iface = gr.Interface(
    create_config,
    [
        gr.Textbox(label="Upload Server URL"),
        gr.Textbox(label="Local ZIP Path"),
        gr.Textbox(label="Perturbation Server URL"),
        gr.Dropdown(label="Perturbation Type", choices=["gaussian_noise", "other"]),
        gr.Slider(label="Severity", minimum=1, maximum=5),
        gr.Textbox(label="Model Server URL"),
        gr.Textbox(label="Model Name"),
        gr.Textbox(label="XAI Server URL"),
        gr.Textbox(label="Dataset ID"),
        gr.Textbox(label="Algorithms (comma separated)"),
        gr.Textbox(label="Evaluation Server URL"),
        gr.Textbox(label="Evaluation Metric"),
    ],
    gr.Textbox(label="Generated Config"),
)

iface.launch()
