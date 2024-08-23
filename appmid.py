import gradio as gr
import json
import asyncio
from fastapi import HTTPException
import httpx
import pandas as pd
from io import StringIO, BytesIO
from PIL import Image
import requests
    # "perturbation_config": {
    #   "server_url": "http://0.0.0.0:8001",
    #   "datasets": {
    #     "t1204": {
    #       "perturbation_type": "gaussian_noise",
    #       "severity": 3
    #     }
    #   }
    # },

default_json = """
{    "perturbation_config": {
      "server_url": "http://0.0.0.0:8001",
      "datasets": {
        "t1204": {
          "perturbation_type": "gaussian_noise",
          "severity": 3
        }
      }
    },
    "model_config": {
      "server_url": "http://0.0.0.0:8002",
      "models": {
        "t1204": {
          "model_name": "resnet",
          "perturbation_type": "gaussian_noise",
          "severity": 3
        }
      }
    },
    "xai_config": {
      "server_url": "http://0.0.0.0:8003",
      "datasets": {
        "t1204": {
          "model_name": "resnet",
          "dataset_id": "t1204_gaussian_noise_3",
          "algorithms": ["GradCAM", "HiResCAM", "GradCAMPlusPlus", "XGradCAM", "LayerCAM"]
        }
      }
    },
    "evaluation_config": {
      "server_url": "http://0.0.0.0:8004",
      "datasets": {
        "t1204": {
          "evaluation_metric": "evaluate_cam",
          "model_name": "resnet50",
          "perturbation_func": "gaussian_noise",
          "severity": "3",
          "xai_method": "cam_xai",
          "algorithms": ["GradCAM", "HiResCAM", "GradCAMPlusPlus", "XGradCAM", "LayerCAM"]
        }
      }
    }
}
"""

async def async_http_post(url, json_data=None, files=None):
    async with httpx.AsyncClient() as client:
        if json_data:
            response = await client.post(url, json=json_data)
        elif files:
            response = await client.post(url, files=files)
        else:
            response = await client.post(url)

        if response.status_code == 307:
            redirect_url = response.headers.get('Location')
            if redirect_url:
                print(f"Redirecting to {redirect_url}")
                return await async_http_post(redirect_url, json_data, files)

        if response.status_code != 200:
            print(f"Error response: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()

async def process_upload_config(upload_config):
    for dataset_id, dataset_info in upload_config['datasets'].items():
        url = upload_config['server_url'] + f"/upload-dataset/{dataset_id}"
        local_zip_file_path = dataset_info['local_zip_path']

        async with httpx.AsyncClient() as client:
            with open(local_zip_file_path, 'rb') as f:
                files = {'zip_file': (os.path.basename(local_zip_file_path), f)}
                response = await client.post(url, files=files)
            response.raise_for_status()

        print(f"Uploaded dataset {dataset_id} successfully.")

async def process_perturbation_config(perturbation_config):
    url = perturbation_config['server_url']
    for dataset, settings in perturbation_config['datasets'].items():
        full_url = f"{url}/apply-perturbation/{dataset}/{settings['perturbation_type']}/{settings['severity']}"
        await async_http_post(full_url)

async def process_model_config(model_config):
    base_url = model_config['server_url']
    for model, settings in model_config['models'].items():
        full_url = f"{base_url}/{settings['model_name']}/{model}/{settings['perturbation_type']}/{settings['severity']}"
        await async_http_post(full_url)

async def process_xai_config(xai_config):
    base_url = xai_config['server_url']
    for dataset, settings in xai_config['datasets'].items():
        dataset_id = settings.get('dataset_id', '')
        algorithms = settings.get('algorithms', [])
        data = {
            "dataset_id": dataset_id,
            "algorithms": algorithms
        }
        print(data)
        full_url = f"{base_url}/cam_xai/"
        print(full_url)
        await async_http_post(full_url, json_data=data)

async def process_evaluation_config(evaluation_config):
    base_url = evaluation_config['server_url']
    for dataset, settings in evaluation_config['datasets'].items():
        data = {
            "dataset_id": dataset,
            "model_name": settings['model_name'],
            "perturbation_func": settings['perturbation_func'],
            "severity": settings['severity'],
            "cam_algorithms": settings['algorithms']
        }
        full_url = f"{base_url}/evaluate_cam"
        await async_http_post(full_url, json_data=data)

async def run_pipeline_from_config(config):
    step_process_functions = {
        'upload_config': process_upload_config,
        'perturbation_config': process_perturbation_config,
        'model_config': process_model_config,
        'xai_config': process_xai_config,
        'evaluation_config': process_evaluation_config,
    }

    for step_key in config.keys():
        process_function = step_process_functions.get(step_key)
        if process_function:
            await process_function(config[step_key])
            print(f"Completed processing {step_key}")

def run_pipeline(config):
    asyncio.run(run_pipeline_from_config(config))

def save_and_run_pipeline(config):
    try:
        config_dict = json.loads(config)
        run_pipeline(config_dict)
        return "Pipeline task has been submitted successfully."
    except json.JSONDecodeError:
        return "Invalid JSON configuration"

import json
import requests
from io import StringIO, BytesIO
import pandas as pd
from PIL import Image
from fastapi import HTTPException

def fetch_results(config):
    config_dict = json.loads(config)
    evaluation_config = config_dict['evaluation_config']
    base_url = evaluation_config['server_url']
    dataset_id = list(evaluation_config['datasets'].keys())[0]
    settings = evaluation_config['datasets'][dataset_id]
    perturbation_func = settings.get('perturbation_func', None)
    severity = settings.get('severity', None)
    model_name = settings['model_name']

    # Construct URL based on the presence of perturbation_func and severity
    if perturbation_func and severity:
        results_directory = f"{dataset_id}_{perturbation_func}_{severity}/{model_name}/evaluation/prediction_changes"
    else:
        results_directory = f"{dataset_id}/{model_name}/evaluation/prediction_changes"

    # Fetch CSV results
    csv_url = f"{base_url}/results/{dataset_id}/{model_name}/{perturbation_func}/{severity}/csv"
    csv_response = requests.get(csv_url)
    if csv_response.status_code != 200:
        raise HTTPException(status_code=csv_response.status_code, detail=csv_response.text)
    csv_data = pd.read_csv(StringIO(csv_response.text))

    # Fetch plot results
    plot_url = f"{base_url}/results/{dataset_id}/{model_name}/{perturbation_func}/{severity}/plot"
    plot_response = requests.get(plot_url)
    if plot_response.status_code != 200:
        raise HTTPException(status_code=plot_response.status_code, detail=plot_response.text)
    plot_image = Image.open(BytesIO(plot_response.content))

    return csv_data, plot_image


#Tabular data

import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig, TabNetModelConfig, TabTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
import shap
import numpy as np

# Loading datasets
compas_df = pd.read_csv('TabularData/compas-scores-two-years.csv')
iot_df = pd.read_csv('TabularData/RT_IOT2022')
product_df = pd.read_csv('TabularData/pricerunner_aggregate.csv')

# Functions to process datasets based on user's choice
def process_compas_data():
    data = compas_df
    cat_cols = ['sex', 'race', 'age_cat']
    num_cols = ['age', 'priors_count', 'decile_score']
    data['two_year_recid'] = data['two_year_recid'].map({0: 'no', 1: 'yes'})
    target = ['two_year_recid']
    return data, cat_cols, num_cols, target

def process_iot_data():
    data = iot_df
    cat_cols = ['proto', 'service']
    num_cols = [
        'flow_duration', 
        'fwd_pkts_tot', 
        'bwd_pkts_tot', 
        'fwd_data_pkts_tot', 
        'bwd_data_pkts_tot', 
        'flow_pkts_per_sec', 
        'down_up_ratio', 
        'fwd_header_size_tot'
    ]
    target = ['Attack_type']
    return data, cat_cols, num_cols, target

def process_product_data():
    data = product_df
    cat_cols = ['Product Title', 'Cluster ID', 'Cluster Label', 'Category ID']
    num_cols = ['Merchant ID']
    target = ['Label']
    return data, cat_cols, num_cols, target

# Global variables to store the trained models and test datasets
models = {}
test_datasets = {}

# Function to handle dataset selection
def handle_dataset_selection(dataset_name):
    if dataset_name == "COMPAS":
        data, cat_cols, num_cols, target = process_compas_data()
    elif dataset_name == "IoT":
        data, cat_cols, num_cols, target = process_iot_data()
    elif dataset_name == "Product":
        data, cat_cols, num_cols, target = process_product_data()
    else:
        return "Invalid dataset selection"
    
    # Handling missing values by filling them with mean (for numerical columns) or mode (for categorical columns)
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
    
    return data.head()  # Displaying the first few rows of the selected dataset

def train_model(data, cat_cols, num_cols, target, model_type="TabTransformer", dataset_name="Dataset"):
    train, test = train_test_split(data, stratify=data[target], test_size=0.2, random_state=42)
    data_config = DataConfig(
        target=target, # target should always be a list.
        continuous_cols=num_cols,
        categorical_cols=cat_cols,
    )

    trainer_config = TrainerConfig(
        batch_size=256,
        max_epochs=10,
        early_stopping="valid_loss",
        early_stopping_mode="min",
        early_stopping_patience=5,
        checkpoints="valid_loss",
        load_best=True,
    )

    optimizer_config = OptimizerConfig()

    head_config = LinearHeadConfig(
        layers="",
        dropout=0.1,
        initialization="kaiming"
    ).__dict__

    if model_type == "FTTransformer":
        model_config = FTTransformerConfig(
            task="classification",
            learning_rate=1e-3,
            head="LinearHead",
            head_config=head_config,
        )
    elif model_type == "TabNet":
        model_config = TabNetModelConfig(
            task="classification",
            learning_rate=1e-3,
            head="LinearHead",
            head_config=head_config,
        )
    elif model_type == "TabTransformer":
        model_config = TabTransformerConfig(
            task="classification",
            learning_rate=1e-3,
            head="LinearHead",
            head_config=head_config,
        )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    tabular_model.fit(train=train)
    result = tabular_model.evaluate(test)
    
    return tabular_model, result, test

def model_inference(dataset_name, model_type, index):
    if dataset_name == "COMPAS":
        data, cat_cols, num_cols, target = process_compas_data()
    elif dataset_name == "IoT":
        data, cat_cols, num_cols, target = process_iot_data()
    elif dataset_name == "Product":
        data, cat_cols, num_cols, target = process_product_data()
    else:
        return "Invalid dataset selection"
    
    # Handling missing values by filling them with mean (for numerical columns) or mode (for categorical columns)
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
    
    model_key = f"{model_type}_{dataset_name}"
    if model_key not in models:
        tabular_model, _, test = train_model(data, cat_cols, num_cols, target, model_type, dataset_name)
        models[model_key] = tabular_model
        test_datasets[model_key] = test
    else:
        tabular_model = models[model_key]
        test = test_datasets[model_key]

    sample_data = data.iloc[int(index), :-1].to_frame().transpose()
    sample_label = data.iloc[int(index), -1]
    
    prediction = tabular_model.predict(sample_data)
    
    return sample_data, sample_label, prediction

def explain_instance(dataset_name, model_type, index):
    if dataset_name == "COMPAS":
        data, cat_cols, num_cols, target = process_compas_data()
    elif dataset_name == "IoT":
        data, cat_cols, num_cols, target = process_iot_data()
    elif dataset_name == "Product":
        data, cat_cols, num_cols, target = process_product_data()
    else:
        return "Invalid dataset selection"
    
    # Handling missing values by filling them with mean (for numerical columns) or mode (for categorical columns)
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
    
    model_key = f"{model_type}_{dataset_name}"
    if model_key not in models:
        tabular_model, _, test = train_model(data, cat_cols, num_cols, target, model_type, dataset_name)
        models[model_key] = tabular_model
        test_datasets[model_key] = test
    else:
        tabular_model = models[model_key]
        test = test_datasets[model_key]
    
    def model_prediction_wrapper(x):
        features = test.columns[:-1]
        data_df = pd.DataFrame(x, columns=features)
        predictions = tabular_model.predict(data_df)
        class_probabilities = predictions.iloc[:, 0].values
        return class_probabilities
    
    background_data = test.iloc[:, :-1].sample(100, random_state=42).values
    explainer = shap.KernelExplainer(model_prediction_wrapper, background_data)
    instance_to_explain = test.iloc[int(index), :-1].values.reshape(1, -1)
    shap_values = explainer.shap_values(instance_to_explain, nsamples=100)
    
    feature_names = test.columns[:-1]
    instance_shap_values = dict(zip(feature_names, shap_values[0]))
    
    return instance_shap_values

def evaluate_global_shap_values(dataset_name, model_type, sample_size):
    if dataset_name == "COMPAS":
        data, cat_cols, num_cols, target = process_compas_data()
    elif dataset_name == "IoT":
        data, cat_cols, num_cols, target = process_iot_data()
    elif dataset_name == "Product":
        data, cat_cols, num_cols, target = process_product_data()
    else:
        return "Invalid dataset selection"
    
    # Handling missing values by filling them with mean (for numerical columns) or mode (for categorical columns)
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
    
    model_key = f"{model_type}_{dataset_name}"
    if model_key not in models:
        tabular_model, _, test = train_model(data, cat_cols, num_cols, target, model_type, dataset_name)
        models[model_key] = tabular_model
        test_datasets[model_key] = test
    else:
        tabular_model = models[model_key]
        test = test_datasets[model_key]
    
    def model_prediction_wrapper(x):
        features = test.columns[:-1]
        data_df = pd.DataFrame(x, columns=features)
        predictions = tabular_model.predict(data_df)
        class_probabilities = predictions.iloc[:, 0].values
        return class_probabilities
    
    background_data = test.iloc[:, :-1].sample(100, random_state=42).values
    explainer = shap.KernelExplainer(model_prediction_wrapper, background_data)
    subset_X_test = test.iloc[:, :-1].sample(int(sample_size), random_state=42).values
    shap_values_subset = explainer.shap_values(subset_X_test, nsamples=100)
    
    global_shap_values = np.abs(shap_values_subset).mean(axis=0)
    feature_names = test.columns[:-1]
    global_shap_dict = dict(zip(feature_names, global_shap_values))
    
    def compute_stability_score(shap_values):
        num_instances, num_features = shap_values.shape
        stability_scores = np.zeros(num_features)
        for d in range(num_features):
            diff_sum = 0
            count = 0
            for i in range(num_instances):
                for j in range(i + 1, num_instances):
                    diff = np.abs(shap_values[i, d] - shap_values[j, d])
                    diff_sum += diff
                    count += 1
            stability_scores[d] = diff_sum / count if count > 0 else 0
        overall_stability_score = np.mean(stability_scores)
        return overall_stability_score
    
    overall_stability_score = compute_stability_score(shap_values_subset)
    
    return global_shap_dict, overall_stability_score


# Vision

import gradio as gr
import json
import asyncio
from fastapi import HTTPException
import httpx
import pandas as pd
from io import StringIO, BytesIO
from PIL import Image
import requests
import numpy as np
import torch
import os
from transformers import (
    SegformerForImageClassification, SwinForImageClassification, 
    CvtForImageClassification, ViTForImageClassification, 
    ConvNextForImageClassification
)
from torchvision.models import resnet50
from torchvision import transforms
from pytorch_grad_cam import (
    GradCAM, HiResCAM, GradCAMPlusPlus, XGradCAM, LayerCAM, GradCAMElementWise
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from functools import partial

# Load ImageNet class index
with open('imagenet_class_index.json', 'r') as f:
    imagenet_class_index = json.load(f)

label_to_index_description = {v[0]: (k, v[1]) for k, v in imagenet_class_index.items()}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def ensure_rgb(img):
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img

# Transformations and reshape transform functions for each model
def get_transform(model_name):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if model_name == "nvidia/mit-b0":
        transform = transforms.Compose([
            transforms.Resize((480, 640)),  # Adjust size for Segformer
            transforms.ToTensor(),
            normalize
        ])
        def reshape_transform(tensor, width, height):
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
            result = result.transpose(2, 3).transpose(1, 2)
            return result

    elif model_name == "microsoft/swin-large-patch4-window12-384-in22k":
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Adjust size for Swin
            transforms.ToTensor(),
            normalize
        ])
        def reshape_transform(tensor, width, height):
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
            result = result.transpose(2, 3).transpose(1, 2)
            return result

    elif model_name == "microsoft/cvt-13":
        transform = transforms.Compose([
            transforms.Resize((480, 640)),  # Adjust size for CVT
            transforms.ToTensor(),
            normalize
        ])
        def reshape_transform(tensor, model, width, height):
            tensor = tensor[:, 1:, :]
            tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
            return tensor.transpose(2, 3).transpose(1, 2)

    elif model_name == "google/vit-large-patch32-384":
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Adjust size for ViT
            transforms.ToTensor(),
            normalize
        ])
        def reshape_transform(x):
            activations = x[:, 1:, :]
            activations = activations.view(activations.shape[0], 12, 12, activations.shape[2])
            activations = activations.transpose(2, 3).transpose(1, 2)
            return activations

    elif model_name == "facebook/convnext-tiny-224":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size for ConvNext
            transforms.ToTensor(),
            normalize
        ])
        def reshape_transform(tensor, model):
            batch, features, height, width = tensor.shape
            tensor = tensor.transpose(1, 2).transpose(2, 3)
            norm = model.convnext.layernorm(tensor)
            return norm.transpose(2, 3).transpose(1, 2)

    else:  # Default transformation for resnet50
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor(),          # Convert the image to a PyTorch tensor
            normalize
        ])
        reshape_transform = None

    return transform, reshape_transform


# Helper function to apply Grad-CAM
def run_grad_cam_on_image(model, target_layer, targets_for_gradcam, input_tensor, input_image, reshape_transform, method):
    with method(model=model, target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets_for_gradcam)[0]
        
        # Resize input image to match the dimensions of the grayscale_cam
        input_image_resized = input_image.resize((grayscale_cam.shape[1], grayscale_cam.shape[0]))
        visualization = show_cam_on_image(np.array(input_image_resized) / 255.0, grayscale_cam, use_rgb=True)
        return visualization

# Function to get the class label
def get_class_label(idx):
    return imagenet_class_index[str(idx)][1]

# Function to predict the class of an image
def predict(image_path, model_name):
    img = Image.open(image_path).convert('RGB')
    transform, _ = get_transform(model_name)
    img_tensor = transform(img).unsqueeze(0).to(device)

    if model_name == "resnet50":
        model = resnet50(pretrained=True).to(device)
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_idx = torch.max(output, 1)
    else:
        model = {
            "nvidia/mit-b0": SegformerForImageClassification.from_pretrained("nvidia/mit-b0").to(device),
            "microsoft/swin-large-patch4-window12-384-in22k": SwinForImageClassification.from_pretrained(model_name).to(device),
            "microsoft/cvt-13": CvtForImageClassification.from_pretrained("microsoft/cvt-13").to(device),
            "google/vit-large-patch32-384": ViTForImageClassification.from_pretrained("google/vit-large-patch32-384").to(device),
            "facebook/convnext-tiny-224": ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224").to(device)
        }[model_name]
        
        model.eval()
        with torch.no_grad():
            logits = model(img_tensor).logits
            _, predicted_idx = torch.max(logits, 1)
    
    predicted_label = get_class_label(predicted_idx.item())
    return predicted_label


# Function to apply CAM
def apply_cam(image_path, method_name, model_name):
    img = Image.open(image_path).convert('RGB')
    transform, reshape_transform = get_transform(model_name)
    img_tensor = transform(img).unsqueeze(0).to(device)

    CAM_ALGORITHMS = {
        "GradCAM": GradCAM,
        "HiResCAM": HiResCAM,
        "GradCAMPlusPlus": GradCAMPlusPlus,
        "XGradCAM": XGradCAM,
        "LayerCAM": LayerCAM,
        "GradCAMElementWise": GradCAMElementWise
    }

    if model_name == "resnet50":
        model = resnet50(pretrained=True).to(device)
        target_layer = model.layer4[-1].conv3
    elif model_name == "nvidia/mit-b0":
        model = SegformerForImageClassification.from_pretrained(model_name).to(device)
        target_layer = model.segformer.encoder.layer_norm[-1]
    elif model_name == "microsoft/swin-large-patch4-window12-384-in22k":
        model = SwinForImageClassification.from_pretrained(model_name).to(device)
        target_layer = model.swin.layernorm
    elif model_name == "microsoft/cvt-13":
        model = CvtForImageClassification.from_pretrained(model_name).to(device)
        target_layer = model.cvt.encoder.stages[-1].layers[-2]
    elif model_name == "google/vit-large-patch32-384":
        model = ViTForImageClassification.from_pretrained("google/vit-large-patch32-384").to(device)
        target_layer = model.vit.encoder.layer[-2].output
    elif model_name == "facebook/convnext-tiny-224":
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224").to(device)
        target_layer = model.convnext.encoder.stages[-1].layers[-1]
    # Add more models here if necessary

    model.eval()

    if method_name == "ALL":
        results = []
        for method in CAM_ALGORITHMS.values():
            visualization = run_grad_cam_on_image(
                model=model,
                target_layer=target_layer,
                targets_for_gradcam=[ClassifierOutputTarget(0)],  # Dummy target
                input_tensor=img_tensor,
                input_image=img,
                method=method,
                reshape_transform=reshape_transform if reshape_transform else None
            )
            results.append(visualization)
        combined_result = np.hstack(results)
        return combined_result
    else:
        method = CAM_ALGORITHMS[method_name]
        result = run_grad_cam_on_image(
            model=model,
            target_layer=target_layer,
            targets_for_gradcam=[ClassifierOutputTarget(0)],  # Dummy target
            input_tensor=img_tensor,
            input_image=img,
            method=method,
            reshape_transform=reshape_transform if reshape_transform else None
        )
        return result



# text
import requests
from io import BytesIO
from PIL import Image

def fetch_codevul_xai(dataset):
    url = "http://0.0.0.0:8005/codevul_xai"
    response = requests.post(url, json={"dataset": dataset})
    
    if response.status_code == 200:
        result = response.json()
        output_json = result["output_json"]
        plot_image_path = result["plot_image"]
        plot_image = Image.open(plot_image_path)
        return json.dumps(output_json, indent=4), plot_image
    else:
        return "Error fetching results", None


# Create the main Gradio app with tabs
with gr.Blocks() as app:
    # Add title and description
    gr.Markdown("# XAIport")
    gr.Markdown("Welcome to XAIport! This application allows you to manage and configure various tasks related to Explainable AI (XAI). Use the tabs below to navigate through different functionalities.")

    with gr.Tabs():
        with gr.TabItem("Task pipeline"):
            json_input = gr.Textbox(lines=20, label="Fill Task Configuration", value=default_json)
            output_text = gr.Textbox(label="Status")
            save_button = gr.Button("Submit Task")

            save_button.click(
                fn=save_and_run_pipeline,
                inputs=json_input,
                outputs=output_text
            )
            
            results_button = gr.Button("Fetch Results")
            results_dataframe = gr.DataFrame(label="Results")
            results_plot = gr.Image(label="Results Plot")
            
            results_button.click(
                fn=lambda config: fetch_results(config),
                inputs=json_input,
                outputs=[results_dataframe, results_plot]
            )
        

        with gr.TabItem("Tabular"):
            gr.Markdown("## Select a Dataset")
            dataset_dropdown = gr.Dropdown(
                choices=["COMPAS", "IoT", "Product"],
                label="Select Dataset",
                value="COMPAS"
            )
            dataset_display = gr.DataFrame(label="Dataset Preview")

            dataset_dropdown.change(
                fn=handle_dataset_selection,
                inputs=dataset_dropdown,
                outputs=dataset_display
            )

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
            gr.Markdown("This tab will contain functionalities for text data, code vulnerability case study.")
            
            dataset_dropdown = gr.Dropdown(choices=["MSR"], label="Select Dataset", value="MSR")
            fetch_button = gr.Button("Fetch Results")
            output_json_display = gr.JSON(label="Output JSON")
            plot_image_display = gr.Image(label="Similarity Plot")

            fetch_button.click(
                fn=fetch_codevul_xai,
                inputs=dataset_dropdown,
                outputs=[output_json_display, plot_image_display]
            )
      
        with gr.TabItem("Vision"):
            gr.Markdown("## Upload an Image for Classification and GradCAM Explanation")
            image_input = gr.Image(type="filepath", label="Upload Image")
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
app.launch(share=True)
