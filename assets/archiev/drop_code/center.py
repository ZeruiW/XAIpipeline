import os
import dataprocess.DataProcess as DataProcess

# DataProcess 
data_processor = DataProcess.DataProcess()
image_folder = 'val_images10k'
image_files = []
for root, dirs, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

dataset_id = 'test_dataset'
storage_address = 'data'  # 指定存储地址
data_processor.upload_dataset(image_files, dataset_id, storage_address, 'image')

info = data_processor.get_dataset_info(dataset_id)
print(info)

perturbed_dataset_path = data_processor.apply_image_perturbation('test_dataset', data_processor.gaussian_noise, 3)


dataset_path0 = info['storage_address']
perturbed_dataset_path = data_processor.apply_image_perturbation('test_dataset', data_processor.gaussian_noise, 3)

dataset_paths = [
    dataset_path0,
    perturbed_dataset_path
]

# Model

import modelserver.Model_ResNet as Model_ResNet
# dataset_paths = [
#     "data/image/test_dataset",
#     "data/image/test_dataset_perturbation_gaussian_noise_3"
# ]
Model_ResNet.model_run(dataset_paths)


## XAI

import xaiserver.xai_resnet as xai_resnet
xai_resnet.xai_run(dataset_paths)


# Evaluation
from evaluation import cam_sammary
original_dataset_path = dataset_paths[0]

model_name = "resnet50"  # Replace with your actual model name
attack_name = "Original"  # Replace with your actual attack name
CAM_ALGORITHMS = ["GradCAM", "HiResCAM", "GradCAMPlusPlus", "XGradCAM", "LayerCAM"]

# base_dir0 = "results/data/image/test_dataset/resnet50"

rs_dir = os.path.join("results", original_dataset_path, model_name)

cam_sammary(rs_dir, model_name, attack_name, CAM_ALGORITHMS)