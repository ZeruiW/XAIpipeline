'''
Version 0.0.1
This is the previous try to create a server that can handle the pipeline of the XAI service.
Without the status check system, we can't correctly run the whole pipeline.
This must be reconstructed to a new version.
'''

from fastapi import FastAPI, HTTPException
import httpx
import json
import os

app = FastAPI(title="Coordination Center")


async def async_http_post(url, json_data=None, files=None):
    async with httpx.AsyncClient() as client:
        if json_data:
            response = await client.post(url, json=json_data)
        elif files:
            response = await client.post(url, files=files)
        else:
            response = await client.post(url)

        # 检查是否是307重定向响应
        if response.status_code == 307:
            redirect_url = response.headers.get('Location')
            if redirect_url:
                print(f"Redirecting to {redirect_url}")
                return await async_http_post(redirect_url, json_data, files)

        if response.status_code != 200:
            print(f"Error response: {response.text}")  # 打印出错误响应内容
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()


# 处理上传配置
async def process_upload_config(upload_config):
    for dataset_id, dataset_info in upload_config['datasets'].items():
        url = upload_config['server_url'] + f"/upload-dataset/{dataset_id}"
        local_zip_file_path = dataset_info['local_zip_path']  # 每个数据集的本地 ZIP 文件路径

        async with httpx.AsyncClient() as client:
            with open(local_zip_file_path, 'rb') as f:
                files = {'zip_file': (os.path.basename(local_zip_file_path), f)}
                response = await client.post(url, files=files)
            response.raise_for_status()

        # 可以添加更多的逻辑处理上传后的结果
        print(f"Uploaded dataset {dataset_id} successfully.")




# 处理扰动配置
async def process_perturbation_config(perturbation_config):
    url = perturbation_config['server_url']
    for dataset, settings in perturbation_config['datasets'].items():
        full_url = f"{url}/apply-perturbation/{dataset}/{settings['perturbation_type']}/{settings['severity']}"
        await async_http_post(full_url)


# 处理模型配置
async def process_model_config(model_config):
    base_url = model_config['base_url']
    for model, settings in model_config['models'].items():
        full_url = f"{base_url}/{settings['model_name']}/{model}/{settings['perturbation_type']}/{settings['severity']}"
        await async_http_post(full_url)


# 处理 XAI 配置
async def process_xai_config(xai_config):
    base_url = xai_config['base_url']
    for dataset, settings in xai_config['datasets'].items():
        dataset_id = settings.get('dataset_id', '')  # 提取 "dataset_id"
        algorithms = settings.get('algorithms', [])  # 提取 "algorithms"
        data = {
            "dataset_id": dataset_id,
            "algorithms": algorithms
        }
        print(data)
        full_url = f"{base_url}/cam_xai/"
        print(full_url)
        await async_http_post(full_url, json_data=data)




# 处理评估配置
async def process_evaluation_config(evaluation_config):
    base_url = evaluation_config['base_url']
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

# 按顺序处理每个配置步骤
async def process_pipeline_step(config, step_key, process_function):
    if step_key in config:
        await process_function(config[step_key])

# # 从配置运行整个 Pipeline，这种不够灵活，需要改进
# async def run_pipeline_from_config(config):
#     await process_pipeline_step(config, 'upload_config', process_upload_config)
#     await process_pipeline_step(config, 'perturbation_config', process_perturbation_config)
#     await process_pipeline_step(config, 'model_config', process_model_config)
#     await process_pipeline_step(config, 'xai_config', process_xai_config)
#     await process_pipeline_step(config, 'evaluation_config', process_evaluation_config)

async def run_pipeline_from_config(config):
    # 定义步骤处理函数的映射
    step_process_functions = {
        'upload_config': process_upload_config,
        'perturbation_config': process_perturbation_config,
        'model_config': process_model_config,
        'xai_config': process_xai_config,
        'evaluation_config': process_evaluation_config,
    }

    # 从config动态获取要执行的步骤列表
    for step_key in config.keys():
        process_function = step_process_functions.get(step_key)
        if process_function:
            await process_function(config[step_key])
            print(f"Completed processing {step_key}")





# 加载配置文件
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import asyncio
import json


def load_config():
    with open("/home/z/Music/devnew_xaiservice/XAIport/task_sheets/task1.json", "r") as file:
        return json.load(file)

def main():
    config = load_config()
    asyncio.run(run_pipeline_from_config(config))

if __name__ == "__main__":
    main()