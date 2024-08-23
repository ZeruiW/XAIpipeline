from fastapi import FastAPI, HTTPException
import httpx
import json

app = FastAPI(title="Coordination Center")


async def async_http_post(url, json_data=None, files=None):
    async with httpx.AsyncClient() as client:
        if json_data:
            response = await client.post(url, json=json_data)
        elif files:
            response = await client.post(url, files=files)
        else:
            response = await client.post(url)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


# 处理上传配置
async def process_upload_config(upload_config):
    url = upload_config['server_url'] + "/upload-dataset/t1"
    zip_path = upload_config['zip_path']

    # 对于大文件，应考虑使用流式上传或更高效的方法
    async with httpx.AsyncClient() as client:
        response = await client.post(url, files={'zip_file': httpx.get(zip_path).content})

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)


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
        data = {
            "dataset_id": f"{dataset}_{settings['perturbation_type']}_{settings['severity']}",
            "algorithms": settings['algorithms']
        }
        full_url = f"{base_url}/cam_xai/"
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

# 从配置运行整个 Pipeline
async def run_pipeline_from_config(config):
    await process_pipeline_step(config, 'upload_config', process_upload_config)
    await process_pipeline_step(config, 'perturbation_config', process_perturbation_config)
    await process_pipeline_step(config, 'model_config', process_model_config)
    await process_pipeline_step(config, 'xai_config', process_xai_config)
    await process_pipeline_step(config, 'evaluation_config', process_evaluation_config)

# API 端点来触发 Pipeline
@app.post("/run_pipeline/")
async def run_pipeline():
    config = load_config()  # 加载配置
    try:
        await run_pipeline_from_config(config)
        return {"message": "Pipeline executed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 加载配置文件
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
