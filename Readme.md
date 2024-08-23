
# XAIport

## Overview

XAIport is designed to deliver interpretable AI model predictions through a microservice architecture, allowing users to understand the underlying decision-making processes of AI models better. The architecture includes a User Interface, Coordination Center, Core Microservices such as Data Processing, AI Model, XAI Method, and Evaluation Services, along with a Data Persistence layer.


![Architecture Diagram](assets/architecture_diagram.webp)



## Initial Setup

### Prerequisites

- Python 3.8 or later
- FastAPI
- httpx
- uvicorn
- Dependencies as listed in `requirements.txt`

### Installation Guide

1. **Environment Setup**:
   Ensure Python is installed on your system. It's recommended to use a virtual environment for Python projects:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install Dependencies**:
   Install the necessary Python libraries with pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Clone the Repository**:
   Clone the repository to get the latest codebase:

   ```bash
   git clone https://github.com/ZeruiW/XAIport.git
   cd XAIport
   ```

### Configuration

Before running the system, configure all necessary details such as API endpoints, database connections, and other service-related configurations in a JSON format. Adjust the `config.json` file as needed.

Example `config.json`:

```json
{
  "upload_config": {
    "server_url": "http://localhost:8000",
    "datasets": {
      "dataset1": {
        "local_zip_path": "/path/to/dataset1.zip"
      }
    }
  },
  "model_config": {
    "base_url": "http://model-service-url",
    "models": {
      "model1": {
        "model_name": "ResNet50",
        "perturbation_type": "noise",
        "severity": 2
      }
    }
  }
}
```

## Running the System

### Starting the Service

Run the FastAPI application using Uvicorn as an ASGI server with the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Using the API

The system provides several RESTful APIs to support operations such as data upload, model prediction, XAI method execution, and evaluation tasks. Here are some examples of how to use these APIs:

- **Upload Dataset**:

  ```bash
  curl -X POST "http://localhost:8000/upload-dataset/dataset1" -F "file=@/path/to/dataset.zip"
  ```

- **Execute XAI Task**:

  ```bash
  curl -X POST "http://localhost:8000/cam_xai" -H "Content-Type: application/json" -d '{"dataset_id": "dataset1", "algorithms": ["GradCAM", "SmoothGrad"]}'
  ```

## Maintenance and Monitoring

### Logging

Configure appropriate logging policies to record key operations and errors within the system. This can be achieved by setting up Python's logging module to handle different log levels and outputs.

### Performance Monitoring

It is recommended to use monitoring tools like Prometheus and Grafana to track system performance and health indicators.

## Frequently Asked Questions (FAQ)

### How do I handle data upload failures?

Check if the target server is reachable and ensure that the file paths in the configuration file are correctly specified.

### How do I update API endpoints in the configuration file?

Modify the API endpoints directly in the JSON configuration file and restart the service to apply changes.

