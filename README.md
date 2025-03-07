# XAIpipeline: A Configuration-Driven Explainable AI Toolchain

[![Watch the video](https://img.youtube.com/vi/Kq6j_wxF7wg/0.jpg)](https://youtu.be/Kq6j_wxF7wg)



## Table of Contents
- [XAIpipeline: A Configuration-Driven Explainable AI Toolchain](#xaipipeline-a-configuration-driven-explainable-ai-toolchain)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Background](#background)
  - [Key Features](#key-features)
  - [Architecture](#architecture)
  - [Supported Models and XAI Methods](#supported-models-and-xai-methods)
    - [Models](#models)
    - [XAI Methods](#xai-methods)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Task Pipeline](#task-pipeline)
    - [Tabular Data Analysis](#tabular-data-analysis)
    - [Text Analysis](#text-analysis)
    - [Vision Analysis](#vision-analysis)
    - [Video Stream Analysis](#video-stream-analysis)

## Introduction

XAIpipeline is a comprehensive Explainable AI (XAI) toolchain that addresses the growing need for transparency and interpretability in AI systems. As AI technologies become increasingly integrated into critical domains such as healthcare, finance, and autonomous systems, the ability to explain AI decisions has become paramount. XAIpipeline provides a configuration-driven approach to applying XAI techniques across diverse AI models and platforms, bridging the gap between advanced AI capabilities and the need for explainable outcomes.

## Background

The development of XAIpipeline is grounded in several key theoretical concepts and challenges in the field of Explainable AI:

1. **Black Box Problem**: Many modern AI models, particularly deep learning systems, operate as "black boxes," making decisions that are difficult for humans to interpret. XAIpipeline addresses this by implementing various XAI techniques to provide insights into model decision-making processes.

2. **Model-Agnostic and Model-Specific Explanations**: XAIpipeline supports both model-agnostic methods (e.g., LIME, SHAP) that can explain any ML model, and model-specific methods tailored to particular architectures.

3. **Local vs. Global Explanations**: The toolchain provides capabilities for both local explanations (explaining individual predictions) and global explanations (understanding overall model behavior).

4. **Evaluation of Explanations**: XAIpipeline incorporates metrics for assessing the quality and reliability of generated explanations, addressing the challenge of quantifying explanation effectiveness.

5. **Regulatory Compliance**: With the increasing focus on AI governance (e.g., EU AI Act, NIST AI Risk Management Framework), XAIpipeline supports efforts to meet emerging regulatory requirements for AI explainability.

## Key Features

1. **Unified Open APIs**: Standardized approach to apply XAI techniques across diverse AI models and platforms.
2. **Multi-Service Integration**: Interfaces with major cloud AI services and open-source model repositories.
3. **Quantifiable XAI Results**: Generates evaluations and robustness measurements for model optimization.
4. **Parallel Processing**: Supports parallel execution of XAI methods across different tasks, enhancing efficiency.
5. **Automated Execution**: Streamlines the XAI process through configurable, automated pipelines.
6. **Reproducibility**: Ensures XAI operations are recorded and reproducible through provenance data.

## Architecture

XAIpipeline implements an API-centric architecture, consisting of the following key components:

1. **Interface Layer**: RESTful API, Command-Line Interface (CLI), and Web Portal.
2. **Data Service**: Dataset management and preprocessing.
3. **Model Service**: Unified interface for cloud AI services and open-source models.
4. **XAI Methods Service**: Containerized XAI algorithms and tools.
5. **XAI Results Service**: Metrics for assessing explanation quality.

## Supported Models and XAI Methods

XAIpipeline supports a wide range of AI models and XAI methods:

### Models
- Tabular data: 6 models
- Text: 2 models
- Image: 6 models
- Video: 1 model

### XAI Methods
- Tabular: SHAP, LIME, PDP, ALE, mean-centroid prediff
- Text: 2 methods for code vulnerability classification
- Image: 5 CAM-based methods
- Video: 3 methods

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/xaipipeline.git
   cd xaipipeline
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure API servers (see [API Configuration](#api-configuration)).

## Usage

Launch the XAIpipeline application: python app.py


Access the web interface at `http://localhost:7860` (or the URL provided in the console output).

### Task Pipeline
1. Enter JSON task configuration.
2. Click "Submit Task" to execute.
3. Use "Fetch Results" to retrieve outputs.

### Tabular Data Analysis
1. Select dataset and model.
2. Enter dataset index for analysis.
3. Use "Inference" for predictions.
4. Explore XAI methods (e.g., SHAP) for explanations.

### Text Analysis
1. Choose dataset (MSR or custom).
2. Generate CWE hierarchy visualization and similarity matrix.

### Vision Analysis
1. Upload image for classification.
2. Select model and CAM algorithm.
3. Generate predictions and visual explanations.

### Video Stream Analysis
1. Upload video for attention analysis.
2. Process video and view results.

