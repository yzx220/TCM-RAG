# TCM-RAG Project Introduction

## Project Overview
This project is a comprehensive framework for Traditional Chinese Medicine (TCM) knowledge retrieval and generation, featuring multiple components for model fine-tuning and application.

## Usage Guide

### 1. BGE-M3 Module
- Contains code for fine-tuning with TCM data
- To perform fine-tuning:
  - Use `finetune_m3.sh`
  - Configure relevant file paths and parameters

### 2. ChatGLM Module
- Includes fine-tuning code and configuration files
- For fine-tuning:
  - Set appropriate file paths and parameters in the config file
  - Run the following command:
    ```bash
    python finetune_hf.py data/data/ model/chatglm3-6b tcm_finetune_config.yaml
    ```

### 3. TCM-RAG Module
- Provides a simple example for quick startup

### 4. Data Module
- Contains various datasets for model fine-tuning:
  - Training datasets
  - Test datasets
  - TCM-10M dataset

## Important Notes
- Please pay attention to the `requirements.txt` files in each directory
- Different packages are required for different modules
- Note that different versions of transformers may cause errors 
