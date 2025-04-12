# TCM-RAG Project Introduction

## Project Overview
The project is a comprehensive framework for Traditional Chinese Medicine (TCM) knowledge retrieval and generation with multiple components including model fine-tuning and application.

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
  - TCM-10M dataset please refer to https://huggingface.co/datasets/NeuralPetal/TCM
## Important Notes
- Please pay attention to the `requirements.txt` files in each directory
- Different packages are required for different modules
- Note that different versions of transformers may cause errors 

## Citation
If you used this in your work, please cite:

```bibtex
@misc{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools}, 
      author={Team GLM and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{FlashRAG,
  author       = {Jiajie Jin and
                  Yutao Zhu and
                  Xinyu Yang and
                  Chenghao Zhang and
                  Zhicheng Dou},
  title        = {FlashRAG: {A} Modular Toolkit for Efficient Retrieval-Augmented Generation
                  Research},
  journal      = {CoRR},
  volume       = {abs/2405.13576},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2405.13576},
  doi          = {10.48550/ARXIV.2405.13576},
  eprinttype    = {arXiv},
  eprint       = {2405.13576},
  timestamp    = {Tue, 18 Jun 2024 09:26:37 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2405-13576.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@misc{bge_m3,
  title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
  author={Chen, Jianlv and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
  year={2023},
  eprint={2309.07597},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
