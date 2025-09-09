# Awesome Open-Source AI
<!-- markdownlint-disable MD013 -->

A curated list of notable open-source AI repositories organized by domain. Use these examples as starting points when populating the main collection.

> **Note:** Always verify each project's license and terms of use before using it in production.

---

## Table of Contents

* [Overview](#overview)
* [Multimodal](#multimodal)
* [Computer Vision](#computer-vision)
* [Natural Language Processing (NLP)](#natural-language-processing-nlp)
* [Audio & Speech](#audio--speech)
* [Reinforcement Learning (RL)](#reinforcement-learning-rl)
* [Tabular & MLOps](#tabular--mlops)
* [Other & Emerging](#other--emerging)
* [Models & Checkpoints](#models--checkpoints)
* [Datasets & Benchmarks](#datasets--benchmarks)

---

## Overview

This document lists representative, widely used open-source repositories organized by domain. Each entry follows this format:

* **Name/Repo** — Short description. `Repo: <URL>`

Keep entries brief and factual. When adding new projects to the main repo, include the license and a one-line justification describing why the project is useful.

---

## Multimodal

* **openai/CLIP** — Zero-shot image–text embeddings for retrieval and multimodal tasks. `Repo: https://github.com/openai/CLIP` — ⭐ 30,601 | License: MIT
* **Salesforce/BLIP** — Bootstrapping Language–Image Pre-training for image captioning and VQA. `Repo: https://github.com/salesforce/BLIP` — ⭐ 5,468 | License: BSD-3-Clause
* **openai/whisper** — High-quality speech-to-text models often used within multimodal pipelines. `Repo: https://github.com/openai/whisper` — ⭐ 87,907 | License: MIT
* **mlfoundations/open\_clip** — Community implementations and checkpoints for CLIP-like models. `Repo: https://github.com/mlfoundations/open_clip` — ⭐ 12,556 | License: MIT
* **OpenGVLab/InternVL** — Large-scale vision-language model for diverse multimodal tasks. `Repo: https://github.com/OpenGVLab/InternVL` — ⭐ 9,058 | License: MIT
* **huggingface/diffusers** — Toolkit for state-of-the-art diffusion models across imaging and audio. `Repo: https://github.com/huggingface/diffusers` — ⭐ 30,669 | License: Apache-2.0

---

## Computer Vision

* **facebookresearch/detectron2** — Modular object detection and segmentation framework. `Repo: https://github.com/facebookresearch/detectron2` — ⭐ 32,709 | License: Apache-2.0
* **open-mmlab/mmdetection** — Flexible object detection toolbox with many model implementations. `Repo: https://github.com/open-mmlab/mmdetection` — ⭐ 31,650 | License: Apache-2.0
* **ultralytics/yolov5** — YOLOv5 implementation for fast and accurate object detection. `Repo: https://github.com/ultralytics/yolov5` — ⭐ 55,290 | License: AGPL-3.0
* **rwightman/pytorch-image-models (timm)** — Collection of image model backbones and utilities for classification. `Repo: https://github.com/rwightman/pytorch-image-models` — ⭐ 35,207 | License: Apache-2.0
* **facebookresearch/segment-anything** — Foundation model for object segmentation with minimal prompts. `Repo: https://github.com/facebookresearch/segment-anything` — ⭐ 51,770 | License: Apache-2.0
* **open-mmlab/mmsegmentation** — Semantic segmentation toolbox with extensive model zoo. `Repo: https://github.com/open-mmlab/mmsegmentation` — ⭐ 9,217 | License: Apache-2.0

---

## Natural Language Processing (NLP)

* **huggingface/transformers** — Transformer models and inference pipelines for NLP and multimodal tasks. `Repo: https://github.com/huggingface/transformers` — ⭐ 149,333 | License: Apache-2.0
* **facebookresearch/fairseq** — Research toolkit for sequence modeling (translation, language models). `Repo: https://github.com/facebookresearch/fairseq` — ⭐ 31,775 | License: MIT
* **UKPLab/sentence-transformers** — Sentence embedding models for semantic search and clustering. `Repo: https://github.com/UKPLab/sentence-transformers` — ⭐ 17,494 | License: Apache-2.0
* **RasaHQ/rasa** — Open-source framework for building conversational AI and chatbots. `Repo: https://github.com/RasaHQ/rasa` — ⭐ 20,660 | License: Apache-2.0
* **openai/gpt-2** — Open-source training code and checkpoints for GPT-2 language models. `Repo: https://github.com/openai/gpt-2` — ⭐ 24,157 | License: MIT
* **allenai/allennlp** — Modular NLP research framework built on PyTorch. `Repo: https://github.com/allenai/allennlp` — ⭐ 11,876 | License: Apache-2.0

---

## Audio & Speech

* **openai/whisper** — Robust automatic speech recognition (ASR) models. `Repo: https://github.com/openai/whisper` — ⭐ 87,907 | License: MIT
* **pytorch/audio** — Audio utilities for PyTorch: datasets, transforms, and model examples. `Repo: https://github.com/pytorch/audio` — ⭐ 2,729 | License: BSD-2-Clause
* **espnet/espnet** — End-to-end speech processing toolkit supporting ASR and TTS. `Repo: https://github.com/espnet/espnet` — ⭐ 9,446 | License: Apache-2.0
* **mozilla/DeepSpeech** — Reference end-to-end speech-to-text implementation (legacy but useful). `Repo: https://github.com/mozilla/DeepSpeech` — ⭐ 26,594 | License: MPL-2.0
* **coqui-ai/TTS** — Multilingual neural text-to-speech system with pretrained models. `Repo: https://github.com/coqui-ai/TTS` — ⭐ 42,475 | License: MPL-2.0
* **CorentinJ/Real-Time-Voice-Cloning** — Clone voices using a three-stage deep learning pipeline. `Repo: https://github.com/CorentinJ/Real-Time-Voice-Cloning` — ⭐ 54,968 | License: Unknown

---

## Reinforcement Learning (RL)

* **DLR-RM/stable-baselines3** — Implementations of popular RL algorithms in PyTorch. `Repo: https://github.com/DLR-RM/stable-baselines3` — ⭐ 11,479 | License: MIT
* **openai/baselines** — Classic reference implementations for reinforcement learning. `Repo: https://github.com/openai/baselines` — ⭐ 16,431 | License: MIT
* **ray-project/ray** — Scalable RL training platform and the RLlib library. `Repo: https://github.com/ray-project/ray` — ⭐ 38,847 | License: Apache-2.0
* **facebookresearch/habitat-sim** — High-performance embodied AI simulator for navigation tasks. `Repo: https://github.com/facebookresearch/habitat-sim` — ⭐ 3,153 | License: MIT
* **openai/gym** — Standard toolkit of environments for RL benchmarking. `Repo: https://github.com/openai/gym` — ⭐ 36,475 | License: Unknown
* **Unity-Technologies/ml-agents** — Bridge between Unity simulations and RL algorithms. `Repo: https://github.com/Unity-Technologies/ml-agents` — ⭐ 18,605 | License: Unknown

---

## Tabular & MLOps

* **scikit-learn/scikit-learn** — Reliable machine learning library for tabular data and classic algorithms. `Repo: https://github.com/scikit-learn/scikit-learn` — ⭐ 63,277 | License: BSD-3-Clause
* **mlflow/mlflow** — Experiment tracking, model packaging, and registry. `Repo: https://github.com/mlflow/mlflow` — ⭐ 21,964 | License: Apache-2.0
* **iterative/dvc** — Data Version Control to manage datasets and ML pipelines. `Repo: https://github.com/iterative/dvc` — ⭐ 14,861 | License: Apache-2.0
* **PrefectHQ/prefect** — Modern workflow orchestration for data and ML pipelines. `Repo: https://github.com/PrefectHQ/prefect` — ⭐ 20,296 | License: Apache-2.0
* **apache/airflow** — Workflow orchestrator for scheduling complex data and ML pipelines. `Repo: https://github.com/apache/airflow` — ⭐ 42,225 | License: Apache-2.0
* **alteryx/featuretools** — Automatic feature engineering for structured datasets. `Repo: https://github.com/alteryx/featuretools` — ⭐ 7,534 | License: BSD-3-Clause

---

## Other & Emerging

* **OpenMined/PySyft** — Tools for privacy-preserving and federated learning. `Repo: https://github.com/OpenMined/PySyft` — ⭐ 9,779 | License: Apache-2.0
* **pytorch/opacus** — Differential privacy library for PyTorch training. `Repo: https://github.com/pytorch/opacus` — ⭐ 1,847 | License: Apache-2.0
* **carla-simulator/carla** — Autonomous driving simulator for research and RL. `Repo: https://github.com/carla-simulator/carla` — ⭐ 12,987 | License: MIT
* **facebookresearch/detectron2** — Production-ready computer vision framework (also listed under Computer Vision). `Repo: https://github.com/facebookresearch/detectron2` — ⭐ 32,709 | License: Apache-2.0
* **jax-ml/jax** — High-performance numerical computing and autodiff for ML research. `Repo: https://github.com/jax-ml/jax` — ⭐ 33,380 | License: Apache-2.0
* **fastai/fastai** — High-level library simplifying training of modern deep learning models. `Repo: https://github.com/fastai/fastai` — ⭐ 27,417 | License: Apache-2.0

---

## Models & Checkpoints

* **EleutherAI/gpt-neox** — Open-source large language model implementation and training code. `Repo: https://github.com/EleutherAI/gpt-neox` — ⭐ 7,296 | License: Apache-2.0
* **CompVis/stable-diffusion** — Text-to-image diffusion model repository; check license and usage rules. `Repo: https://github.com/CompVis/stable-diffusion` — ⭐ 71,434 | License: CreativeML Open RAIL-M
* **openai/CLIP** — Image–text model with available checkpoints (listed above under Multimodal). `Repo: https://github.com/openai/CLIP` — ⭐ 30,601 | License: MIT
* **meta-llama/llama** — Reference implementation and weights for LLaMA models. `Repo: https://github.com/meta-llama/llama` — ⭐ 58,713 | License: Unknown
* **Stability-AI/StableLM** — Open-source language models for research and experimentation. `Repo: https://github.com/Stability-AI/StableLM` — ⭐ 15,804 | License: Apache-2.0
* **openlm-research/open_llama** — Open reproduction of LLaMA weights and training code. `Repo: https://github.com/openlm-research/open_llama` — ⭐ 7,527 | License: Apache-2.0

---

## Datasets & Benchmarks

* **cocodataset/cocoapi** — COCO evaluation tools and API for object detection and segmentation. `Repo: https://github.com/cocodataset/cocoapi` — ⭐ 6,295 | License: Unknown
* **mozilla/DeepSpeech** — Includes dataset helpers and scripts for LibriSpeech and CommonVoice. `Repo: https://github.com/mozilla/DeepSpeech` — ⭐ 26,594 | License: MPL-2.0
* **huggingface/datasets** — Library and utilities for using and sharing datasets reproducibly. `Repo: https://github.com/huggingface/datasets` — ⭐ 20,616 | License: Apache-2.0
* **tensorflow/datasets** — Ready-to-use dataset collection with preprocessing utilities. `Repo: https://github.com/tensorflow/datasets` — ⭐ 4,475 | License: Apache-2.0
* **openml/openml-python** — Python client for accessing OpenML's curated datasets. `Repo: https://github.com/openml/openml-python` — ⭐ 306 | License: Unknown
* **EleutherAI/lm-evaluation-harness** — Framework for standard LLM evaluations across tasks. `Repo: https://github.com/EleutherAI/lm-evaluation-harness` — ⭐ 10,053 | License: MIT

---
