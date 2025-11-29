# Grading-at-Scale

**Thesis Project**: Using Large Language Models for Evaluation of Short Student Answers Based on Course Materials

## Overview

This project implements a comprehensive system for automatically grading short student answers using Small Language Models (~1b paramters). It supports both zero-shot evaluation using DSPy and fine-tuned models using LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning. The system performs 3-way classification to determine if student answers are incorrect, partially correct, or correct.

## Purpose

This project is designed to:

- Automatically grade short student answers based on course materials
- Generate synthetic training data using LLMs
- Fine-tune language models efficiently using LoRA for answer grading
- Evaluate models with comprehensive metrics including accuracy, F1 scores, and quadratic weighted kappa
- Track experiments using MLflow for reproducibility

## Setup Instructions

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (for fine-tuning) or CPU-only mode available
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))

### Installation

```bash
# Install dependencies using uv
uv sync

# Install development dependencies (for linting, etc.)
uv sync --group dev
```

### Environment Variables

If you want to generate synthetic data, test proprietary models like GPT-4o, or use models through VLLM or OLLAMA for generation and evalution please create the `.env`. This is not needed for fine-tuning and evaluting models from Hugging Face.

Create a `.env` file in the project root with the following variables:


```bash
# Azure OpenAI (for GPT-4o, GPT-4o-mini)
AZURE_API_KEY=your_azure_api_key
AZURE_API_BASE=https://your-resource.openai.azure.com/

# Ollama (optional, for local models)
OLLAMA_API_BASE=http://localhost:11434

# vLLM (optional, for hosted vLLM models)
VLLM_API_BASE=http://localhost:8000
```

### SciEntsBank Data Setup

Simply run the following, and you are ready to go:

```bash
uv run src/data_prep/prepare_scientsbank.py
```

### Custom Data Setup

Note: for a simple test of the pipeline, just use the SciEntsBank dataset, for its simple setup.

Expected CSV format (semicolon-separated):

- Columns: `task_id`, `question`, `reference_answer`, `topic`, `student_answer`, `labels`
- Labels: `"incorrect"` (0), `"partial"` (1), `"correct"` (2)

Steps

1. Place dataset in the `data/` directory
2. (Optional, if not done already) Split data into train, val, test-set by running

```bash
uv run src/data_prep/test_train_split.py
```

3. Example structure:

```bash
data/
├── gras/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── SciEntsBank_3way/
    ├── train.csv
    └── ...
```


### MLflow Setup

MLflow tracking is configured automatically using SQLite (`mlflow.db` in the project root). To view results:

```bash
# Start MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000` in your browser.

## Usage Examples

### Data Generation

Generate synthetic student answers from reference questions:

```bash
uv run src/data_prep/answer_generation.py
```

This reads configuration from `configs/answer_generation.yaml` and generates synthetic answers based on the specified parameters. Requires prepared questions and reference answers, which can be served in json and converted to csv with `json_tasks_to.py`

### Fine-tuning

#### Single Model Training

```bash
uv run src/finetuning/dispatch.py
```

Uses configuration from `configs/finetuning.yaml`. Set `TRAINING_CONFIG_PATH` environment variable to use a different config:

```bash
TRAINING_CONFIG_PATH=configs/custom_training.yaml uv run src/finetuning/dispatch.py
```

We currently do not support multiple GPUs, so please use the environment flag `CUDA_VISIBLE_DEVICES=0` if you have multiple ones running, to only use one. 

It is possible to queue multiple runs with different models and seeds in `configs/finetuning.yaml` under the `dispatcher` section.

#### Grid Search

Perform hyperparameter grid search:

```bash
uv run src/finetuning/lora_gridsearch.py
```

Configure the search space in `configs/lora_gridsearch.yaml`.

### Evaluation

#### Evaluate Fine-tuned Model

Evaluate a fine-tuned model on a test set:

```bash
uv run src/evaluation/local.py
```

Configuration is read from `configs/evaluation_local.yaml`. Supports:
- Loading adapters from HuggingFace Hub or local files
- CPU-only evaluation mode (`enforce_cpu: true`), which was used for Experiment III

#### Zero-shot Evaluation with DSPy

Evaluate a model from an API (currently supports Azure, Ollama and VLLM) without fine-tuning using DSPy:

```bash
uv run src/evaluation/dspy_eval.py
```

Uses configuration from `configs/dspy_eval.yaml`. Supports single question or batch evaluation modes.

If you want to add other models, do so in `src/model_builder.py`



## Project Structure

```
grading-at-scale/
├── configs/              # YAML configuration files
│   ├── base.yaml
│   ├── answer_generation.yaml
│   ├── finetuning.yaml
│   ├── evaluation_local.yaml
│   ├── dspy_eval.yaml
│   └── lora_gridsearch.yaml
├── data/                 # Datasets (CSV files)
│   ├── gras/
│   ├── SciEntsBank_3way/
│   └── raw/              # Raw data files
├── results/              # Training outputs and results
│   ├── peft_output/      # LoRA adapter outputs
│   └── ...
├── src/
│   ├── data_prep/        # Data preparation scripts
│   │   ├── answer_generation.py
│   │   ├── json_tasks_to_csv.py
│   │   ├── prepare_scientsbank.py
│   │   ├── train_test_split.py
│   │   └── signatures.py
│   ├── finetuning/       # LoRA fine-tuning scripts
│   │   ├── lora.py
│   │   ├── lora_gridsearch.py
│   │   └── dispatch.py
│   ├── evaluation/       # Evaluation scripts
│   │   ├── local.py
│   │   ├── dspy_eval.py
│   │   └── signatures.py
│   ├── logic/            # Logic-related utilities
│   ├── plots/            # Plotting utilities
│   ├── scripts/          # Utility scripts
│   ├── common.py         # Shared utilities
│   ├── model_builder.py  # DSPy model builder
│   └── mlflow_config.py  # MLflow configuration
├── pyproject.toml        # Project dependencies and metadata
├── uv.lock               # Locked dependency versions
├── mlflow.db             # MLflow SQLite database
└── README.md             # This file
```


## Code Overview and Structure

### Core Modules (`src/`)

- **`common.py`**: Shared utilities for data loading, tokenization, model setup, training, and evaluation metrics. Contains functions for:
  - Loading and preprocessing datasets from CSV files
  - Tokenizing datasets with optional reference answer inclusion
  - Setting up models and tokenizers
  - Computing evaluation metrics (accuracy, F1, quadratic weighted kappa)
  - Detailed evaluation with per-topic metrics

- **`model_builder.py`**: Builds DSPy language models from configuration. Supports multiple backends:
  - Azure OpenAI (GPT-4o, GPT-4o-mini)
  - Ollama (llama3.2:3b, llama3.2:1b)
  - vLLM (hosted models like Qwen, Llama, GPT-2, Flan-T5)

- **`mlflow_config.py`**: MLflow tracking setup and configuration. Handles SQLite-based tracking URI resolution.

### Data Preparation (`src/data_prep/`)

- **`json_tasks_to_csv.py`**: Converts JSON task files to CSV format. Processes all JSON files in the raw tasks directory and creates a unified CSV with columns: `question`, `answer`, `topic`.

- **`answer_generation.py`**: Generates synthetic student answers using DSPy and LLMs. Supports three generation modes:
  - `single`: Generate one answer at a time
  - `per_question`: Generate multiple answers per question
  - `all`: Generate answers for all questions at once
  - Generates correct, partial, and incorrect answers based on configuration

- **`train_test_split.py`**: Splits datasets into train/val/test sets by `task_id` to ensure no data leakage. Supports stratified splitting by topic.

- **`prepare_scientsbank.py`**: Prepares SciEntsBank dataset from HuggingFace. Converts 5-way classification labels to 3-way (incorrect, partial, correct).

- **`signatures.py`**: DSPy signatures for answer generation:
  - `CorrectAnswerGenerator`: Generates correct student answers
  - `PartialAnswerGenerator`: Generates partially correct answers
  - `IncorrectAnswerGenerator`: Generates incorrect answers
  - Supports batch generation variants (`*All`, `*PerQuestion`)

### Fine-tuning (`src/finetuning/`)

- **`lora.py`**: Main LoRA fine-tuning script using PEFT (Parameter-Efficient Fine-Tuning). Features:
  - Supports multiple models (Qwen, Llama, GPT-2, Flan-T5)
  - Model-specific hyperparameter configuration
  - Early stopping support
  - MLflow experiment tracking
  - Optional model saving to HuggingFace Hub

- **`lora_gridsearch.py`**: Hyperparameter grid search for LoRA training. Explores combinations of:
  - Learning rates
  - LoRA rank (r) values
  - LoRA alpha ratios
  - LoRA dropout values
  - Batch sizes
  - Selects best combination based on optimization metric

- **`dispatch.py`**: Dispatcher for running multiple training runs with different models and seeds. Supports:
  - Multiple models from configuration
  - Custom seed lists or random seed generation
  - Parallel execution management

### Evaluation (`src/evaluation/`)

- **`local.py`**: Evaluates fine-tuned models locally with comprehensive metrics:
  - Overall metrics: accuracy, macro F1, weighted F1, quadratic weighted kappa
  - Per-class metrics: precision, recall, F1 for each label
  - Per-topic metrics: topic-specific performance evaluation
  - Confusion matrix visualization
  - CPU-only mode support for environments without GPU
  - Timing metrics (examples/min, time per example)

- **`dspy_eval.py`**: Zero-shot evaluation using DSPy (for non-fine-tuned models). Features:
  - Single question or batch evaluation modes
  - MLflow experiment tracking
  - Comprehensive metrics and visualizations
  - Support for multiple evaluation runs

- **`signatures.py`**: DSPy signatures for grading/evaluation:
  - `GraderSingle`: Grade a single answer
  - `GraderPerQuestion`: Grade multiple answers per question
  - Supports optional reference answer inclusion

### Configuration (`configs/`)

All configurations use OmegaConf YAML files with hierarchical merging:

- **`base.yaml`**: Base configuration shared across all modules:
  - Project seed
  - Data and output directory paths
  - MLflow tracking URI (SQLite by default)

- **`answer_generation.yaml`**: Synthetic data generation parameters:
  - Generation model selection
  - Number of answers per category (correct/partial/incorrect)
  - Generation mode (single/per_question/all)
  - Reference answer passing configuration

- **`finetuning.yaml`**: LoRA training hyperparameters:
  - Model selection and dispatcher configuration
  - Dataset configuration
  - LoRA parameters (r, alpha, dropout, target_modules)
  - Training hyperparameters (epochs, batch size, learning rate, etc.)
  - Model-specific parameter overrides

- **`evaluation_local.yaml`**: Local evaluation settings:
  - Model and adapter configuration
  - Dataset path and sampling options
  - CPU enforcement and timing options
  - MLflow reporting configuration

- **`dspy_eval.yaml`**: DSPy evaluation configuration:
  - Model and mode selection
  - Dataset configuration
  - Evaluation run count

- **`lora_gridsearch.yaml`**: Grid search parameters:
  - Grid search space definition
  - Optimization metric selection
  - Dispatcher configuration

## Configuration Guide

### Configuration System

All configurations use OmegaConf YAML files with hierarchical merging:

1. Base configuration (`configs/base.yaml`) is always loaded first
2. Module-specific configurations are merged on top
3. Environment variables can override specific values

### Key Configuration Patterns

#### Model-Specific Hyperparameters

In `configs/finetuning.yaml`, you can specify model-specific parameters:

```yaml
model_specific_params:
  Qwen/Qwen3-0.6B:
    batch_size:
      train: 16
    learning_rate: 0.0005
```

#### Adapter Loading

In `configs/evaluation_local.yaml`, configure adapter source:

```yaml
adapter:
  source: hub  # Options: 'local', 'hub', or 'none'
  huggingface_username: your_username
  dataset_trained_on_name: gras
```

#### Data Sampling

For faster evaluation, use data sampling:

```yaml
dataset:
  sample_fraction: 0.1  # Use 10% of data
  sample_seed: 42
```

## Key Features

- **Parameter-Efficient Fine-tuning**: Uses LoRA for efficient model fine-tuning
- **Synthetic Data Generation**: Generates training data using LLMs 
- **Comprehensive Evaluation Metrics**:
  - Overall: accuracy, macro F1, weighted F1, quadratic weighted kappa
  - Per-class: precision, recall, F1 for each label
  - Per-topic: topic-specific performance evaluation
- **MLflow Experiment Tracking**: All experiments are tracked with parameters, metrics, and artifacts
- **CPU-Only Evaluation Mode**: Supports evaluation on CPU-only environments
- **Multiple Model Backends**: Supports Azure OpenAI, Ollama, and vLLM backends
- **Flexible Configuration**: Hierarchical YAML configuration with model-specific overrides
- **Reproducibility**: Seed control and deterministic data splitting by task_id

## Dependencies

Key dependencies (see `pyproject.toml` for complete list):

- **Core ML**: `torch`, `transformers`, `peft`, `datasets`
- **LLM Framework**: `dspy`
- **Experiment Tracking**: `mlflow`
- **Data Processing**: `pandas`, `numpy`, `scikit-learn`
- **Configuration**: `omegaconf`
- **Visualization**: `matplotlib`, `seaborn`
- **Utilities**: `tqdm`, `accelerate`, `bitsandbytes`

## Important Notes

### GPU Training

- Currently supports single GPU training only
- Always use `CUDA_VISIBLE_DEVICES=0` before running training commands
- For multi-GPU setups, modify the code to support distributed training

### MLflow Tracking

- Uses SQLite database (`mlflow.db`) by default
- Tracking URI can be configured in `configs/base.yaml`
- Start MLflow UI with: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

### Model Saving

- Models can be saved locally (`save_model_locally: true` in config)
- Models can be pushed to HuggingFace Hub (`push_to_hub: true` in config)
- Adapters are saved separately from base models

### Data Format

- All CSV files use semicolon (`;`) as separator
- Labels must be: `"incorrect"`, `"partial"`, `"correct"` (case-insensitive)
- `task_id` is used to prevent data leakage between splits

## AI Usage Disclosure

A substantial part of this codebase has been created with the help of generative AI tools (Claude Sonnet 4, Composer 1). Usage includes but is not limited to: refactoring, boilerplate generation, writing of commit messages, translating comments to code. All code has been manually reviewedm, verified and tested by the author, to ensure correctness.

## Contributing

This is a thesis project. For questions or issues, please contact the project maintainer at <mail@lucasaur.com>.
