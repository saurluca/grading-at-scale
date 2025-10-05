# Configuration Structure

This directory contains the refactored configuration files for the grading-at-scale ML pipeline.

## File Structure

```
configs/
├── base.yaml              # Shared settings across all modules
├── data_generation.yaml   # Data preparation configuration
├── training.yaml          # Model training/fine-tuning configuration
├── evaluation.yaml        # Model evaluation configuration
└── README.md             # This file
```

## Configuration Files

### `base.yaml`
Contains shared settings used across all pipeline modules:
- Project metadata (name, seed)
- Common paths (data directories, cache)
- Model registry (predefined model configurations)
- Label configuration for classification tasks

### `data_generation.yaml`
Configuration for synthetic data generation (`src/data_prep/`):
- Generation and teacher models
- Input/output paths
- Generation parameters (num answers, mode, chain of thought)
- Evaluation parameters for grading generated answers
- LM settings (temperature, caching)
- External dataset configuration (SciEntsBank)

**Used by:**
- `src/data_prep/create_synth_data.py`
- `src/data_prep/json_tasks_to_csv.py`

### `training.yaml`
Configuration for model training and fine-tuning (`src/optimisation/`):
- Model selection and type (vanilla, lora, lora_quantized, lora_typst)
- Dataset configuration
- Output settings
- LoRA configuration
- Quantization settings
- Training hyperparameters
- MLflow tracking
- Typst-specific settings (for Typst dataset training)

**Used by:**
- `src/optimisation/finetune_base.py` (vanilla training)
- `src/optimisation/finetune_lora.py` (LoRA training)
- `src/optimisation/finetune_lora_quantised.py` (Quantized LoRA)
- `src/optimisation/finetune_lora_typst.py` (Typst dataset LoRA)

### `evaluation.yaml`
Configuration for model evaluation (`src/evaluation/`):
- Evaluation type (API-based or local classifier)
- API evaluation settings (DSPy graders)
- Local classifier evaluation (fine-tuned models on SciEntsBank)
- Output settings

**Used by:**
- `src/evaluation/evaluate_api.py` (API-based evaluation)
- `src/evaluation/evaluate_local.py` (Local classifier evaluation)

## OmegaConf Features

The configs use OmegaConf which supports:

### 1. Variable Interpolation
Reference other config values using `${path.to.value}`:
```yaml
paths:
  data_dir: data
  synth_dir: ${paths.data_dir}/synth  # Resolves to "data/synth"
```

### 2. Config Composition
The `defaults` directive merges base config automatically:
```yaml
defaults:
  - base  # Automatically merges base.yaml
```

### 3. Command-Line Overrides
Override any config value from command line:
```bash
python finetune_lora.py training.num_epochs=3 model.base=llama_3b
```

### 4. Programmatic Merging
Merge multiple configs in Python:
```python
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/training.yaml")
# base.yaml is automatically merged due to 'defaults' directive
```

## Migration from Old Configs

### Old Structure
- `synthetic_data.yaml` - Mixed data generation and evaluation settings
- `peft_lora.yaml` - LoRA training settings
- `peft_lora_typst.yaml` - Typst-specific LoRA settings
- `vanilla.yaml` - Vanilla training settings

### New Structure
- `base.yaml` - Shared settings
- `data_generation.yaml` - Replaces data generation parts of `synthetic_data.yaml`
- `training.yaml` - Unifies `peft_lora.yaml`, `peft_lora_typst.yaml`, and `vanilla.yaml`
- `evaluation.yaml` - Replaces evaluation parts of `synthetic_data.yaml`

### Key Changes

#### Path Updates
- Old: `cfg.output_dir` → New: `cfg.output.dir`
- Old: `cfg.dataset_csv` → New: `cfg.dataset.csv_path`
- Old: `cfg.model_name` → New: `cfg.model.base`
- Old: `cfg.hf_cache_dir` → New: `cfg.paths.hf_cache_dir`

#### Training Parameters
- Old: `cfg.training.num_train_epochs` → New: `cfg.training.num_epochs`
- Old: `cfg.training.per_device_train_batch_size` → New: `cfg.training.batch_size.train`
- Old: `cfg.training.per_device_eval_batch_size` → New: `cfg.training.batch_size.eval`

#### LoRA Parameters
- Old: `cfg.lora.lora_alpha` → New: `cfg.lora.alpha`
- Old: `cfg.lora.lora_dropout` → New: `cfg.lora.dropout`
- Old: `cfg.lora.init_lora_weights` → New: `cfg.lora.init_weights`

#### Data Generation
- Old: `cfg.model_name` → New: `cfg.generation_model`
- Old: `cfg.create_mode` → New: `cfg.generation.mode`
- Old: `cfg.num_correct_answers` → New: `cfg.generation.num_correct_answers`
- Old: `cfg.output_dir` → New: `cfg.output.dir`

#### Evaluation
- Old: `cfg.teacher_model_name` → New: `cfg.api_eval.model`
- Old: `cfg.eval_mode` → New: `cfg.api_eval.mode`
- Old: `cfg.lm_temp_eval` → New: `cfg.api_eval.temperature`

## Examples

### Running Data Generation
```bash
cd src/data_prep
python create_synth_data.py
```

### Running Training
```bash
cd src/optimisation
# LoRA training
python finetune_lora.py

# Vanilla training (set model.type=vanilla in config)
python finetune_base.py

# Quantized LoRA
python finetune_lora_quantised.py

# Typst dataset training
python finetune_lora_typst.py
```

### Running Evaluation
```bash
cd src/evaluation
# API-based evaluation
python evaluate_api.py

# Local classifier evaluation
python evaluate_local.py
```

### Command-Line Overrides
```bash
# Change model
python finetune_lora.py model.base=meta-llama/Llama-3.2-3B-Instruct

# Change training epochs and batch size
python finetune_lora.py training.num_epochs=5 training.batch_size.train=8

# Change dataset
python finetune_lora.py dataset.csv_path=data/synth/student_answers_c3_p3_i3_gpt-4o_per_question.csv

# Multiple overrides
python finetune_lora.py \
  model.base=meta-llama/Llama-3.2-3B-Instruct \
  training.num_epochs=3 \
  training.batch_size.train=8 \
  lora.r=32
```

## Benefits of New Structure

1. **Clear Separation of Concerns**: Each module has its own config
2. **Reduced Duplication**: Shared settings in `base.yaml`
3. **Better Version Control**: See exactly which pipeline stage changed
4. **Easier Experimentation**: Modify training without touching data/eval configs
5. **Improved Maintainability**: Smaller, focused config files
6. **Type Safety**: Hierarchical structure makes config paths clear
