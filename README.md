# 3R-Fold: Retrieval-Augmented RNA Secondary Structure Prediction via Deep Self-Representation and RNA Foundation Models

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Project](#running-the-project)
7. [Local Machine Setup](#local-machine-setup)
6. [Google Colab Setup](#google-colab-setup)
8. [Advanced Configuration](#advanced-configuration)
9. [Troubleshooting](#troubleshooting)
10. [Expected Results](#expected-results)

## Overview

3R-Fold is a retrieval-augmented framework for RNA secondary structure prediction that integrates RNA foundation model embeddings with representation learning and structural similarity search. In the first stage, RNA sequences are encoded using RNA-FM together with a Deep Self-Representation (DSR) model to learn structure-aware embeddings capturing global relationships between RNA sequences. In the second stage, FAISS similarity search retrieves structurally related RNAs from a knowledge base, and their secondary structures are aggregated to produce a consensus structural signal that guides folding prediction. Experiments across multiple RNA families and sequence lengths show that retrieval-based guidance substantially improves prediction accuracy compared with classical thermodynamic folding. The proposed 3R-Fold framework achieves the strongest overall performance among evaluated approaches and does not require prior knowledge of RNA family annotations. More broadly, 3R-Fold can be viewed as a modular retrieval layer that augment existing RNA folding algorithms and provides a promising direction toward scalable, data-driven RNA structure prediction

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- CUDA-compatible GPU (recommended for training)

### Recommended Requirements
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 50GB+ free disk space

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/chaimaetoubali-lab/3R-Fold
cd 3R-Fold
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n 3rfold python=3.9
conda activate 3rfold

# Using venv
python -m venv 3rfold_env
source 3rfold_env/bin/activate  # Linux/Mac
# or
3rfold_env\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

##  Configuration

### 1. Global Path Configuration

If you will run this code outside the project folder make sure to change the project Path. 

The configuration file is `config/paths.py` :

```python
# Set your base path
BASE_PATH = "./"  # Relative project path
# BASE_PATH = "/path/to/your/project"  # Full path to your python code
# BASE_PATH = "/content/drive/MyDrive/" + "/path/to/your/project"  # Google Colab

# Or use environment variable
# BASE_PATH = os.getenv("FOLDV_BASE_PATH", "/default/path")
```

### 2. Configuration Files

Main configuration is in `configs/default.yaml`:

```yaml
seed: 42

autoencoder:
  latent_dim: 64

retrieval:
  top_k: 7

training:
  epochs: 60
  lr: 0.0001
  lambda1: 0.001
  lambda2: 0.001

dataset:
  family_threshold: 200

folding:
  bonus: 1.5
  min_pair_freq: 0.4
  unpaired_bonus: 0.3
```

## ️Running The Project

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook Main_Runner.ipynb
```

### Option 2: Command Line
Convert `Main_Runner.ipynb` to `Main_Runner.py` and run it.

### Option 3: Individual Scripts

For each cell in the notebook, run the corresponding script as follows
```bash
# Build dataset
python -c "from data.build_dataset import build_dataset; build_dataset()"

# Generate embeddings
python -c "
from data.build_dataset import build_dataset
from models.pretrained_embedder import PretrainedRNAEmbedder
from retrieval.build_kb import build_kb
from config import get_results_path

dataset = build_dataset()
embedder = PretrainedRNAEmbedder()
build_kb(dataset, embedder, cache_path=f'{get_results_path()}/embeddings.pt')
"
```

## Local Machine Setup

### 1. Environment Setup
```bash
# Create project directory
mkdir -p ~/projects/3rfold
cd ~/projects/3rfold

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Clone and install
git clone "https://github.com/chaimaetoubali-lab/3R-Fold" .
pip install -r requirements.txt
```

## Google Colab Setup

### 1. Mount Google Drive
### 2. Setup Environment
```python
# Install dependencies
!pip install -r requirements.txt

# Clone repository (if not already done)
!git clone "https://github.com/chaimaetoubali-lab/3R-Fold"
%cd 3R-Fold
```

### 3. Run in Colab
Run the notebook


## Advanced Configuration

### GPU Memory Management
```python
# For large datasets, reduce batch size
embedder = PretrainedRNAEmbedder(
    device='cuda',
    max_length=512  # Reduce from default 1022
)

# Use gradient checkpointing
torch.cuda.empty_cache()
```

### Parallel Processing
```python
# Multi-GPU setup (if available)
if torch.cuda.device_count() > 1:
    device = torch.device('cuda:0')
    # Implement DataParallel for training
```

### Custom Paths
```python
# For different directory structures
from config import set_base_path

# Temporary override
set_base_path('/path/to/experiment/data')

# Check current paths
print(f"Base: {get_base_path()}")
print(f"Results: {get_results_path()}")
print(f"Plots: {get_plots_path()}")
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Check Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -c "import sys; print(sys.path)"
```

#### 2. CUDA Out of Memory
```python
# Solutions:
# - Reduce batch size
# - Use CPU fallback
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# - Clear cache
torch.cuda.empty_cache()
```

#### 3. FAISS Installation Issues
```bash
# CPU version (safer)
pip install faiss-cpu

# GPU version (if CUDA available)
pip install faiss-gpu

# Verify installation
python -c "import faiss; print(faiss.__version__)"
```

### Performance Optimization

#### Memory Usage
```python
# Monitor memory
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Process in chunks
def process_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]
```

##  Expected Results

### Output Files
- **Results**: `results/*.csv` - Benchmark metrics and comparisons
- **Plots**: `results/plots/*.png` - Visualization of results
- **Models**: `models/*.pt` - Trained model checkpoints
- **Embeddings**: `results/*.pt` - Computed embeddings

### Benchmark Metrics
- **F1 Score**: Secondary structure prediction accuracy
- **Sensitivity**: True positive rate
- **PPV**: Positive predictive value
- **Runtime**: Computational efficiency
