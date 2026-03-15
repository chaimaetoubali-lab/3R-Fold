# 3R-Fold: RNA Secondary Structure Prediction with Retrieval

A comprehensive framework for RNA secondary structure prediction using retrieval-guided folding with Deep Self Representation (DSR) and FAISS-based similarity search.

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Project](#running-the-project)
6. [Google Colab Setup](#google-colab-setup)
7. [Local Machine Setup](#local-machine-setup)
8. [Jupyter Notebook Usage](#jupyter-notebook-usage)
9. [Troubleshooting](#troubleshooting)

## 🎯 Overview

3R-Fold implements a novel approach to RNA secondary structure prediction that uses self representation to find top-k similarities. This code is used to perform benchmarking with the following methods:
- **RNA-FM**: Pre-trained RNA language model for embeddings
- **FAISS**: Efficient similarity search for retrieval
- **DSR**: Deep Self Representation for improved retrieval
- **Retrieval-Guided Folding**: Using neighbor structures to improve predictions

## 💻 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- CUDA-compatible GPU (recommended for training)

### Recommended Requirements
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 50GB+ free disk space

## 🚀 Installation

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

## ⚙️ Configuration

### 1. Global Path Configuration

The project uses a centralized path configuration system. Edit `config/paths.py`:

```python
# Set your base path
BASE_PATH = "/path/to/your/project"  # Full path to your python code
# BASE_PATH = "./"  # Relative project path
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

## 🏃‍♂️ Running the Project

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook 3RFold_NB.ipynb
```

### Option 2: Command Line
```bash
# Run main pipeline
python run_3RFold.py

# Run individual components
python experiments/run_benchmark.py
python experiments/run_embedding_umap.py
```

### Option 3: Individual Scripts
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

## 📊 Google Colab Setup

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Setup Environment
```python
# Install dependencies
!pip install torch torchvision torchaudio
!pip install fair-esm biotite
!pip install faiss-cpu
!pip install umap-learn seaborn tqdm pandas matplotlib

# Clone repository (if not already done)
!git clone <repository-url>
%cd 3R-Fold
```

### 3. Configure Paths
```python
# The config should already be set for Colab
# Verify in config/paths.py:
# BASE_PATH = "/content/drive/MyDrive/Foldv"
```

### 4. Run in Colab
```python
# Run the notebook
%run 3RFold_NB.ipynb

# Or execute individual cells
import sys
sys.path.append('.')
from run_3RFold import main
main()
```

## 🖥️ Local Machine Setup

### 1. Environment Setup
```bash
# Create project directory
mkdir -p ~/projects/3rfold
cd ~/projects/3rfold

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Clone and install
git clone <repository-url> .
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Create necessary directories
mkdir -p data results results/plots models

# Download Rfam database (automatic)
python -c "from data.build_dataset import download_rfam; download_rfam()"

# Or manually download
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz
gunzip Rfam.seed.gz
```

### 3. GPU Setup (Optional but Recommended)
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

## 📓 Jupyter Notebook Usage

### Opening the Notebook
```bash
# From project root
jupyter notebook 3RFold_NB.ipynb

# Or with specific IP/port
jupyter notebook --ip=0.0.0.0 --port=8888 3RFold_NB.ipynb
```

### Notebook Structure
The notebook `3RFold_NB.ipynb` contains:

1. **Setup Cell**: Imports and configuration
2. **Data Loading**: Dataset preparation and embedding
3. **Model Training**: DSR training (optional)
4. **Retrieval**: FAISS index building and search
5. **Evaluation**: Benchmark and metrics
6. **Visualization**: Results plotting and analysis

### Running Specific Cells

#### Cell-by-Cell Execution:
```python
# Cell 1: Setup
import sys
import os
sys.path.append('.')
from config import get_base_path, get_results_path, get_plots_path

# Cell 2: Data Loading
from data.build_dataset import build_dataset
dataset = build_dataset()
print(f"Dataset size: {len(dataset)}")

# Cell 3: Embeddings
from models.pretrained_embedder import PretrainedRNAEmbedder
from retrieval.build_kb import build_kb

embedder = PretrainedRNAEmbedder(device='cuda' if torch.cuda.is_available() else 'cpu')
embeddings = build_kb(dataset, embedder)
```

#### Full Execution:
```python
# Run all cells sequentially
# Kernel → Restart & Run All

# Or use command line
jupyter nbconvert --to python 3RFold_NB.ipynb
python 3RFold_NB.py
```

### Custom Configuration in Notebook
```python
# Override default config
import yaml

custom_config = {
    'retrieval': {'top_k': 10},
    'training': {'epochs': 100},
    'folding': {'bonus': 2.0}
}

# Save custom config
with open('configs/custom.yaml', 'w') as f:
    yaml.dump(custom_config, f)

# Use custom config
main('configs/custom.yaml')
```

## 🔧 Advanced Configuration

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

## 🐛 Troubleshooting

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

#### 4. Dataset Download Issues
```python
# Manual download alternative
import requests
from io import BytesIO
import gzip

url = "ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz"
response = requests.get(url.replace('ftp://', 'https://'))
with open('Rfam.seed.gz', 'wb') as f:
    f.write(response.content)
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

#### Speed Optimization
```python
# Use multiprocessing for embedding
from multiprocessing import Pool
with Pool(processes=4) as p:
    results = p.map(embed_sequences, sequence_chunks)
```

## 📊 Expected Results

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

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

[Add your license information here]

## 📞 Support

For issues and questions:
- Create GitHub issue
- Check troubleshooting section
- Review configuration examples

---

**Quick Start Commands:**

```bash
# Google Colab
1. Open Colab
2. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
3. Run: `%run 3RFold_NB.ipynb`

# Local Machine
1. Clone repo
2. Setup environment: `python -m venv venv && source venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Configure: Edit `config/paths.py`
5. Run: `jupyter notebook 3RFold_NB.ipynb`
```
