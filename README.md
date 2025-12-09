# A3TP: Automated, Accurate, and Adaptive UAV Task Planning for Power Transmission Networks Inspection

## Abstract
UAV-assisted inspection is essential for maintaining large-scale power transmission networks. While recent work has explored automated trajectory generation, high-quality inspection requires more than safe paths. Expert-level inspection task planning must jointly optimize UAV trajectories and camera configurations, while adapting to diverse tower structures and established inspection standards to ensure comprehensive and reliable visual data.

We propose A3TP, an automated, accurate, and adaptive UAV task planning system tailored for large-world power transmission network inspection. A3TP integrates:

1. **Expertise-driven task modeling** to encode human planning knowledge from historical plans
2. **Geometry-centric planning** to generate reference trajectories and shooting parameters for identifying all inspection targets
3. **Deployment-adaptive trajectory optimization** to ensure safety and inspection coverage in complex environments

Operating in a knowledge-driven manner, A3TP achieves up to 55.78% IoU improvement over supervised baselines, over 96% classification accuracy, and trajectory planning results comparable to human experts. In a real-world deployment covering 373 towers across 65.84 km, A3TP reduced planning time by 95.3%, achieved an over 88% expert acceptance rate, and reliably detected field defects. These results demonstrate A3TP's scalability and practical value for maintaining real-world power transmission networks.

![System Overview](https://github.com/fqykinhgt/A3TP/blob/main/System%20overview.jpg)

**Code and dataset are available at:** https://github.com/fqykinhgt/A3TP.git

## Installation

### Prerequisites
- Conda package manager
- CUDA 12.1 compatible GPU (for PyTorch with CUDA support)

### Setup Environment

1. **Create and activate Conda environment:**

```bash
# Create a new conda environment with Python 3.8
conda create -n A3TP python=3.8

# Activate the environment
conda activate A3TP
```

2. **Install PyTorch with CUDA 12.1 support:**

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

3. **Install other dependencies:**

```bash
pip install numpy==1.24.0 scipy==1.10.0 scikit-learn==1.3.0
pip install pandas==2.0.2 matplotlib==3.7.0 seaborn==0.12.2
pip install pyvista==0.39.0 vtk==9.2.0 open3d==0.17.0
pip install laspy==2.5.0 pillow==9.5.0 imageio==2.31.1
pip install tqdm==4.65.0 pyproj==3.5.0 chardet==5.1.0 lxml==4.9.2
pip install tensorboard==2.13.0 ipywidgets==8.0.6
```

### CheckPoints
Download the [model](https://huggingface.co/datasets/qiyong2025/TransmissionTowerPointCloud/blob/main/pointnet.pth) trained using the PointNet network below, pay attention to modifying the model address of [function_pool.py](https://github.com/fqykinhgt/A3TP/blob/main/function_pool.py)

### DataSets
Please download the dataset from the Dataset

