# **Tower Point Cloud Dataset**

This dataset contains multiple 3D point clouds of transmission towers. All files are stored in **`.txt`** format, and each file includes **XYZRGB** information for every point.

------

## **1. Data Format**

Each `.txt` file corresponds to one tower point cloud. The data is stored in **space-separated** format with the following fields:

```
X  Y  Z  R  G  B
```

- **X, Y, Z**: 3D coordinates of each point (in meters)
- **R, G, B**: Color values (0–255)

Example (partial):

```
12.320  5.812  37.552  124 130 121
12.325  5.820  37.540  125 129 120
...
```

------

## **2. Directory Structure**

```
dataset/
│── tower_001.txt
│── tower_002.txt
│── tower_003.txt
│── ...
└── README.md
```

------

## **3. Visualizing with PyVista**

You can use **PyVista** (a Python interface to VTK) to load and render the point cloud data.

### **3.1 Install Dependencies**

```
pip install pyvista numpy
```

### **3.2 Example: Loading and Visualizing the Point Cloud**

```
import numpy as np
import pyvista as pv

# Load the point cloud file
file_path = "tower_001.txt"
data = np.loadtxt(file_path)

# Extract xyz coordinates and rgb values
xyz = data[:, :3]
rgb = data[:, 3:6] / 255.0  # Normalize to 0–1

# Create a point cloud object
point_cloud = pv.PolyData(xyz)

# Assign colors
point_cloud["rgb"] = rgb

# Visualization
plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars="rgb", rgb=True, point_size=2)
plotter.show()
```

------

## **4. License**

This dataset is provided **for research purposes only** and commercial use is prohibited.
 For commercial licensing, please contact the dataset provider.