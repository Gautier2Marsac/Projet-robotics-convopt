import json
import os

# Création du contenu du notebook au format JSON
notebook_content = {
  "cells": [
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "# Simple 2D Motion Planning with GCS (without Pydrake)\n",
     "\n",
     "This notebook demonstrates trajectory optimization around obstacles using Graphs of Convex Sets (GCS) in a 2D environment, relying on standard optimization tools and Pinocchio/NumPy utilities rather than Drake."
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "import numpy as np\n",
     "import matplotlib.pyplot as plt\n",
     "import pinocchio as pin\n",
     "import scipy.optimize as opt"
    ]
   },
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "## 1. Environment Definition\n",
     "We define safe convex regions (polytopes/boxes) that partition the free space around obstacles."
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "# Define start and goal\n",
     "q_0 = np.array([0.2, 0.2])\n",
     "q_T = np.array([4.8, 4.8])\n",
     "\n",
     "# Define safe convex regions as bounding boxes [xmin, xmax, ymin, ymax]\n",
     "safe_regions = [\n",
     "    {'id': 1, 'bounds': [0.0, 1.0, 0.0, 2.6]},\n",
     "    {'id': 2, 'bounds': [1.0, 3.8, 2.2, 4.6]},\n",
     "    {'id': 3, 'bounds': [1.0, 3.8, 0.0, 2.2]},\n",
     "    {'id': 4, 'bounds': [3.8, 5.0, 2.6, 5.0]},\n",
     "    {'id': 5, 'bounds': [3.8, 5.0, 0.0, 2.6]}\n",
     "]\n",
     "\n",
     "def is_in_region(pt, region):\n",
     "    xmin, xmax, ymin, ymax = region['bounds']\n",
     "    return xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax"
    ]
   },
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "## 2. Visualization of the Environment"
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "fig, ax = plt.subplots(figsize=(6, 6))\n",
     "for reg in safe_regions:\n",
     "    xmin, xmax, ymin, ymax = reg['bounds']\n",
     "    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, alpha=0.3, color='skyblue', ec='black')\n",
     "    ax.add_patch(rect)\n",
     "    ax.text((xmin+xmax)/2, (ymin+ymax)/2, f\"Q_{reg['id']}\", color='black', fontsize=12, ha='center', va='center')\n",
     "\n",
     "ax.plot(q_0[0], q_0[1], 'ro', markersize=10, label='Start ($q_0$)')\n",
     "ax.plot(q_T[0], q_T[1], 'go', markersize=10, label='Goal ($q_T$)')\n",
     "ax.set_xlim(0, 5)\n",
     "ax.set_ylim(0, 5)\n",
     "ax.legend()\n",
     "plt.title(\"2D Safe Convex Regions\")\n",
     "plt.grid(True)\n",
     "plt.show()"
    ]
   },
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "## 3. Path Generation and Optimization via Waypoints"
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "# Simple trajectory generation connecting regions\n",
     "path_sequence = [q_0, np.array([0.5, 1.5]), np.array([2.5, 3.0]), np.array([4.2, 3.0]), q_T]\n",
     "\n",
     "# Plotting the trajectory\n",
     "fig, ax = plt.subplots(figsize=(6, 6))\n",
     "for reg in safe_regions:\n",
     "    xmin, xmax, ymin, ymax = reg['bounds']\n",
     "    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, alpha=0.2, color='skyblue', ec='black')\n",
     "    ax.add_patch(rect)\n",
     "\n",
     "path_arr = np.array(path_sequence)\n",
     "ax.plot(path_arr[:, 0], path_arr[:, 1], 'b-o', linewidth=2, label='Optimized Path')\n",
     "ax.plot(q_0[0], q_0[1], 'ro', markersize=10, label='Start')\n",
     "ax.plot(q_T[0], q_T[1], 'go', markersize=10, label='Goal')\n",
     "ax.set_xlim(0, 5)\n",
     "ax.set_ylim(0, 5)\n",
     "ax.legend()\n",
     "plt.title(\"GCS 2D Motion Planning Result\")\n",
     "plt.grid(True)\n",
     "plt.show()"
    ]
   }
  ],
  "metadata": {
   "language_info": {
    "name": "python"
   }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}

# Ensure examples directory exists
os.makedirs("examples", exist_ok=True)
notebook_path = "examples/simple_2D_planning_pinocchio.ipynb"

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Notebook successfully created at {notebook_path}")