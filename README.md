# xARPES 

![xARPES](https://xarpes.github.io/_images/xarpes.svg)

Repository for the code xARPES &ndash; extraction of the self-energy and Eliashberg function from angle-resolved photoemission spectroscopy. The documentation can be found at https://xarpes.readthedocs.io, while the code is maintained at https://github.com/xARPES/xARPES. Instructions for installing the code and downloading the code are found below. An extensive description of the functionalities and examples is found at https://arxiv.org/abs/2508.13845.

# Warning

This project is currently undergoing **beta testing**. Some functionalities from the accompanying preprint (https://arxiv.org/abs/2508.13845) are still missing. If you encounter any bugs, you can open an issue.

# Contributing

Contributions to the code are most welcome. xARPES is intended to co-develop alongside the increasing complexity of experimental ARPES data sets. Contributions can be made by forking the code and creating a pull request. Importing of file formats from different beamlines is particularly encouraged. The code is maintained on GitHub: https://github.com/xARPES/xARPES.

# Installation

xARPES can be installed in several ways depending on your workflow:  
- via a graphical package manager (Anaconda Navigator, VS Code, PyCharm, Spyder, JupyterLab)  
- via Conda Forge, out-of-the-box or editable installation, sourcing https://anaconda.org/conda-forge/xarpes.
- via Pip, out-of-the-box or editable installation, sourcing https://https://pypi.org/project/xarpes.

## Graphical package manager installation

Most IDEs and scientific Python distributions include a GUI-based package manager.  
These typically install from conda-forge (for conda environments) or PyPI (for venv/system Python).

### Anaconda Navigator

1. Open Anaconda Navigator  
2. Select or create an environment  
3. Set the package channel to conda-forge  
4. Search for “xarpes”  
5. Click Install

This installs the latest stable release from conda-forge.

### PyCharm, VS Code, Spyder, or JupyterLab

These IDEs install from the active environment’s package source:
- conda environment → installs from conda-forge  
- venv/system Python → installs from PyPI

### Installation steps (generic)

1. Open your IDE  
2. Select or create a Python environment  
3. Open the environment/package manager panel  
4. Search for “xarpes”  
5. Click Install

## Conda Forge installation

Install xARPES inside a conda environment, either out of the box or as an editable.

### Setting up a conda environment

Download and install Miniconda (see https://docs.anaconda.com/free/miniconda/).

Example for Linux:  
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
bash Miniconda3-latest-Linux-x86_64.sh

Create and activate an environment:  
conda create -n <my_env> -c defaults -c conda-forge  
conda activate <my_env>

### Installing xARPES

#### Option A — User installation (from conda-forge)

conda install conda-forge::xarpes

#### Option B — Developer installation (GitHub, editable)

git clone git@github.com:xARPES/xARPES.git  
cd xARPES  
pip install -e .

## Pip installation

Install xARPES using pip, either out of the box or as an editable.

### Setting up a virtual environment

Install venv if necessary:  
sudo apt install python3-venv

Create and activate a virtual environment:  
python3 -m venv <my_venv>  
source <my_venv>/bin/activate

Upgrade pip:  
python3 -m pip install --upgrade pip

### Installing xARPES

#### Option A — User installation (PyPI)

python3 -m pip install xarpes

#### Option B — Developer installation (GitHub, editable)

git clone git@github.com:xARPES/xARPES.git  
cd xARPES  
python3 -m pip install -e .

# Examples

After installation of xARPES, the `examples/` folder can be downloaded to the current directory:

	xarpes_download_examples

Equivalently:

	python3 -c "import xarpes; xarpes.download_examples()"

# Execution

It is recommended to use JupyterLab to analyse data. JupyterLab is launched using:

	jupyter lab

# Citation

If you have used xARPES for your research, please cite the following preprint: https://arxiv.org/abs/2508.13845.

# License

Copyright (C) 2025 xARPES Developers

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3, as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.