# xARPES 

![xARPES](https://xarpes.github.io/_images/xarpes.svg)

Repository for the code xARPES &ndash; extraction from angle resolved photoemission spectra.

# Warning

This project is currently undergoing **beta testing**. Some functionalities from the accompanying preprint (https://arxiv.org/abs/2508.13845) are still missing. If you encounter any bugs, you can open an issue.

# Contributing

Contributions to the code are most welcome. xARPES is intended to co-develop alongside the increasing complexity of experimental ARPES data sets. Contributions can be made by forking the code and creating a pull request. Importing of file formats from different beamlines is particularly encouraged.

# Installing xARPES using a graphical package manager

Some users may prefer not to install xARPES via the command line.  
Most Python development environments include a graphical or integrated package manager that can install packages from either **PyPI** or **conda-forge**, depending on which environment you are using.

Below is a brief guide to the most common tools.

## If you are using Anaconda Navigator

Anaconda Navigator installs Python packages from **conda-forge** when the environment is configured to use that channel.

1. Open *Anaconda Navigator*  
2. Select your environment (or create a new one)
3. Switch the package channel to **conda-forge**
4. Search for `xarpes`
5. Click **Install**

This installs the latest stable version from conda-forge.

## If you are using PyCharm, VS Code, Spyder, or JupyterLab

These IDEs use the package source associated with your active environment:

- If the active environment is a **conda environment**, packages typically come from **conda-forge** (if the channel is enabled).
- If the active environment is a **venv** or **system Python**, packages come from **PyPI**.

### Installation steps (generic)

1. Open your IDE  
2. Select or create a Python environment  
3. Open the environment/package manager panel  
4. Search for `xarpes`  
5. Click **Install**

This uses:
- **PyPI** for standard Python environments
- **conda-forge** for conda environments

# Pip installation

The following steps describe how to install xARPES in a clean Python virtual environment using `pip`.  
These instructions apply to both user installation (from PyPI) and developer installation (from GitHub).

## Setting up the environment

It is highly recommended to set up a pristine Python virtual environment.  
First, the `venv` module may have to be installed:

    sudo apt install python3-venv

Create a virtual environment named `<my_venv>`:

    python3 -m venv <my_venv>

Activate it whenever installing or running xARPES:

    source <my_venv>/bin/activate

Upgrade pip:

    python3 -m pip install --upgrade pip

## Installing xARPES

### Option A — User installation (from PyPI)

This installs the latest stable release:

    python3 -m pip install xarpes

### Option B — Developer installation (editable install from GitHub)

This installs the development version and allows live editing:

    git clone git@github.com:xARPES/xARPES.git
    cd xARPES
    python3 -m pip install -e .

---

# Conda installation

The following steps describe how to install xARPES within a conda environment.  
These instructions apply to both user installation (from conda-forge) and developer installation (from GitHub).

## Setting up the environment

Download the installer appropriate for your system from:

    https://docs.anaconda.com/free/miniconda/

Example for Linux:

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

When prompted:

- Press Enter to scroll through the license.
- Answer `yes` to *accept license terms*.
- Choose your installation location.
- Optionally answer `yes` to:

        You can undo this by running `conda init --reverse $SHELL`? [yes|no]

If you said `yes`, the base environment is activated automatically in new shells.  
Otherwise, activate conda manually:

    eval "$(<your_path>/miniconda3/bin/conda shell.<your_shell> hook)"

For development workflows, install `conda-build` (optional but recommended):

    conda install conda-build

Create a fresh environment named `<my_env>`:

    conda create -n <my_env> -c defaults -c conda-forge
    conda activate <my_env>

## Installing xARPES

### Option A — User installation (from conda-forge)

This installs the latest stable release:

    conda install conda-forge::xarpes

### Option B — Developer installation (editable install from GitHub)

This installs the development version and allows live editing:

    git clone git@github.com:xARPES/xARPES.git
    cd xARPES
    pip install -e .

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
