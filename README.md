# exubi
Private repository for the code Eliashberg eXtraction Using Bayesian Inference

## Installation

A virtual environment such as conda is recommanded.

    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh

Initialize and create the new environment:

    ~/miniconda3/bin/conda init bash
    conda create --name exubi
    sudo env "PATH=$PATH" conda update conda
    conda install -c conda-forge numpy
    conda install -c conda-forge matplotlib
    conda install -c conda-forge scipy
    pip install igor2

To install the lastest version of exubi:

    git clone git@github.com:TeetotalingTom/exubi.git
    cd exubi/
    sudo python3 -m pip install -e .

At present exubi requires JupyterLab:

    conda install -c conda-forge jupyterlab
    conda install -c conda-forge jupytext

## Execution

The Jupyter Lab can be launched using

	jupyter lab

