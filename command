# venv virtual environment commands

## activate
source venv/bin/activate

## deactivate
deactivate

# file conversion commands

## .ipynb → .Rmd → .py (venv environment)
source venv/bin/activate && python examples/ipynb2Rmd2py.py

## .Rmd → .ipynb (venv environment)
source venv/bin/activate && python examples/Rmd2ipynb.py
