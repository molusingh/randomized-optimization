# cs7641-randomized-optimization

# Conda Setup Instructions (need conda, can get miniconda off chocolatey for windows or homebrew on mac)
### Using conda to create python environment
conda env create -f environment.yml

### activate the environemnt
conda activate cs7641

### Install mlrose-hiive
pip install mlrose-hiive

### if needed, add debugger
jupyter labextension install @jupyterlab/debugger

### update environment after changes to environment.yml file (deactivate env first)
conda env update --file environment.yml --prune

### Open up jupyter lab to access notebook if desired
jupyter lab

# generate final results, outputs charts in ./output directory
python main.py 

References:

mlrose:
https://mlrose.readthedocs.io/

n-queens:
https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb

Dataset Sources
diabetes: https://www.kaggle.com/uciml/pima-indians-diabetes-database