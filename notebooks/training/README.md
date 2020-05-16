## Training

The notebooks used for training the Tf-keras neural networks for both patents and citations are here, as well as the configuration files, plists, and training logs.

To view the notebooks in Jupyter's NBViewer:
- [patents](https://nbviewer.jupyter.org/github/tr7200/Patent-text-similarity-and-venture-backed-innovation/blob/master/notebooks/training/Patent_model_train-2-14-20.ipynb)
- [citations](https://nbviewer.jupyter.org/github/tr7200/Patent-text-similarity-and-venture-backed-innovation/blob/master/notebooks/training/Citation_model_train-2-17-20.ipynb)

The patent training script includes a custom metric measuring the loss as a function of the floor of the estimated dollar value of a patent, as defined 
[here](https://www.ipwatchdog.com/2017/07/12/patent-portfolio-valuations/id=85409/).
