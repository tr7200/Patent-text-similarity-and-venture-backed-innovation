## Hyperas

Hyperparameter tuning scripts for both the patent count and citation count models.

Tuning and training scripts use the same data because the logarithmic nature of the patent and citations variables make stratified sampling difficult, even at 4.5m observations.

These tuning scripts just served as a starting point since the parameters chosen by these scripts led to erratic loss during training.

To view notebooks, load them in the Jupyter nbviewer by clicking these links:

- [patent counts](https://nbviewer.jupyter.org/github/tr7200/Patent-text-analytics-and-Venture-backed-Innovation/blob/master/Hyperas/Hyperas_tuning_for_patent_count-1-27-20.ipynb)
- [citation counts](https://nbviewer.jupyter.org/github/tr7200/Patent-text-analytics-and-Venture-backed-Innovation/blob/master/Hyperas/Hyperas_tuning_for_citation_count-11-30-19.ipynb)
