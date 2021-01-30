## Hyperas

Hyperparameter tuning scripts for both the patent count and citation count models.

Tuning and training scripts use the same data because the logarithmic nature of the 
patent and citations variables make stratified sampling difficult, even at 4.5m 
observations.

These tuning scripts just served as a starting point since the parameters chosen 
by these scripts led to erratic loss during training.
