# Predictions

A jupyter notebook here shows how to predict on new data and includes 
a short data dictionary (see the [research paper](https://drive.google.com/open?id=1sqighkgCou1QalQ04polmRmMXUBmQOJN) 
for further details).

It can be viewed in Jupyter's NBViewer [here](https://nbviewer.jupyter.org/github/tr7200/Patent-text-similarity-and-venture-backed-innovation/blob/master/notebooks/predict/Predict_on_image.ipynb).

There is also a command line script here that may be used to create your 
own cosine similarity variable. The script takes:

- a CSV file with your patent's description as input
- the SDC Platinum category into which the patent 
- the number of patents to compare it with

and returns an appropriate cosine similarity measure.

The command line script uses the Loughran-McDonald stop word list for 
finance and accounting text data, which slightly modifies the default list 
from the python programming language's NLTK natural language. I combine 
it with Keith van Rijsbergen's stop word list. Additional patent-specific 
stop words were removed and that list is in the command line script.

### References:
[Tim Loughran and Bill McDonald, 2011, When is a Liability not a Liability?  Textual Analysis, Dictionaries, and 10-Ks, Journal of Finance, 66:1, 35-65.](http://ssrn.com/abstract=1331573)

[Keith van Rijsbergen, Information Retrieval](http://www.dcs.gla.ac.uk/Keith/Chapter.2/Table_2.1.html)
