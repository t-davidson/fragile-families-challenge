# ffchallenge

This repository contains code to reproduce the results of my final submissions for the [Fragile Families Challenge](http://www.fragilefamilieschallenge.org) as described in the [paper](http://www.thomasrdavidson.com) submitted to Socius.

The main code and results are all contained in the notebook `gpa.ipynb`, stored in the `model` directory. This notebook contains a lot of output that will require scrolling through in the Github version. To view it I suggest downloading or cloning this repository and [opening the file as a Jupyter notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html). This will require Python 3.5 and `jupyter` to be installed, as well as any other dependencies if you intend to run the notebook.

The `preprocess` directory contains the file used to clean, impute, and preprocess the raw data, although the final pre-proccessing tasks are done in the modeling notebook.

The `results` directory contains the CSVs necessary to reproduce the figures in the paper as well as the predicted values for GPA obtained from the final 5 models, as discussed in the paper.

***Note: This repository does not contain the Fragile Families data, which the challenge does not allow me to share with others and I will delete once the manuscript has been completed.***
