# Black Box Models and Sociological Explanations: Predicting High School GPA Using Neural Networks

## Introduction

This repository contains code to reproduce the results of my final submissions for the [Fragile Families Challenge](http://www.fragilefamilieschallenge.org) as described in the paper submitted to Socius. A pre-print is available [here](https://osf.io/preprints/socarxiv/7nsrf/).

***Note: This repository does not contain the Fragile Families data, which I am unable to share and I will delete once the manuscript has been completed.***

## What this repository contains

The `model` directory contains a Jupyter notebook with the code used to run the neural networks, `gpa.ipynb`. This notebook contains a lot of output that will require scrolling through in the Github version. To view it I suggest downloading or cloning this repository and [opening the file as a Jupyter notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html). This will require Python 3.5 and `jupyter` to be installed, as well as any other dependencies if you intend to run the notebook. The notebook `regression_baseline.ipynb` contains the code to implement the baseline OLS model.

The `preprocess` directory contains the Python script used to clean, impute, and preprocess the raw data, although note that the final pre-proccessing tasks are done in the modeling notebook.

The `LIME` directory contains the a notebook used to run the LIME algorithm (`LIME_explanations.ipynb`) and a notebook used to examine these explanations more closely (`examining_explanations.ipynb`). The explanations themselves, the output of the first notebook, are stored in `new_lime_explanations_dict.p`, a pickled Python dictionary. In addition, there is also a copy of the variable metadata CSV created by Connor Gilroy (see [this repository](https://github.com/fragilefamilieschallenge/variables-metadata)).

The `results` directory contains the CSVs necessary to reproduce the figures in the paper, as well as the predicted values for GPA obtained from the final 5 models and the baseline model, as discussed in the paper.

The `figures` directory contains the two figures used in the body of the papers along with the notebook used to create the activation functions visualization. The network diagram was created using [Draw.io](https://www.draw.io/), a free online tool to draw diagrams.

The `supplementary` directory contains notebooks, data, and figures discussed in the Supplementary Materials section of the paper.

## Questions?

If you have any questions please don't hesitate to contact me via e-mail (see my Github profile)  or to make an `Issue` on this repository.
