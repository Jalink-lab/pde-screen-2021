# Dynamic FRET-FLIM based screens of signal transduction pathways: a feasibility study
by: Rolf Harkes, Olga Kukk, Sravasti Mukherjee, Jeffrey Klarenbeek, Bram van den Broek, Kees Jalink


This repository contains the code a accompanying [this paper](https://www.google.com) and has been used for all analysis and generation of the figures.

# Layout
* All raw data can be found in [this zenodo repository](https://zenodo.org/record/4746173).
* Running `analyse_data.py` on the raw data produces the results.
* `PDE_Analysis` contains the functions for analysis of the data.
* `PDE_Display` contains the functions for display of the results.
* The folder `Figures` contains the jupyter notebooks that use the result of `analyse_data.py` to generate the figures.

# How to setup the environment
We have used [PyCharm](https://www.jetbrains.com/pycharm/) and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to write and execute the code. 
To create an environment with all the packages required you can run the following command from a terminal `conda create -f environment.yml`.
After the environment is created you can activate it with `conda activate PDE_Screening`.
