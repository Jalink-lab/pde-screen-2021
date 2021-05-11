# Dynamic FRET-FLIM based screens of signal transduction pathways: a feasibility study
by: Rolf Harkes, Olga Kukk, Sravasti Mukherjee, Jeffrey Klarenbeek, Bram van den Broek, Kees Jalink

This repository contains the code accompanying [this paper](https://www.google.com) and has been used for all analysis and generation of the figures. All raw data can be found in [this zenodo repository](https://zenodo.org/record/4746173).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4746173.svg)](https://doi.org/10.5281/zenodo.4746173)

# Layout
* Running `analyse_data.py` on the raw data produces the results.
* `PDE_Analysis` contains the functions for the analysis of the data.
* `PDE_Display` contains the functions for displaying the results.
* The folder `Figures` contains the jupyter notebooks that use the result of `analyse_data.py` to generate the figures.

# How to setup the environment
We have used [PyCharm](https://www.jetbrains.com/pycharm/) and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to write and execute the code. 
To create an environment with all the packages required you can run the following command from a terminal `conda create -f environment.yml`.
After the environment is created you can activate it with `conda activate PDE_Screening`.
For more information on managing environments, see the [conda documentation](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/getting-started.html#managing-envs).
