This repository contains the code for reproducing the results of the paper "Fairness-Accuracy Trade-Offs: A Causal Perspective".

Please check `R/zzz-deps.R` for the R-packages required to reproduce the results. Once these are installed, the experiments can be reproduced using the `experiments.R` script as follows:

1. Choose the dataset among options `"census"`, `"compas"`, and `"credit"`, corresponding to the experiments on Census 2018 and COMPAS data (in the main text), and the UCI credit experiment (in the appendix).
2. The script invokes the `accuracy_decomposition_boot()` functionality that performs bootstrap repetitions over the data. 
3. Internally, `accuracy_decomposition()`is called, which returns the object with all the required information.
4. Finally, after the computation, `vis_route()` functionality is used to obtain all the plots appearing in the manuscript.