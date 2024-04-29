# Adaptive-stepsize-algorithms-for-Langevin-dynamics
Code to reproduce results from: 
A. Leroy, B. Leimkuhler, J. Latz and D. Higham, Adaptive stepsize algorithms for Langevin dynamics, 2024, https://arxiv.org/abs/2403.11993.

This folder contains Jupyter Notebook that allows the user to generate the results and/or plot the results from the paper. Each notebooks contain the instructions to reproduce the results.

Please note that the results from 2.MotivatingExample.ipynb and 6.Numericalexperiments.ipynb require extremely large simulations. In practice, those simulations have been run on the ECDF Linux Compute Cluster. However, the user can still reproduce results of lesser quality by decreasing the number of samples. The work used the ECDF Linux Compute Cluster, see Edinburgh Compute and Data Facility web site. 29 April 2024. U of Edinburgh <www.ecdf.ed.ac.uk>.

## 2. Motivating example
The notebook 2.MotivatingExample.ipynb is self-contained. The code is written in Python and parallelised using the Numba environment (the instructions on how to install the package are in the notebook). Running the notebook as it is, without modifying the code, will yield the plots using the data used for the paper. If you wish to generate your results, please comment out the path in the indicated cell (see notebook). 

## 5. Numerical Integrators and 6. Numerical Experiments
Those sections use code written in C++ to generate results. The notebooks 5.NumericalIntegrators.ipynb and 6.NumericalExperiments.ipynb contains Python code to plot the generated results. The notebooks also contain information on how to run the C++ code. If the user simply runs the notebook, the results plotted will use the data used for the paper. If the user wishes to use the data they generated, then they need to comment out the path where indicated in the Notebook. Please note that the parallel environment used in the C++ code is openmp, which needs to be installed, see <https://www.openmp.org/>.
