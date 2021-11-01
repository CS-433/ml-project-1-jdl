# CS-433 Machine Learning - Project 1

This Project aims to find a solution for the popular machine learning challenge of finding the Higgs Boson. This is a binary classification task. Therefore, a training set with 250000 data points and 30 features is provided as well as the test set (do not include the output variable). With several linear optimisation algorithms we tried to find accurate predictions. To go beyond those linear methods, polynomial feature expansion was implemented at the end.

## Code description
The code used in this project is divided in 5 python files and one jupyter notebook. 
Here is a description of the .py files :
<ul>
    <li>`proj1_helpers.py`: File with all the important help functions used in the other files;</li>
    <li>`implementations.py`: File with the 6 linear optimization algorithms (least squares, GD, SGD, ridge regression, logistic regression and regularized logistic regression);</li>
    <li>`data_processing.py`: File with the functions that were used to process the data sets;</li>
    <li>`cross_validation.py`: File with the functions that were used to perform the Cross Validations.</li>
    <li>`run.py`: File that allows to reproduce our final submission. Running this file writes in the output file 
the prediction of the test set, using ouf model, in the output file</li>        
</ul>

The Jupyter Notebook `project1.ipynb` reproduces the different steps that we did in order to build the model.
