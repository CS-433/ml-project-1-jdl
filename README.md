# CS-433 Machine Learning - Project 1

This Project aims to find a solution for the popular machine learning challenge of finding the Higgs Boson. This is a binary classification task. Therefore, a training set with 250000 data points and 30 features is provided as well as the test set (do not include the output variable). With several linear optimisation algorithms we tried to find accurate predictions. To go beyond those linear methods, polynomial feature expansion was implemented at the end.

## Exploratory data analysis
<ul>
    <li>shape</li>
    <li>type</li>
    <li>first and last values of rows/columns</li>
    <li>check for missing values</li>
    <li>deal with missing values (suppression-subsitution-interpolation_...)</li>
    <li>check for outliers (?)</li>
    <li>feature engineering (standardization etc)</li>
    <li>Pattern in data
        <ul>
            <li>univariate analysis (histogram). And we know that : 'Low variance features tend to contribute less to the prediction of outcome variable'.</li>
            <li>correlation analysis (between features) using heatmaps -> we should eliminate correlated features (one feature from a correlated pair is dropped)</li>
            <li>bivariate analysis. And we know that : ' If the relative variability is large, then it may be an indication of that these features could contribute to predicting the labels. '</li>
        </ul>
    </li>
    <li>feature selection : filter methods or wrapper methods?</li>
</ul>

## Code description
<ul>
    <li>project1.ipynb: main Jupyter Notebook. There, one can follow all the steps that were done.
    <li>proj1_helpers.py.: File with important helper functions.
    <li>implementations.py: File with the 6 linear optimization algorithms (least squares, GD, SGD, ridge regression, locistic regression and regularized logistic regression)
    <li>data_processing.py: File with the functions that were used to process the data sets
    <li>cross_validation.py: File with the functions that were used to perform the Cross Validations
   
        
</ul>
