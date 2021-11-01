# CS-433 Machine Learning - Project 1

This Project aims to find a solution for the popular machine learning challenge of finding the Higgs Boson. This is a binary classification task. Therefore, a training set with 250000 data points and 30 features is provided as well as the test set (do not include the output variable). With several linear optimisation algorithms we tried to find accurate predictions. To go beyond those linear methods, polynomial feature expansion was implemented at the end.

## Exploratory data analysis
<ul>
    <li>Training data: 250000 datapoint x 30 features + 250000 Predictions</li>
    <li>Output variable: Binary Classification: 'b' and 's'</li>
    <li>Missing values are indicated by -999. 11 features contains missing values</li>
    <li>The Training data was divided into a training set and a validation set with a ratio of 0.7<\li>
    <li>Features with more than 50% missing values were deleted. The other missing values were replaced by the mean of the remaining values of the feature in the training set.</li>
    <li>The features were standardized using the mean and std of the training set</li>
    <li>Pattern in data:
        <ul>
            <li>univariate analysis (histogram and boxplots). To find features with uniform distributions that do not contribiute to the classification.</li>
            <li>correlation analysis (between features) using heatmaps ->To find correlated features (one feature from a correlated pair is dropped)</li
        </ul>
    </li>
    <li>feature selection : We dropped 13 features due to missing values, uniform distributions and correlation.</li>
</ul>

## Code description
<ul>
    <li>project1.ipynb: main Jupyter Notebook. There, one can follow all the steps that were done.</li>
    <li>proj1_helpers.py.: File with important helper functions.</li>
    <li>implementations.py: File with the 6 linear optimization algorithms (least squares, GD, SGD, ridge regression, locistic regression and regularized logistic regression)</li>
    <li>data_processing.py: File with the functions that were used to process the data sets</li>
    <li>cross_validation.py: File with the functions that were used to perform the Cross Validations</li>
    <li>run.py: File that allows to reproduce our final submission.</li>
   
        
</ul>
