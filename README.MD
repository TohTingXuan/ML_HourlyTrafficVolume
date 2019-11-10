# Hourly Traffic Volume Prediction

The machine learning pipeline design consists of 4 main stages, namely data preprocessing, learning, model evaluation and prediction. 


## Data Preprocessing

In the data preprocessing phase, data engineering and feature engineering were performed on the raw dataset. Columns with zero or little correlation with the target variable were dropped since these will not serve to improve the model performances. Binning was also performed. Notably, new features were engineered from the date_time column of the data as per the insights gained from the seasonality trend plots in the Exploratory Data Analysis. This gives a preprocessed dataset that can be utilised for the training of the model in the next phase


## Model Learning

Random forest classifier was selected as the learning algorithm for the prediction of the hourly traffic volume. This algorithm is pretty robust and works well with dataset having both categorical and numerical features. Model engineering was also conducted to find the optimal parameter so as to achieve better model performance.


## Evaluation and Prediction

The accuracy on the training and test sets are at 89.164% and 88.223% respectively. The variance is less than 1%, which is an indicator that the learned model is generalising well on the test set. From the evaluation metrics, it gives a clear direction that the next step is to further reduce the bias in the model, rather than the variance. To do so, we can explore different learning algorithms or having access to more data. Further optimisation of the current learning algorithm can also be carried out.


