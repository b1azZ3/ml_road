# Load libraries
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

# Libraries for Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

# Libraries for Statistical Models
import statsmodels.api as sm

# Libraries for Saving the Model
from pickle import dump
from pickle import load

# Time series Models
from statsmodels.tsa.arima.model import ARIMA

# Error Metrics
from sklearn.metrics import mean_squared_error

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

# Plotting
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf

# added by eh
from urllib.parse import urlencode
import yfinance as yf

# Disable the warnings
import warnings
warnings.filterwarnings('ignore')


# loading the data
def get_data(stk_tickers):
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

    stk_data = yf.download(stk_tickers, period='max')
    ccy_data = web.DataReader(ccy_tickers, 'fred')
    idx_data = web.DataReader(idx_tickers, 'fred')
    return stk_data, ccy_data, idx_data


# Create x and y variables
def create_x_and_y_variables(target_stk, stk_data, ccy_data, idx_data, return_period=5):
    Y = np.log(stk_data.loc[:, ('Adj Close', target_stk)]).diff(return_period).shift(-return_period)
    Y.name = Y.name[-1] + '_pred'

    X1 = np.log(stk_data.loc[:, ('Adj Close', (stk_tickers_[2], stk_tickers_[1]))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X4 = pd.concat([np.log(stk_data.loc[:, ('Adj Close', target_stk)]).diff(i) for i in
                    [return_period, return_period * 3, return_period * 6, return_period * 12]], axis=1).dropna()
    X4.columns = [target_stk + '_DT', target_stk + '_3DT', target_stk + '_6DT', target_stk + '_12DT']

    X = pd.concat([X1, X2, X3, X4], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]
    return X, Y, dataset


# exploratory analysis
def exploratory_analysis(ds):
    # pd.set_option('precision', 3)
    pd.set_option('display.precision', 3)
    ds.describe()
    ds.head()

    # histogram
    ds.hist(bins=50, sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12, 12))
    pyplot.show()

    # density distribution
    ds.plot(kind='density', subplots=True, layout=(4, 4), sharex=True, legend=True, fontsize=1, figsize=(15, 15))
    pyplot.show()

    # scatter plot and correlation matrix
    correlation = ds.corr()
    pyplot.figure(figsize=(15, 15))
    pyplot.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    scatter_matrix(ds, figsize=(12, 12))
    pyplot.show()

    # time series analysis
    res = sm.tsa.seasonal_decompose(y_, period=52)
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    pyplot.show()


# best_features_analysis
def best_features_analysis(x, y):
    # feature importance
    bestfeatures = SelectKBest(k=5, score_func=f_regression)
    fit = bestfeatures.fit(x, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)    # concat two dataframes for better visualization
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    featureScores.nlargest(10, 'Score').set_index('Specs')  # print 10 best features


# Train Test Split and Evaluation Metrics
def create_train_and_test_set(x, y):
    validation_size = 0.2
    train_size = int(len(x) * (1 - validation_size))
    x_train, x_test = x[0:train_size], x[train_size:len(x)]
    y_train, y_test = y[0:train_size], y[train_size:len(x)]
    return x_train, x_test, y_train, y_test


# create models
def create_models():
    # Regression and Tree Regression algorithms
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVR', SVR()))

    # Neural Network algorithms
    models.append(('MLP', MLPRegressor()))

    # Boosting methods
    models.append(('ABR', AdaBoostRegressor()))
    models.append(('GBR', GradientBoostingRegressor()))

    # Bagging methods
    models.append(('RFR', RandomForestRegressor()))
    models.append(('ETR', ExtraTreesRegressor()))
    return models


# run_all_models
def run_all_models(models, data, num_folds, scoring):
    names = []
    kfold_results = []
    test_results = []
    train_results = []
    x_train, x_test, y_train, y_test = data

    for name, model in models:
        names.append(name)

        # K Fold analysis:

        # kfold = KFold(n_splits=num_folds, random_state=seed)
        kfold = KFold(n_splits=num_folds)

        # converted mean square error to positive. The lower the better
        cv_results = -1 * cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        kfold_results.append(cv_results)

        # Full Training period
        res = model.fit(x_train, y_train)
        train_result = mean_squared_error(res.predict(x_train), y_train)
        train_results.append(train_result)

        # Test results
        test_result = mean_squared_error(res.predict(x_test), y_test)
        test_results.append(test_result)

        msg = "%s: %f (%f) %f %f" % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
        print(msg)
    return names, kfold_results, train_results, test_results


def make_box_plot(names, kfold_results):
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison: Kfold results')
    ax = fig.add_subplot(111)
    pyplot.boxplot(kfold_results)
    ax.set_xticklabels(names)
    fig.set_size_inches(15, 8)
    pyplot.show()


def compare_train_and_test_error(names, train_results, test_results):
    # compare algorithms
    fig = pyplot.figure()

    ind = np.arange(len(names))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.bar(ind - width / 2, train_results, width=width, label='Train Error')
    pyplot.bar(ind + width / 2, test_results, width=width, label='Test Error')
    fig.set_size_inches(15, 8)
    pyplot.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    pyplot.show()


if __name__ == '__main__':
    stk_tickers_ = ['MSFT', 'IBM', 'GOOGL']
    target_stock = stk_tickers_[0]
    stk_data_, ccy_data_, idx_data_ = get_data(stk_tickers_)
    x_, y_, ds_ = create_x_and_y_variables(target_stock, stk_data_, ccy_data_, idx_data_)
    exploratory_analysis(ds_)
    best_features_analysis(x_, y_)
    # x_train, x_test, y_train, y_test = create_train_and_test_set(x_, y_)
    data_ = create_train_and_test_set(x_, y_)
    models_ = create_models()
    scoring_ = 'neg_mean_squared_error'
    num_folds_ = 10
    # run_all_models(models_, data_, num_folds=10, scoring_)
    names_, kfold_results_, train_results_, test_results_ = run_all_models(models_, data_, num_folds_, scoring_)
    make_box_plot(names_, kfold_results_)
    compare_train_and_test_error(names_, train_results_, test_results_)





