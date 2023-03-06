"""
This will implement our program
"""


# imports
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from scipy import stats

from typing import Any

import processing


# define functions

# research question 1
"""
Takes in pandas DataFrame ??and GeoDataFrame??, returns
a map of all squirrel sightings in Central Park.
"""
def plot_squirrel_sightings(df: pd.DataFrame) -> None:
    pass


# research question 2
"""
Takes in a pandas DataFrame and creates a bar chart of the
most common fur colors. Returns None.
"""
def common_fur_colors(df: pd.DataFrame) -> None:
    pass


"""
Takes in a pandas DataFrame and creates a bar chart of the
most common highlight colors. Returns None.
"""
def common_highlight_colors(df: pd.DataFrame) -> None:
    pass


"""
Takes in a pandas DataFrame and creates a bar chart of the
most common behaviors. Returns None.
"""
def common_behaviors(df: pd.DataFrame) -> None:
    pass


# research question 3
"""
Takes in a pandas DataFrame and trains DecisionTreeClassifier
to predict the type of behavior a squirrel will exhibit based
on its observed actions. Returns a list that contains the saved
model, the test size, and accuracy scores of the training and test data.
"""
def fit_and_predict_behavior(features: pd.DataFrame, labels: pd.Series,
                             test_size: float) -> list[Any]:
    # to save a model:
        # saved_model = pickle.dumps(model)
    pass


"""
Takes in a list of the saved model, test size, and accuracy scores of
the training and test data, runs the DecisionTreeClassifier with the
highest accuracy score and plots the most important features of that
model. Returns an array of predictions on the test data.
"""
def plot_feature_importance(info: list[Any]) -> np.ndarray:
   # to load saved model:
        # model_from_pickle = pickle.loads(saved_model)
   # to make predictions w/ saved model:
        # model_from_pickle.predict(insert data here)
    
   # run DecisionTreeClassifier w/ highest accuracy
   # feat_importances = pd.Series(model.feature_importances_,
   #                              index=X.columns)
   # feat_importances.nlargest(NUMBER OF FEATURES).plot(kind='barh)
   pass


# research question 4
"""
Takes in a pandas DataFrame and an array of predictions on the test
data, returns a float that represents the p-value of the chi-square
GOF test.
"""
def determine_validity(df: pd.DataFrame, expected: np.ndarray) -> float:
    # to test result validity:
        # 1) filter df to observed behaviors
        # 2) create array to represent observed behaviors, append
        #    behaviors
        # 3) calculate degrees of freedom (# of groups - 1)
        # 4) scipy.stats.chisquare(f_obs: array_like, f_exp: array_like,
        #                          dof: int)
    pass


def main():
    df = processing.clean_data()
    # run methods here

    # for ML model: should we drop rows where all behaviors are labeled
    # false??
    # identify features and labels of model, one-hot encode + run getdummies()
    # on features


if __name__ == '__main__':
    main()