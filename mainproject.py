"""
This will implement our program
"""


# imports
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from scipy import stats

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
Takes in a pandas DataFrame and trains different 
DecisionTreeClassifiers to predict the type of behavior a
squirrel will exhibit based on its observed actions. Returns 
a list of floats that represent different accuracy scores of
the model's predictions.
"""
def fit_and_predict_behavior(df: pd.DataFrame) -> list[float]:
    pass


"""
Takes in a pandas DataFrame and runs the DecisionTreeClassifier
with the highest accuracy score and plots the most important
features of that model. Returns None.
"""
def plot_feature_importance(df: pd.DataFrame) -> None:
   # run DecisionTreeClassifier w/ highest accuracy
   # feat_importances = pd.Series(model.feature_importances_,
   #                              index=X.columns)
   # feat_importances.nlargest(??).plot(kind='barh)
   pass


# research question 4
"""

"""
def determine_validity(df: pd.DataFrame) -> float:
    # scipy.stats.chisquare(f_obs: array_like, f_exp: array_like,
    #                       dof: int)
    # need array of observed behaviors, array of expected behaviors
    # from best model
    pass


def main():
    df = processing.clean_data()


if __name__ == '__main__':
    main()