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
from shapely.geometry import Point
import contextily as cx

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from scipy import stats

from typing import Any

from processing import clean_data


SHAPE_DATA = 'CentralAndProspectParks\\CentralPark.shp'
# define functions

# research question 1
"""
Takes in pandas DataFrame ??and GeoDataFrame??, returns
a map of all squirrel sightings in Central Park.
"""
def plot_squirrel_sightings(df, shape_file) -> None:
    coordinates = zip(df['X'], df['Y'])
    df['coord'] = [Point(lon, lat) for lon, lat in coordinates]

    df = gpd.GeoDataFrame(df, geometry='coord')
    df.crs = shape_file.crs
    new = df[['Hectare', 'coord', 'Unique Squirrel ID']]
    new = new.dissolve(by='Hectare', aggfunc='count')

    fig, ax = plt.subplots(1, figsize=(15, 7))

    # data_test = data_test.plot(color='#EEEEEE')
    test_new = shape_file.to_crs(epsg=3857)
    shape_file = test_new.plot(ax=ax, alpha=0, color='#FFFFFF')
    cx.add_basemap(shape_file, alpha=0.5)
    # cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels)
    # ax.set_axis_off()

    data_new = new.to_crs(epsg=3857)
    # df = data_new.plot(ax=ax, column='Unique Squirrel ID', marker='.',
    # markersize=4, cmap='Spectral', legend=True)
    data_new = data_new.plot(ax=ax, column='Unique Squirrel ID', marker='.',
                             markersize=4, legend=True)
    plt.title('Squirrel Population in Central Park')
    plt.savefig('map.png')


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
    df = clean_data()
    # run methods here
    #shape_file = gpd.read_file(SHAPE_DATA)
    shape_file = gpd.read_file(SHAPE_DATA)

    plot_squirrel_sightings(df, shape_file)

    # for ML model: should we drop rows where all behaviors are labeled
    # false??
    # identify features and labels of model, one-hot encode + run getdummies()
    # on features


if __name__ == '__main__':
    main()