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
    updated_df = df[['Hectare', 'coord', 'Unique Squirrel ID']]
    updated_df = updated_df.dissolve(by='Hectare', aggfunc='count')

    fig, ax = plt.subplots(1, figsize=(15, 7))

    updated_shape = shape_file.to_crs(epsg=3857)
    shape_file = updated_shape.plot(ax=ax, alpha=0, color='#FFFFFF')
    cx.add_basemap(shape_file, alpha=0.5)
    # cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels)
    # ax.set_axis_off()

    new = updated_df.to_crs(epsg=3857)
    # df = data_new.plot(ax=ax, column='Unique Squirrel ID', marker='.',
    # markersize=4, cmap='Spectral', legend=True)
    new = new.plot(ax=ax, column='Unique Squirrel ID', marker='.',
                             markersize=4, legend=True)
    plt.title('Squirrel Population in Central Park')
    plt.savefig('map.png')


# research question 2
"""
Takes in a pandas DataFrame and creates a bar chart of the
most common fur colors. Returns None.
"""
def common_fur_colors(df: pd.DataFrame) -> None:
    fur_color = df['Primary Fur Color'].value_counts().rename_axis('Primary Fur Color').reset_index(name='counts')

    sns.catplot(data = fur_color, x = 'Primary Fur Color', y = 'counts', kind = 'bar')
    plt.ylabel("Count")
    plt.title("Prevalence of Fur Color")
    plt.savefig("fur_color_plot.png", bbox_inches="tight")  


"""
Takes in a pandas DataFrame and creates a bar chart of the
most common highlight colors. Returns None.
"""
def common_highlight_colors(df: pd.DataFrame) -> None:
    highlight_color = df['Highlight Fur Color'].value_counts().rename_axis('Highlight Fur Color').reset_index(name='counts')
    sns.catplot(data = highlight_color, x = 'Highlight Fur Color', y = 'counts', kind = 'bar')
    plt.ylabel("Count")
    plt.xticks(rotation='vertical')
    plt.title("Prevalence of Fur Highlight Color")
    plt.savefig("highlight_color_plot.png", bbox_inches="tight")


"""
Takes in a pandas DataFrame and creates a bar chart of the
most common behaviors. Returns None.
"""
def common_behaviors(df: pd.DataFrame) -> None:
    approach = df['Approaches'].value_counts().rename_axis('Approaches').reset_index(name='counts')
    indifferent = df['Indifferent'].value_counts().rename_axis('Indifferent').reset_index(name='counts')
    runs_from = df['Runs from'].value_counts().rename_axis('Runs From').reset_index(name='counts')


    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3)

    approach.plot(ax =ax1, x='Approaches', kind = 'bar', stacked=True, figsize=(10,7), legend = False)
    indifferent.plot(ax =ax2, x='Indifferent', kind = 'bar', stacked=True, figsize=(10,7), legend = False, title='Behavior Types')
    runs_from.plot(ax =ax3, x='Runs From', kind = 'bar', stacked=True, figsize=(10,7), legend = False)

    plt.savefig("behavior_type_plot.png", bbox_inches="tight")


# research question 3
"""
Takes in a pandas DataFrame and trains DecisionTreeClassifier
to predict the type of behavior a squirrel will exhibit based
on its observed actions. Returns a list that contains the saved
model, the test size, and accuracy scores of the training and test data.
"""
def fit_and_predict_behavior(df: pd.DataFrame, test_size: float) -> list[Any]:
    # filtering data to squirrels that only exhibit one behavior
    app = df['Approaches'] == True
    indiff = df['Indifferent'] == True
    run = df['Runs from'] == True
    filtered = df[(app & ~indiff & ~run) | (indiff & ~app & ~run) | (run & ~app & ~indiff)]
    # set features to columns: 'Running', 'Chasing', 'Climbing', 'Eating', 'Foraging'
    # one-hot encode features, set features + labels
    # set training + test data, test size
    # fit model
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
    common_fur_colors(df)
    common_highlight_colors(df)
    common_behaviors(df)


if __name__ == '__main__':
    main()