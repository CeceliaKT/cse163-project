"""
This will implement our program
"""


# imports
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import Point
import contextily as cx

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from scipy import stats

from typing import Any

import processing

sns.set()

# define functions

# research question 1
def plot_squirrel_sightings(df: pd.DataFrame, shape_file) -> None:
    """
    Takes in pandas DataFrame and a shape file and returns
    a map of all squirrel sightings in Central Park.
    """
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
    # plt.savefig('map.png')
    


# research question 2
def common_fur_colors(df: pd.DataFrame) -> None:
    """
    Takes in a pandas DataFrame and creates a bar chart of the
    most common fur colors. Returns None.
    """
    fur_color = df['Primary Fur Color'].value_counts().rename_axis('Primary Fur Color').reset_index(name='counts')

    sns.catplot(data = fur_color, x = 'Primary Fur Color', y = 'counts', kind = 'bar')
    plt.ylabel("Count")
    plt.title("Prevalence of Fur Color")
    # plt.savefig("fur_color_plot.png", bbox_inches="tight")  


def common_highlight_colors(df: pd.DataFrame) -> None:
    """
    Takes in a pandas DataFrame and creates a bar chart of the
    most common highlight colors. Returns None.
    """
    highlight_color = df['Highlight Fur Color'].value_counts().rename_axis('Highlight Fur Color').reset_index(name='counts')
    sns.catplot(data = highlight_color, x = 'Highlight Fur Color', y = 'counts', kind = 'bar')
    plt.ylabel("Count")
    plt.xticks(rotation='vertical')
    plt.title("Prevalence of Fur Highlight Color")
    # plt.savefig("highlight_color_plot.png", bbox_inches="tight")


def common_behaviors(df: pd.DataFrame) -> None:
    """
    Takes in a pandas DataFrame and creates a bar chart of the
    most common behaviors. Returns None.
    """
    approach = df['Approaches'].value_counts().rename_axis('Approaches').reset_index(name='counts')
    indifferent = df['Indifferent'].value_counts().rename_axis('Indifferent').reset_index(name='counts')
    runs_from = df['Runs from'].value_counts().rename_axis('Runs From').reset_index(name='counts')


    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3)

    approach.plot(ax =ax1, x='Approaches', kind = 'bar', stacked=True, figsize=(10,7), legend = False)
    indifferent.plot(ax =ax2, x='Indifferent', kind = 'bar', stacked=True, figsize=(10,7), legend = False, title='Behavior Types')
    runs_from.plot(ax =ax3, x='Runs From', kind = 'bar', stacked=True, figsize=(10,7), legend = False)

    # plt.savefig("behavior_type_plot.png", bbox_inches="tight")


# research question 3
def add_behavior_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a pandas DataFrame, creates a new column that represents
    the type of behavior exhibited by the squirrel, and returns the new
    DataFrame.
    """
    conditions = [(df['Approaches'] == True),
                  (df['Indifferent'] == True),
                  (df['Runs from'] == True)
                  ]
    values = ['Approaches', 'Indifferent', 'Runs from']
    df = df.drop(columns=['Approaches', 'Indifferent', 'Runs from'])
    df['Behavior'] = np.select(conditions, values)
    return df


def fit_and_predict_behavior(df: pd.DataFrame):
    """
    Trains and tests a Random Forest Classifer with different feature combinations.
    Returns a tuple containing the resulting DataFrame and the trained model.
    """
    df = df.drop(columns=['X', 'Y', 'Lat/Long', 'Unique Squirrel ID', 'Hectare', 'Highlight Fur Color'])
    X = pd.get_dummies(df.drop(columns=['Behavior']))
    y = df['Behavior']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit a Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    return X_train, rfc


def plot_feature_importance(X_train , rfc) -> None:
    """
    Creates a bar chart of the feature importances of the Random Foreset
    Classifier.
    """
    importances = rfc.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(1, figsize=(20, 7))

    plt.barh(range(len(importances)), importances[indices])
    plt.xlabel("Feature Importance")
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(X_train.columns)[indices])
    plt.title('Feature Importance Scores')
    plt.savefig('feature_score.png')


# research question 4
def determine_validity(df: pd.DataFrame, expected: np.ndarray) -> float:
    """
    Takes in a pandas DataFrame and an array of predictions on the test
    data, returns a float that represents the p-value of the chi-square
    GOF test.
    """
    # to test result validity:
    # 1) filter df to observed behaviors
    # 2) create array to represent observed behaviors
    observed = df['Behavior'].tolist()
    # 3) calculate degrees of freedom (# of groups - 1) = 2
    # 4) scipy.stats.chisquare(f_obs: array_like, f_exp: array_like,
    #                          dof: int)
    p_value = stats.chisquare(observed, expected, 2)
    return p_value


def main():
    shape_data = 'CentralAndProspectParks//CentralPark.shp'

    df = processing.clean_data()
    filtered = processing.filter_behavior(df)
    # run methods here
    shape_file = gpd.read_file(shape_data)

    plot_squirrel_sightings(df, shape_file)
    common_fur_colors(df)
    common_highlight_colors(df)
    common_behaviors(df)

    full_df = add_behavior_column(filtered)
    X_train, rfc = fit_and_predict_behavior(full_df)
    plot_feature_importance(X_train, rfc)


if __name__ == '__main__':
    main()