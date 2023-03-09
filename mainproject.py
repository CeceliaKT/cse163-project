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
import itertools
from shapely.geometry import Point
import contextily as cx

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from scipy import stats

from typing import Any

import processing


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
    plt.savefig('map.png')
    


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
    plt.savefig("fur_color_plot.png", bbox_inches="tight")  


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
    plt.savefig("highlight_color_plot.png", bbox_inches="tight")


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

    plt.savefig("behavior_type_plot.png", bbox_inches="tight")


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
    df['Behavior'] = np.select(conditions, values)
    return df


def fit_and_predict_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains and tests Decision Tree models with different feature combinations.
    Returns a DataFrame with the feature combinations and their accuracy scores.
    """
    # Generate all possible combinations of features
    X = pd.get_dummies(df.drop(columns=['X', 'Y', 'Unique Squirrel ID','Behavior']))
    y = df['Behavior']
    feature_names = X.columns.tolist()

    results = []
    for n in range(3, len(feature_names) + 1):
        for combo in itertools.combinations(feature_names, n):
            X_subset = X[list(combo)]
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size = 0.2)
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((combo, accuracy))
    
    results_df = pd.DataFrame(results, columns=['Features', 'Accuracy'])
    return results_df


def plot_feature_importance(df: pd.DataFrame) -> None:
    """
    Creates a bar chart of the feature importance scores from a DataFrame
    produced by the fit_and_predict_behavior function
    """
    results_df = df.sort_values(by='Accuracy', ascending=False)
    results_df['Feature Names'] = df['Features'].apply(lambda x: ', '.join(x))
    fig, ax = plt.subplots(1, figsize=(15, 7))
    plt.barh(results_df['Feature Names'], results_df['Accuracy'])
    plt.xlabel('Accuracy Score')
    plt.ylabel("Feature Combinations")
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
    # run methods here
    shape_file = gpd.read_file(shape_data)

    plot_squirrel_sightings(df, shape_file)
    common_fur_colors(df)
    common_highlight_colors(df)
    common_behaviors(df)

    full_df = add_behavior_column(df)
    feature_df = fit_and_predict_behavior(full_df)
    plot_feature_importance(feature_df)

    # filtering data to squirrels that only exhibit one behavior for question 3
    filtered = processing.filter_behavior(df)
    # new dataframe w/ added column to represent behavior exhibited (use new column as label!)
    # result = add_behavior_column(filtered)
    # model_info = fit_and_predict_behavior(result, 0.2)


if __name__ == '__main__':
    main()