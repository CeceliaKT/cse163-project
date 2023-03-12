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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score

from scipy.stats import chisquare

from typing import Any

import processing

sns.set()

# research question 1
def plot_squirrel_sightings(df: pd.DataFrame, shape_file: Any) -> None:
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
def plot_common_fur_colors(df: pd.DataFrame) -> None:
    """
    Takes in a pandas DataFrame and creates a bar chart of the
    most common fur colors. Returns None.
    """
    fur_color = df['Primary Fur Color'].value_counts().rename_axis('Primary Fur Color').reset_index(name='counts')

    sns.catplot(data = fur_color, x = 'Primary Fur Color', y = 'counts', kind = 'bar')
    plt.ylabel("Count")
    plt.title("Prevalence of Fur Color")
    # plt.savefig("fur_color_plot.png", bbox_inches="tight")  


def plot_common_highlight_colors(df: pd.DataFrame) -> None:
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


def plot_common_behaviors(df: pd.DataFrame) -> None:
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
def fit_behavior(df: pd.DataFrame) -> list[Any]:
    """
    Trains and tests a RandomForestClassifier. Returns a list containing the
    trained model, the DataFrame with columns of features used to train the model,
    and the two DataFrames with columns of features and labels that will be used
    to test the model.
    """
    # process data + split up into training and test datasets
    df = df.drop(columns=['X', 'Y', 'coord', 'Unique Squirrel ID', 'Hectare',
                          'Highlight Fur Color'])
    features = df.loc[:, df.columns != 'Behavior']
    features = pd.get_dummies(features)
    labels = df['Behavior']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.28,
                         random_state = 42)
    
    # create a random forest classifier
    rf = RandomForestClassifier()

    # fit the model to the data
    rf.fit(features_train, labels_train)

    # accuracy
    training_pred1 = rf.predict(features_train)
    test_pred1 = rf.predict(features_test)
    print('Training accuracy:', accuracy_score(labels_train, training_pred1))
    print('Test accuracy:', accuracy_score(labels_test, test_pred1))

    # number of trees in random forest
    n_estimators = np.linspace(100, 3000, int((3000 - 100) / 200) + 1,
                               dtype=int)
    # number of features to consider at every split
    max_features = ['sqrt']
    # maximum number of levels in tree
    max_depth = [1, 5, 10, 20, 50, 75, 100, 150, 200]
    # minimum number of samples required to split a node
    min_samples_split = [1, 2, 5, 10, 15, 20, 30]
    # minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4]
    # method of selecting samples for training each tree
    bootstrap = [True, False]
    # criterion
    criterion = ['gini', 'entropy']

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion': criterion}
    
    rf_base = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf_base,
                                   param_distributions = random_grid,
                                   n_iter = 30, cv = 5,
                                   verbose = 2,
                                   random_state = 42, n_jobs = 4)
    rf_random.fit(features_train, labels_train)
    print('Best hyperparameters:', rf_random.best_params_)
    training_pred2 = rf_random.predict(features_train)
    test_pred2 = rf_random.predict(features_test)
    print('Training accuracy:', accuracy_score(labels_train, training_pred2))
    print('Test accuracy:', accuracy_score(labels_test, test_pred2))
    
    return [rf, features_train, features_test, labels_test]


def plot_feature_importance(model_info: list[Any]) -> None:
    """
    Creates a bar chart of the feature importances of the Random Forest
    Classifier.
    """
    model = model_info[0]
    features_train = model_info[1]
    importances = model.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(1, figsize=(20, 7))

    plt.barh(range(len(importances)), importances[indices])
    plt.xlabel("Feature Importance")
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(features_train.columns)[indices])
    plt.title('Feature Importance Scores')
    # plt.savefig('feature_score.png')


# research question 4
def determine_validity(model_info: list[Any], df: pd.DataFrame) -> float:
    """
    Takes in a list of information about the model and a pandas DataFrame,
    returns a float that represents the p-value of a chi-square GOF test.
    Null hypothesis is that the model predictions are the same as the
    behaviors listed in the DataFrame.
    """
    model = model_info[0]
    df = df.drop(columns=['X', 'Y', 'coord', 'Unique Squirrel ID', 'Hectare',
                          'Highlight Fur Color'])
    features = df.loc[:, df.columns != 'Behavior']
    features = pd.get_dummies(features)
    predictions = model.predict(features)
    # 1) filter df to observed behaviors
    # 2) create array to represent observed behaviors
    data_obs = df['Behavior'].tolist()
    expected = [0, 0, 0]
    observed = [0, 0, 0]
    for behavior in predictions:
        if behavior == 'Approaches':
            expected[0] += 1
        elif behavior == 'Indifferent':
            expected[1] += 1
        else:
            expected[2] += 1
    for behavior in data_obs:
        if behavior == 'Approaches':
            observed[0] += 1
        elif behavior == 'Indifferent':
            observed[1] += 1
        else:
            observed[2] += 1
    chi_square_test_statistic, p_value = chisquare(observed, expected)
    return p_value


def main():
    shape_data = 'CentralAndProspectParks//CentralPark.shp'
    shape_file = gpd.read_file(shape_data)
    df = processing.clean_data()
    
    # run methods here
    plot_squirrel_sightings(df, shape_file)
    plot_common_fur_colors(df)
    plot_common_highlight_colors(df)
    plot_common_behaviors(df)

    filtered = processing.filter_behavior(df)
    added_column = processing.add_behavior_column(filtered)
    full_df = processing.drop_null(added_column)
    model_info = fit_behavior(full_df)
    plot_feature_importance(model_info)
    p_value = determine_validity(model_info, full_df)


if __name__ == '__main__':
    main()