"""
This program implements functions to process, visualize, predict, and verify
outcomes about data from the 2018 New York City Squirrel Census dataset.
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
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

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
    ax.set_axis_off()

    new = updated_df.to_crs(epsg=3857)
    new = new.plot(ax=ax, column='Unique Squirrel ID', marker='.',
                   markersize=4, legend=True)
    plt.title('Squirrel Population in Central Park')
    plt.savefig('map.png')


# research question 2
def plot_common_fur_colors(df: pd.DataFrame) -> None:
    """
    Takes in a pandas DataFrame and creates a bar chart of the
    most common fur colors. Returns None.
    """
    fur_color = df['Primary Fur Color'].value_counts().rename_axis(
        'Primary Fur Color').reset_index(name='counts')

    fplot = sns.catplot(data=fur_color, x='Primary Fur Color', y='counts',
                        kind='bar')
    ax = fplot.facet_axis(0, 0)
    ax.bar_label(ax.containers[0])

    plt.ylabel('Count')
    plt.title('Prevalence of Fur Color')
    plt.savefig('fur_color_plot.png', bbox_inches='tight')


def plot_common_highlight_colors(df: pd.DataFrame) -> None:
    """
    Takes in a pandas DataFrame and creates a bar chart of the
    most common highlight colors. Returns None.
    """
    highlight_color = df['Highlight Fur Color'].value_counts().rename_axis(
        'Highlight Fur Color').reset_index(name='counts')
    hplot = sns.catplot(data=highlight_color, x='Highlight Fur Color',
                        y='counts', kind='bar')
    ax = hplot.facet_axis(0, 0)
    ax.bar_label(ax.containers[0])

    plt.ylabel('Count')
    plt.xticks(rotation='vertical')
    plt.title('Prevalence of Fur Highlight Color')
    plt.savefig('highlight_color_plot.png', bbox_inches='tight')


def plot_common_behaviors(df: pd.DataFrame) -> None:
    """
    Takes in a pandas DataFrame and creates a bar chart of the
    most common behaviors. Returns None.
    """
    approach = df['Approaches'].value_counts().rename_axis(
        'Approaches').reset_index(name='counts')
    indifferent = df['Indifferent'].value_counts().rename_axis(
        'Indifferent').reset_index(name='counts')
    runs_from = df['Runs from'].value_counts().rename_axis(
        'Runs From').reset_index(name='counts')

    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3)

    ax1 = approach.plot(ax=ax1, x='Approaches', kind='bar', stacked=True,
                        figsize=(10, 7), legend=False)
    ax2 = indifferent.plot(ax=ax2, x='Indifferent', kind='bar', stacked=True,
                           figsize=(10, 7), legend=False,
                           title='Behavior Types')
    ax3 = runs_from.plot(ax=ax3, x='Runs From', kind='bar', stacked=True,
                         figsize=(10, 7), legend=False)

    ax1.bar_label(ax1.containers[0], label_type='edge')
    ax2.bar_label(ax2.containers[0], label_type='edge')
    ax3.bar_label(ax3.containers[0], label_type='edge')

    plt.savefig('behavior_type_plot.png', bbox_inches='tight')


# research question 3
def fit_behavior(df: pd.DataFrame) -> list[Any]:
    """
    Takes in a pandas DataFrame and trains and tests a RandomForestClassifier.
    Returns a list containing the trained model, the DataFrame with columns of
    features used to train the model, and the two DataFrames with columns of
    features and labels that will be used to test the model.
    """
    # process data + split up into training and test datasets
    df = df.drop(columns=['X', 'Y', 'coord', 'Unique Squirrel ID', 'Hectare',
                          'Highlight Fur Color'])
    features = df.loc[:, df.columns != 'Behavior']
    features = pd.get_dummies(features)
    labels = df['Behavior']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.28,
                         random_state=24)

    # create a random forest classifier
    rf = RandomForestClassifier()

    # fit the model to the data
    rf.fit(features_train, labels_train)

    # accuracy
    training_pred1 = rf.predict(features_train)
    test_pred1 = rf.predict(features_test)
    print('Base RandomForestClassifier:')
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

    # find best RandomForestClassifier
    rf_base = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf_base,
                                   param_distributions=random_grid,
                                   n_iter=30, cv=5,
                                   verbose=2,
                                   random_state=42, n_jobs=4)
    rf_random.fit(features_train, labels_train)
    print('Best hyperparameters:', rf_random.best_params_)
    training_pred2 = rf_random.predict(features_train)
    test_pred2 = rf_random.predict(features_test)
    print('Adjusted RandomForestClassifier:')
    print('Training accuracy:', accuracy_score(labels_train, training_pred2))
    print('Test accuracy:', accuracy_score(labels_test, test_pred2))

    # fit best model
    hp = rf_random.best_params_
    new_rf = RandomForestClassifier(n_estimators=hp['n_estimators'],
                                    criterion=hp['criterion'],
                                    max_depth=hp['max_depth'],
                                    min_samples_split=hp['min_samples_split'],
                                    min_samples_leaf=hp['min_samples_leaf'],
                                    max_features=hp['max_features'],
                                    bootstrap=hp['bootstrap'])
    new_rf.fit(features_train, labels_train)
    return [new_rf, features_test, labels_test]


def plot_feature_importance(model: RandomForestClassifier,
                            features: list[Any]) -> None:
    """
    Takes in a RandomForestClassifier and a list that represents the
    features of the model. Creates a bar chart of the feature importances
    of the model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(1, figsize=(20, 7))

    plt.barh(range(len(importances)), importances[indices])
    plt.xlabel("Feature Importance")
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(features.columns)[indices])
    plt.title('Feature Importance Scores')
    plt.savefig('feature_score.png')


# research question 4
def verify_results(model: RandomForestClassifier,
                   features_test: list[Any], labels_test: list[Any]) -> None:
    """
    Takes in a RandomForestClassifier and lists that represents the features
    and labels used to test the model. Calculates the precision, recall, and
    F1 score of the model and plots a confusion matrix comparing the actual
    and predicted squirrel behaviors.
    """
    y_true = labels_test
    y_pred = model.predict(features_test)

    # calc precision, recall, f1 score
    print('Precision:', precision_score(y_true=y_true, y_pred=y_pred,
                                        labels=['Approaches', 'Indifferent',
                                                'Runs from'],
                                        average='macro',
                                        zero_division=0))
    print('Recall:', recall_score(y_true=y_true, y_pred=y_pred,
                                  labels=['Approaches', 'Indifferent',
                                          'Runs from'],
                                  average='macro'))
    print('F:', f1_score(y_true=y_true, y_pred=y_pred,
                         labels=['Approaches', 'Indifferent',
                                 'Runs from'],
                         average='macro'))

    # plot confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred,
                          labels=['Approaches', 'Indifferent',
                                  'Runs from'])
    print('Confusion matrix:')
    print(cm)
    cm_df = pd.DataFrame(cm, index=['Approaches', 'Indifferent', 'Runs from'],
                         columns=['Approaches', 'Indifferent', 'Runs from'])
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')


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
    plot_feature_importance(model_info[0], model_info[1])
    verify_results(model_info[0], model_info[1], model_info[2])


if __name__ == '__main__':
    main()
