"""
This program implements functions to read in and organize our data and drop
columns we don't need.
"""


import pandas as pd
import numpy as np


def clean_data(file_name: str) -> pd.DataFrame:
    """
    Reads in data from the csv file, converts columns 'Approaches',
    'Indifferent', and 'Runs from' to int, and returns a DataFrame with columns
    that we are planning to use.
    """
    df = pd.read_csv(file_name)
    filtered = df[['X', 'Y', 'Unique Squirrel ID', 'Hectare', 'Age',
                   'Primary Fur Color', 'Highlight Fur Color',
                   'Running', 'Chasing', 'Climbing', 'Eating',
                   'Foraging', 'Approaches', 'Indifferent', 'Runs from']]
    filtered = filtered.astype({'Approaches': 'int', 'Indifferent': 'int',
                                'Runs from': 'int'})
    return filtered


def filter_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a pandas DataFrame and returns a DataFrame that only includes
    squirrels who exhibited one behavior.
    """
    app = df['Approaches'] == 1
    ind = df['Indifferent'] == 1
    run = df['Runs from'] == 1
    f = df[(app & ~ind & ~run) | (ind & ~app & ~run) | (run & ~app & ~ind)]
    return f


def add_behavior_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a pandas DataFrame, creates a new column that represents
    the type of behavior exhibited by the squirrel, and returns the new
    DataFrame.
    """
    conditions = [(df['Approaches'] == 1),
                  (df['Indifferent'] == 1),
                  (df['Runs from'] == 1)
                  ]
    values = ['Approaches', 'Indifferent', 'Runs from']
    df = df.drop(columns=['Approaches', 'Indifferent', 'Runs from'])
    df['Behavior'] = np.select(conditions, values)
    return df


def drop_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a pandas DataFrame and returns a DataFrame that only includes
    squirrels who have a defined age.
    """
    df[['Age']].replace('', pd.NA)
    df[['Primary Fur Color']].replace('', pd.NA)
    filtered = df.dropna(subset=['Age', 'Primary Fur Color'])
    return filtered
