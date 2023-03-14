"""
This program implements functions to read in and organize our data and drop
columns we don't need.
"""


import pandas as pd
import numpy as np


def clean_data() -> pd.DataFrame:
    """
    Reads in data from the csv file, returns a DataFrame with columns that
    we are planning to use.
    """
    squirrels = pd.read_csv(
        '2018_Central_Park_Squirrel_Census_-_Squirrel_Data.csv')
    filtered = squirrels[['X', 'Y', 'Unique Squirrel ID', 'Hectare', 'Age',
                          'Primary Fur Color', 'Highlight Fur Color',
                          'Running', 'Chasing', 'Climbing', 'Eating',
                          'Foraging', 'Approaches', 'Indifferent',
                          'Runs from']]
    return filtered


def filter_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a pandas DataFrame and returns a DataFrame that only includes
    squirrels who exhibited one behavior.
    """
    app = df['Approaches'] == True
    ind = df['Indifferent'] == True
    run = df['Runs from'] == True
    f = df[(app & ~ind & ~run) | (ind & ~app & ~run) | (run & ~app & ~ind)]
    return f


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


def drop_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a pandas DataFrame and returns a DataFrame that only includes
    squirrels who have a defined age.
    """
    df[['Age']].replace('', pd.NA)
    df[['Primary Fur Color']].replace('', pd.NA)
    filtered = df.dropna(subset=['Age', 'Primary Fur Color'])
    return filtered
