"""
This program implements a function to read in our data and drop columns we don't need.
"""

import pandas as pd

"""
Reads in data from the csv file, returns a DataFrame with columns that we are planning to use.
"""
def clean_data() -> pd.DataFrame:
    squirrels = pd.read_csv("2018_Central_Park_Squirrel_Census_-_Squirrel_Data.csv")
    # only include:
    # 'X', 'Y', 'Hectare', 'Age', 'Primary Fur Color', 'Highlight Fur Color',
    # 'Running', 'Chasing', 'Climbing', 'Eating', 'Foraging', 'Approaches',
    # 'Indifferent', 'Runs from', 'Lat/Long'
    filtered = squirrels[['X', 'Y', 'Hectare', 'Age', 'Primary Fur Color', 'Highlight Fur Color',
                        'Running', 'Chasing', 'Climbing', 'Eating', 'Foraging', 'Approaches',
                        'Indifferent', 'Runs from', 'Lat/Long']]
    return filtered