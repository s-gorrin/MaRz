# data source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

import numpy as np
import pandas as pd


def get_heart_dataset(as_dataframe=False):
    """
    INTENT: prepare the heart disease dataset for modeling with

    PRE: heart_data.csv is a file containing heart disease information

    POST 1: non-numeric columns of the dataset are replaced with one-hot columns
    POST 2: the dataset is converted to a numpy array and returned
    """
    with open("heart_data.csv", 'r') as heart_csv:
        df = pd.read_csv(heart_csv)

    # save the targets to put back on the end after
    targets = df['HeartDisease']
    df = df.drop(['HeartDisease'], axis=1)

    # convert all non-numeric columns to one-hots
    df = pd.concat([df, pd.get_dummies(df['Sex'], 'sex')], axis=1).drop(['Sex'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['ChestPainType'], 'pain_type')], axis=1).drop(['ChestPainType'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['RestingECG'], 'ecg')], axis=1).drop(['RestingECG'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['ExerciseAngina'], 'angina')], axis=1).drop(['ExerciseAngina'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['ST_Slope'], 'slope')], axis=1).drop(['ST_Slope'], axis=1)
    df = pd.concat([df, targets], axis=1)

    pd.set_option('display.max_columns', None)
    print(df[:5].T)

    if as_dataframe:
        return df
    return np.array(df), list(df)
