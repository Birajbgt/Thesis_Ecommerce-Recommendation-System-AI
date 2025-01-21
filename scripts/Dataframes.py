import pandas as pd
from pandas import DataFrame


class TrainingDataFrame():
    def __init__(self):
        self.interactions_df = pd.read_csv("./data/cleaned/interactions_cleaned.csv")
        self.user_df = pd.read_csv("./data/cleaned/users_cleaned.csv")
        self.prod_df = pd.read_csv("./data/cleaned/products_cleaned.csv")