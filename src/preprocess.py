import os
import json
import numpy as np
import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", default="data/housing-data.csv")

class Preprocessor:
    DATA_PATH = os.getenv("DATA_PATH", default="data/nigeria_houses_data.csv")
    STATE_CONFIG = os.getenv("STATE_CONFIG", default="config/state_target_enc.json")
    TITLE_CONFIG = os.getenv("TITLE_CONFIG", default="config/title.json")
    TOWN_CONFIG = os.getenv("TOWN_CONFIG", default="config/town_freq_enc.json")
    TRAINING_COLUMNS_CONFIG = os.getenv("TRAINING_COLUMNS_CONFIG", default="config/training_columns.json")

    def __init__(self):
        self.data = pd.read_csv(self.DATA_PATH)
        self.training_columns = self.load_training_columns()
        self.state = self.load_state()
        self.title = self.load_title()
        self.town = self.load_town()

    @classmethod
    def load_training_columns(cls):
        with open(cls.TRAINING_COLUMNS_CONFIG, "r") as f:
            return json.load(f)

    @classmethod
    def load_state(cls):
        with open(cls.STATE_CONFIG, "r") as f:
            return json.load(f)

    @classmethod
    def load_title(cls):
        with open(cls.TITLE_CONFIG, "r") as f:
            return json.load(f)

    @classmethod
    def load_town(cls):
        with open(cls.TOWN_CONFIG, "r") as f:
            return json.load(f)

    @staticmethod
    def column_encoder(df: pd.DataFrame, config: dict, column_name: str) -> pd.DataFrame:
        """
        One-hot encodes specified categorical column based on provided config.
        """
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
        
        # One-hot encode based on provided config
        for key in config.keys():
            df[config[key]] = False
        
        for index, row in df.iterrows():
            if row[column_name] in config:
                df.loc[index, config[row[column_name]]] = True

        df.drop(column_name, axis=1, inplace=True)
        
        return df

    @staticmethod
    def reorder_and_cast_columns(df: pd.DataFrame, reference_dict: dict) -> pd.DataFrame:
        """
        Reorders and casts columns based on the provided reference dictionary.
        """
        column_order = list(reference_dict.keys())
        reordered_df = df.reindex(columns=column_order)

        for column, dtype in reference_dict.items():
            if column in reordered_df.columns:
                reordered_df[column] = reordered_df[column].astype(dtype)

        return reordered_df

    def preprocess(self, input_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Preprocess the data by applying encoding, feature engineering, outlier handling, and type conversion.
        """
        df = input_df.copy() if input_df is not None else self.data.copy()

        # Step 1: Convert specified columns to integer
        integer_columns = ['bedrooms', 'bathrooms', 'toilets', 'parking_space']
        df[integer_columns] = df[integer_columns].astype(int)

        # Step 2: Remove duplicates
        df.drop_duplicates(inplace=True)

        # Step 3: Handle outliers
        df['price'] = np.log1p(df['price'])
        for col in ['toilets', 'parking_space']:
            upper_limit = df[col].quantile(0.99)
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

        # Step 4: Create new features
        df['price_per_bedroom'] = df['price'] / (df['bedrooms'] + 1)  # Avoid division by zero
        df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['toilets']
        df['room_to_parking'] = df['total_rooms'] / (df['parking_space'] + 1)  # Avoid division by zero

        # Step 5: One-hot encode 'title' and other categorical columns using config
        df = self.column_encoder(df, self.title, 'title')

        # Step 6: Target encode the 'state' feature based on the mean of 'price'
        df['state_encoded'] = df['state'].map(self.state)
        df.drop(columns=['state'], inplace=True)

        # Step 7: Frequency encode the 'town' feature
        df['town_encoded'] = df['town'].map(self.town)
        df.drop(columns=['town'], inplace=True)

        # Step 8: Reorder and cast columns based on training columns configuration
        df = self.reorder_and_cast_columns(df=df, reference_dict=self.training_columns)

        # Final clean-up: Drop any remaining NaNs
        df.dropna(inplace=True)

        return df
