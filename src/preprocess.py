import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    DATA_PATH = os.getenv("DATA_PATH_NIGERIA", default="dataset/nigeria_houses_data.csv")
    PROPERTY_TYPE_CONFIG = os.getenv("PROPERTY_TYPE_CONFIG", default="config/property_type.json")
    STATE_CONFIG = os.getenv("STATE_CONFIG", default="config/state.json")
    COLUMN_CONFIG = os.getenv("COLUMN_CONFIG", default="config/columns.json")
    TRAINING_COLUMNS_CONFIG = os.getenv("TRAINING_COLUMNS_CONFIG", default="config/training_columns.json")

    def __init__(self):
        # Load dataset
        self.data = pd.read_csv(self.DATA_PATH)
        self.columns = self.load_columns()
        self.training_columns = self.load_training_columns()
        self.property_type_mapping = self.load_property_type()
        self.state_mapping = self.load_state()
    
    @classmethod
    def load_columns(cls):
        with open(cls.COLUMN_CONFIG, "r") as f:
            return json.load(f)

    @classmethod
    def load_training_columns(cls):
        with open(cls.TRAINING_COLUMNS_CONFIG, "r") as f:
            return json.load(f)
    
    @classmethod
    def load_property_type(cls):
        with open(cls.PROPERTY_TYPE_CONFIG, "r") as f:
            return json.load(f)

    @classmethod
    def load_state(cls):
        with open(cls.STATE_CONFIG, "r") as f:
            return json.load(f)

    @staticmethod
    def column_encoder(df: pd.DataFrame, config: dict, column_name: str) -> pd.DataFrame:
        """
        Maps values in `column_name` based on `config` dictionary, encoding as per provided keys.
        """
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
        
        if not config:
            raise ValueError("The configuration dictionary cannot be empty.")
        
        df[column_name] = df[column_name].map(config).astype(int)
        
        return df
    
    @staticmethod
    def reorder_and_cast_columns(df: pd.DataFrame, reference_dict: dict) -> pd.DataFrame:
        """
        Reorder the columns of df to match reference_dict and cast to specified types.
        """
        column_order = list(reference_dict.keys())
        reordered_df = df.reindex(columns=column_order)

        for column, dtype in reference_dict.items():
            if column in reordered_df.columns:
                reordered_df[column] = reordered_df[column].astype(dtype)

        return reordered_df

    def preprocess(self, input_df: pd.DataFrame=None) -> pd.DataFrame:
        """
        Preprocess the data specific to Nigeria housing price prediction project.
        """
        df = input_df.copy() if input_df is not None else self.data.copy()

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Convert columns from float to int
        columns_to_convert = ['bedrooms', 'bathrooms', 'toilets', 'parking_space']
        df[columns_to_convert] = df[columns_to_convert].astype('int64')

        # Apply log transformation to the 'price' feature
        df['price'] = np.log1p(df['price'])

        # Encode property type and state
        df = self.column_encoder(df, self.property_type_mapping, "title")
        df.rename(columns={"title": "property_type"}, inplace=True)

        df = self.column_encoder(df, self.state_mapping, "state")
        df.rename(columns={"state": "state_encoded"}, inplace=True)

        # Price per bedroom feature
        df['price_per_bedroom'] = df['price'] / df['bedrooms']

        # One-Hot Encoding for 'town' column
        df = pd.get_dummies(df, columns=['town'], drop_first=True)

        # Dropping unnecessary columns
        columns_to_drop = ["property_type", "state_encoded"]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Reorder and cast columns
        df = self.reorder_and_cast_columns(df=df, reference_dict=self.training_columns)

        return df
