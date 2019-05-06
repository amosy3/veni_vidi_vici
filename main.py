import pandas as pd
from feature_extractor import extract
from models_trainer import train_models
from datetime import datetime


# read data - TODO add here robust Train/Test split
df = pd.read_csv('./raw_data/fifa19/data.csv')

# extract and save all features
df = (extract(df))
    # extract and save for val and test...
df.to_csv('./features/all_features_' + str(datetime.now()))

# train models
    # read features and concat dataframes
models = train_models(df)

# prediction


# enssamble

# submission





