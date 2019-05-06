import pandas as pd
from feature_extractor import extract
from models_trainer import train_models
from datetime import datetime


# read data - TODO add here robust Train/Test split
df = pd.read_csv('./raw_data/fifa19/data.csv')

# extract all features
df = (extract(df))
df.to_csv('./features/all_features_' + str(datetime.now()))

# train models

models = train_models(df)

# prediction

# enssamble

# submission





