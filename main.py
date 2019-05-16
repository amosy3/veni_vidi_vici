import pandas as pd
from feature_extractor import extract , select_simple_features, combine_features
from models_trainer import train_models,compute_hodor_blending
from datetime import datetime
import os



features_dir = '../feature_extraction/features'
data_dir = '../raw_data'
id_column = "Id"

# read data - TODO add here robust Train/Test split

train = pd.read_parquet(os.path.join(data_dir,'train.parquet'), engine='pyarrow')
test = pd.read_parquet(os.path.join(data_dir,'test_kaggle.parquet'), engine='pyarrow')

train_ids = train[id_column]
test_ids = test[id_column]
#
#
#
# # extract and save all features
# df_train = (extract(train))
#
y_train = train['is_click'].values
#
#
# df_test = (extract(test))
# df_train = select_simple_features(df_train)
# df_test = select_simple_features(df_test)
#


df_train = combine_features(features_dir,'train',train_ids)
df_test = combine_features(features_dir,'test',test_ids)


print("Train shape {}".format(df_train.shape))
print("Test shape {}".format(df_test.shape))


df_train.set_index(id_column,inplace=True)
df_test.set_index(id_column,inplace=True)

train_index = df_train.index
test_index = df_test.index


X_train = df_train.values
X_test = df_test.values
#Test on simple features for now





# extract and save for val and test...
#df_train.to_csv('./features/all_features_' + str(datetime.now()))

# train models

models = compute_hodor_blending(X_train,y_train,X_test)

# prediction


# ensemble


# submission





