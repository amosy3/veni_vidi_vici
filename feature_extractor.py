import numpy as np
import pandas as pd
import os


class FeatureExtractionTemplate():

    def __init__(self, init_param):
        self.init_param = init_param

    def fit(self,X):
        return X

    def transform(self,X):
        return X

    def fit_transform(self,X):
        self._X_ = self.fit(X)
        return self.transform(self._X_)


class BasicStatisticFeatures():

    def __init__(self, col_names):
        self.col_names = col_names
        self.means = dict()
        self.vars = dict()
        self.modes = dict()

    def fit(self, df):
        for col in self.col_names:
            self.means[col] = df[col].mean()
            self.vars[col] = df[col].var()
            self.modes[col] = df[col].mode()
        return df

    def transform(self, df):
        for col in self.col_names:
            df["std_count_" + col] = (df[col]-self.means[col])/self.vars[col]
        return df

    def fit_transform(self, df):
        _X_ = self.fit(df)
        return self.transform(_X_)


def basic_time_features(df: pd.DataFrame):
    """
      The purpose of this method is to:
      1) Break 'page_view_start_time' into several features such as year,hour and so on.
      2) Add some new featurs

      :param df: input dataframe to add feature to.
      :return: df.
      """
    df['date'] = pd.to_datetime(df.page_view_start_time, unit='ms')
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek

    df['empiric_prb'] = (df['empiric_clicks']) / (df['empiric_recs'] + 1)
    df['user_prb'] = (df['user_clicks']) / (df['user_recs'] + 1)
    df['non_work_hours'] = df['hour'].apply(lambda x: 1 if (x < 8 or x > 17) else 0)
    df['os_family=2'] = df['os_family'].apply(lambda x: 1 if (x == 2) else 0)
    return df




def combine_features(features_dir,dataset_type,ids):
    assert dataset_type in ['train','test']
    all_features_df = pd.DataFrame(data = {"Id":ids})
    features_files = [x for x in os.listdir(features_dir) if x.startswith(dataset_type)]
    for file in features_files:
        print("Reading feature {}".format(file))
        try:
            df_path = os.path.join(features_dir,file)
            df = pd.read_pickle(df_path)
            all_features_df = pd.merge(all_features_df, df,on='Id')
        except:
            raise Exception("Feature {} is bad".format(file))

    return all_features_df


def select_simple_features(df):
    """
    Only for testing purposes
    :param df:
    :return: simple features
    """
    simple_features = ['hour','dayofweek','empiric_prb','user_prb','non_work_hours','os_family=2']
    return df[simple_features]



def quantile_normalize(preds_folders):
  """
  Compute quantile normalization on prediction folder, save it to <preds_folders>/quantile_normed
  :param preds_folders: folder that contains csv predictions
  :return: None

  """
  all_preds_df = pd.DataFrame(columns=["Id"])
  output_dir = os.path.join(preds_folders,'quantile_normed')
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

  pred_files = [x for x in os.listdir(preds_folders) if "csv" in x]
  for file in pred_files:
    df_path = os.path.join(preds_folders, file)
    df = pd.read_csv(df_path)
    all_preds_df["Id"] = df["Id"]
    all_preds_df["Predicted_{}".format(file)] = df["Predicted"]


  quantile_transformer = preprocessing.QuantileTransformer(random_state=0,n_quantiles=10000)
  preds_quantiles = quantile_transformer.fit_transform(all_preds_df.ix[:,1:].values)
  all_preds_df.iloc[:,1:] = preds_quantiles



  for file in pred_files:
    filename, _ = os.path.splitext(file)
    df_path = os.path.join(output_dir, filename)+".csv"
    out_df = pd.DataFrame()
    out_df['Id'] = all_preds_df['Id']
    out_df['Predicted'] = all_preds_df["Predicted_{}".format(file)]
    out_df.to_csv(df_path)



  print(1)




def extract(df):
    # create here all the features
    df = df.drop('campaign_language',axis=1)
    df = basic_time_features(df)
    return df


