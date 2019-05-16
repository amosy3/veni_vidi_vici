import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ensemble_weapons.blend_proba import blend_proba
import os


predictions_folder = '../models_predictions'


# define all desirable models
def create_models(X):
    models_dict = dict()
    md = 10
    #for md in range(2,int(np.sqrt(X.shape[1])), 5):
    for criterion in ['gini', 'entropy']:
        models_dict['random_forest_'+str(md)+'_'+criterion] = \
            RandomForestClassifier(n_estimators=500, max_depth=md, criterion=criterion,n_jobs=-1)


    return models_dict



def train_models(df,y,models_dict):
    X = df.values
    for model_name, model in models_dict.items():
        print(model_name + ' is training...')
        model.fit(X,y)



def predict_models(df,models_dict):
    prediction_df = pd.DataFrame()
    prediction_df = prediction_df.reindex_like(df)
    X = df.values
    prediction_df.to_csv()
    for model_name, model in models_dict.items():
        prediction_path = os.path.join(predictions_folder,model_name) + ".csv"
        print(model_name + ' is predicting... ')
        preds = model.predict_proba(X)
        prediction_df["Predicted"] = preds[:,1]
        print("Saving model to {}".format(prediction_path))
        prediction_df['Predicted'].to_csv(prediction_path,header=True)







def compute_hodor_blending(X_train,y,X_test):
    models_dict = create_models(X_train)
    for model_name , model in models_dict.items():
        #TODO - configure all args to that beast
        _ , _ , score = blend_proba(model,X_train,y,X_test,clf_name=model_name,return_score=True)
        print("model {} , validation loss {}".format(model_name,score))