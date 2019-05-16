import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ensemble_weapons.blend_proba import blend_proba
import os

from sklearn.metrics import log_loss, accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold

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



def train_models(df,y,models_dict,nfolds=2):
    X_train = df.values
    for model_name, model in models_dict.items():
        print(model_name + ' is training...')
        skf = StratifiedKFold(nfolds, shuffle=True, random_state=42)
        folds = skf.split(X_train, y)
        best_fold = -1
        best_auc = 0
        best_model = []
        for i, (train_index, test_index) in enumerate(folds):
            print("Train Fold {}/{}".format(i + 1, nfolds))
            fold_X_train = X_train[train_index]
            fold_y_train = y[train_index]
            fold_X_test = X_train[test_index]
            fold_y_test = y[test_index]

            model.fit(fold_X_train,fold_y_train)
            fold_preds = model.predict_proba(fold_X_test)
            fold_auc = roc_auc_score(fold_y_test, fold_preds[:, 1])
            print("AUC: {}".format(roc_auc_score(fold_y_test, fold_preds[:, 1])))
            print("Logistic loss: {}".format(log_loss(fold_y_test, fold_preds)))

            if fold_auc > best_auc:
                best_auc = fold_auc
                best_fold = i+1
                best_model = model

            print("best fold: {}, best auc: {}".format(best_fold,best_auc))
    return best_model


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