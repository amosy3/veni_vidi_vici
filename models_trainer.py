import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ensemble_weapons.blend_proba import blend_proba


# define all desirable models
def create_models(X):
    models_dict = dict()

    for md in range(2,int(np.sqrt(X.shape[1])), 5):
        for criterion in ['gini', 'entropy']:
            models_dict['random_forest_'+str(md)+'_'+criterion] = \
                RandomForestClassifier(n_estimators=500, max_depth=md, criterion=criterion)

    return models_dict


def train_models(X, y):
    models_dict = create_models(X)
    for model_name, model in models_dict.items():
        print(model_name + ' is training...')
        model.fit(X,y)




def compute_hodor_blending(X_train,y,X_test):
    models_dict= create_models(X_train)
    for model_name , model in models_dict.items():
        #TODO - configure all args to that beast
        _ , _ , score = blend_proba(model,X_train,y,X_test,clf_name=model_name,return_score=True)
    pass