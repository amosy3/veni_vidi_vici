from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss, accuracy_score,roc_auc_score
import numpy as np
from hashlib import md5
import json

def blend_proba(clf, X_train, y, X_test, nfolds=5, save_preds="",
                save_test_only="", seed=300373, save_params="",
                clf_name="XX", generalizers_params=[], minimal_loss=0,
                return_score=False, minimizer="log_loss"):
  print("\nBlending with classifier:\n\t{}".format(clf))
  skf = StratifiedKFold(nfolds,shuffle=True,random_state=seed)
  folds = skf.split(X_train, y)
  dataset_blend_train = np.zeros((X_train.shape[0],np.unique(y).shape[0]))

  #iterate through train set and train - predict folds
  loss = 0
  for i, (train_index, test_index) in enumerate( folds ):
    print("Train Fold {}/{}".format(i+1,nfolds))
    fold_X_train = X_train[train_index]
    fold_y_train = y[train_index]
    fold_X_test = X_train[test_index]
    fold_y_test = y[test_index]
    clf.fit(fold_X_train, fold_y_train)

    fold_preds = clf.predict_proba(fold_X_test)
    print("Logistic loss: {}".format(log_loss(fold_y_test,fold_preds)))
    print("AUC: {}".format(roc_auc_score(fold_y_test,np.argmax(fold_preds,axis=1))))

    dataset_blend_train[test_index] = fold_preds
    if minimizer == "log_loss":
      loss += log_loss(fold_y_test,fold_preds)
    if minimizer == "accuracy":
      fold_preds_a = np.argmax(fold_preds, axis=1)
      loss += accuracy_score(fold_y_test,fold_preds_a)
    #fold_preds = clf.predict(fold_X_test)

    #loss += accuracy_score(fold_y_test,fold_preds)

    if minimal_loss > 0 and loss > minimal_loss and i == 0:
      return False, False
    fold_preds = np.argmax(fold_preds, axis=1)
    print("Accuracy:      {}".format(accuracy_score(fold_y_test,fold_preds)))
    print("AUC: {}".format(roc_auc_score(fold_y_test,fold_preds)))

  avg_loss = loss / float(i+1)
  print("\nAverage loss:\t{}\n".format(avg_loss))
  #predict test set (better to take average on all folds, but this is quicker)
  print("Test Fold 1/1")
  clf.fit(X_train, y)
  dataset_blend_test = clf.predict_proba(X_test)

  if clf_name == "XX":
    clf_name = str(clf)[1:3]

  if len(save_preds)>0:
    id = md5.new("{}".format(str(clf.get_params()))).hexdigest()
    print("storing meta predictions at: {}".format(save_preds))
    np.save("{}_{}_{}_{}_train.npy".format(save_preds,clf_name,avg_loss,id),dataset_blend_train)
    np.save("{}_{}_{}_{}_test.npy".format(save_preds,clf_name,avg_loss,id),dataset_blend_test)

  if len(save_test_only)>0:
    id = md5.new("{}".format(str(clf.get_params()))).hexdigest()
    print("storing meta predictions at: {}".format(save_test_only))

    dataset_blend_test = clf.predict(X_test)
    np.savetxt("{}_{}_{}_test.txt".format(save_test_only,clf_name,avg_loss,id),dataset_blend_test)
    d = {}
    d["stacker"] = clf.get_params()
    d["generalizers"] = generalizers_params
    with open("{}_{}_{}_params.json".format(save_test_only,clf_name,avg_loss, id), 'wb') as f:
      json.dump(d, f)

  if len(save_params)>0:
    id = md5.new("{}".format(str(clf.get_params()))).hexdigest()
    d = {}
    d["name"] = clf_name
    d["params"] = { k:(v.get_params() if "\n" in str(v) or "<" in str(v) else v) for k,v in clf.get_params().items()}
    d["generalizers"] = generalizers_params
    with open("{}_{}_{}_{}_params.json".format(save_params,clf_name,avg_loss, id), 'wb') as f:
      json.dump(d, f)

  if np.unique(y).shape[0] == 2: # when binary classification only return positive class proba
    if return_score:
      return dataset_blend_train[:,1], dataset_blend_test[:,1], avg_loss
    else:
      return dataset_blend_train[:,1], dataset_blend_test[:,1]
  else:
    if return_score:
      return dataset_blend_train, dataset_blend_test, avg_loss
    else:
      return dataset_blend_train, dataset_blend_test