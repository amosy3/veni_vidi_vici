from collections import defaultdict
from glob import glob
import sys
import os
import pandas as pd
from sklearn import preprocessing
from feature_extractor import quantile_normalize

glob_files = '../../models_predictions/*.csv'
loc_outfile ="../../models_predictions/ensembles/avarage_ensemble.csv"


def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
  if method == "average":
    scores = defaultdict(float)
  with open(loc_outfile,"w") as outfile:
    for i, glob_file in enumerate(glob(glob_files)):
      print("parsing: {}".format(glob_file))
      # sort glob_file by first column, ignoring the first line
      lines = open(glob_file).readlines()
      lines = [lines[0]] + sorted(lines[1:])
      for e, line in enumerate( lines ):
        if i == 0 and e == 0:
          outfile.write(line)
        if e > 0:
          row = line.strip().split(",")
          scores[(e,row[0])] += float(row[1])
    for j,k in sorted(scores):
      outfile.write("%s,%f\n"%(k,scores[(j,k)]/(i+1)))
    print("wrote to {}".format(loc_outfile))

kaggle_bag(glob_files, loc_outfile)


# quantile_normalize('../../models_predictions')