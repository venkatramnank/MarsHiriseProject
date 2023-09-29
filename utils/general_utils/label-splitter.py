import pandas as pd
import os
import copy

trainFileNames = os.listdir('../data/AllDatasets/train')
valFileNames = os.listdir('../data/AllDatasets/val')
testFileNames = os.listdir('../data/AllDatasets/test')

main_label_df = pd.read_csv('../data/labels.txt', delimiter=' ',header=None, names=['name', 'class'])

train_labels_df = copy.deepcopy(main_label_df)
val_labels_df = copy.deepcopy(main_label_df)
test_labels_df = copy.deepcopy(main_label_df)

train_labels_df_new = copy.deepcopy(train_labels_df)
val_labels_df_new = copy.deepcopy(val_labels_df)
test_labels_df_new = copy.deepcopy(test_labels_df)

for index,row in train_labels_df.iterrows():
    if row['name'] not in trainFileNames:
        print('removing row in traning data ... ')
        train_labels_df_new.drop(index, inplace=True)

for index,row in val_labels_df.iterrows():
    if row['name'] not in valFileNames:
        print('removing row in val data ... ')
        val_labels_df_new.drop(index, inplace=True)

for index,row in test_labels_df.iterrows():
    if row['name'] not in testFileNames:
        print('removing row in test data ... ')
        test_labels_df_new.drop(index, inplace=True)

train_labels_df_new.to_csv('../data/AllDatasets/train_labels.txt', sep=' ', index=False)
val_labels_df_new.to_csv('../data/AllDatasets/val_labels.txt', sep=' ', index=False)
test_labels_df_new.to_csv('../data/AllDatasets/test_labels.txt', sep=' ', index=False)