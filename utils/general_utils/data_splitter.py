import os
import numpy as np
import shutil
import random

# creating train / val /test
root_dir = '/home/venkat/OSU/ML_challenges_winter23/Project/MLProjectNASA/data/'
new_root = 'AllDatasets/'



# os.makedirs(root_dir + new_root+ 'train/')
# os.makedirs(root_dir +new_root +'val/')
# os.makedirs(root_dir +new_root + 'test/')
    
## creating partition of the data after shuffeling


src = '../data/map-proj-v3' # folder to copy images from


allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)

## here 0.75 = training ratio , (0.95-0.75) = validation ratio , (1-0.95) =  
##training ratio  
train_FileNames,val_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.75),int(len(allFileNames)*0.95)])

# #Converting file names from array to list

train_FileNames = [src+'/'+ name for name in train_FileNames]
val_FileNames = [src+'/' + name for name in val_FileNames]
test_FileNames = [src+'/' + name for name in test_FileNames]

print('Total images  : ' +str(len(allFileNames)))
print('Training : '+str(len(train_FileNames)))
print('Validation : '+str(len(val_FileNames)))
print('Testing : '+str(len(test_FileNames)))

## Copy pasting images to target directory

for name in train_FileNames:
    shutil.copy(name, '../data/AllDatasets/train/')


for name in val_FileNames:
    shutil.copy(name, '../data/AllDatasets/val/')


for name in test_FileNames:
    shutil.copy(name,'../data/AllDatasets/test/')