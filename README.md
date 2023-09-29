__author__ = "Venkat Ramnan K"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "kalyanav@oregonstate.com"
__date__ = March 20 2023


# Directory Structure   
```bash
.
├── backup_script
├── chkpts
├── data
├── docs
├── images
│   └── gradcam
├── losses
│   └── __pycache__
├── models
│   └── __pycache__
├── notebooks
├── results
│   └── cfmatrices
├── scripts
│   └── __pycache__
├── src
│   └── __pycache__
├── tests
└── utils
    ├── data
    │   └── __pycache__
    ├── data_utils
    │   └── __pycache__
    ├── explainable_utils
    │   └── __pycache__
    ├── general_utils
    └── __pycache__
```
# Description of project

The goal behind the project is to train a classifier to classify the mars HiRISE images. The focus would be
to understand the decisions made by the classifier so as to understand the interpretability of the model. This
project aims to build an explainable CNN model to understand the following:
• Why did my CNN classify this object as such?
• Can I trust my CNN model?
This project serves as an introduction to learn building explainable and interpretable AI models. This will help
use trust us the ML model and truly understand the insights of the data. The final product can help build ML
modules that can be integrated with multiple decision making tasks, in this understanding what my model
sees and explain why what my camera sees is indeed a crater. In some cases when the ML model classifies a
completely new object into one of the known classes, it lets us know why and where the model looked at to
make such a decision. Finally this will truly help us the bias factor in the model and understand what the model
prejudices over.

# Prerequisites


In order to run the files for ML Challenges project, you will need python and pip3. 
(You can also use a conda environment)

Some important points:
1. It is highly recommended to have a machine with CUDA supported GPU for pytorch
2. Building conda environment is recommended for mangaging packages and modules
3. (Not recommended) The program is setup to use Pytorch CPU as well if GPU is not available
4. For visualization, please see the jupyter notebooks present in notebooks folder
5. (Recommended) There is a backup code in backup_script folder. If the following evaluate.py file fails, it is recommended to use the python script in this folder.

### Checkpoints and Dataset
The checkpoint for the Resnet 34 is present at : https://drive.google.com/file/d/1-2bUT6kls9bGVSbAiXc9saFzbdIPM0x2/view?usp=sharing 

The original dataset is present at : https://doi.org/10.5281/zenodo.4002935 

The data which has been split is present at : https://drive.google.com/file/d/1phWbjBjJePITe85z9gJmByPu4HH6GXkE/view?usp=sharing 


### pip

If pip, after activating environment, run :
```pip install -r requirements.txt```


### conda

```conda activate <env>```
```conda install --file requirements.txt```


# Usage

Activate the conda environment using 
```conda activate env_name```

Change directory to scripts.
To run using pretrained model (must be in chkpts directory) :
```python evaluate.py --pretrained-model-path <path to pth file>```

To train a new model:
```python evaluate.py```

To run the backup code, after getting into backup_script directory :
```python mlchallengesprojectfocalloss.py```

The newly trained model weights are stored as a pth file in results folder. To re use them, please store them in chkpts directory.


# License

This project is licensed under the MIT License.
