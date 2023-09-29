#!/usr/bin/env python
"""Explaining NASA images: Project"""
__author__ = "Venkat Ramnan K"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "kalyanav@oregonstate.edu"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import sys
sys.path.insert(1, '/home/venkat/OSU/ML_challenges_winter23/Project/MLProjectNASA/')
load_dotenv(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".env"))
DATA_PATH = os.environ["DATA_PATH"]
from src import utils



df = pd.read_csv(f"{DATA_PATH}/data/labels.txt", sep=" ", names=['Image', 'label'])
print(utils.dataProfiler(df))

img_files = ['/home/venkat/OSU/ML_challenges_winter23/Project/MLProjectNASA/data/map-proj-v3/ESP_011283_2265_RED-0030-brt.jpg',
            '/home/venkat/OSU/ML_challenges_winter23/Project/MLProjectNASA/data/map-proj-v3/ESP_030136_0930_RED-0026-r90.jpg',
            '/home/venkat/OSU/ML_challenges_winter23/Project/MLProjectNASA/data/map-proj-v3/ESP_016631_2535_RED-0098-brt.jpg',
            '/home/venkat/OSU/ML_challenges_winter23/Project/MLProjectNASA/data/map-proj-v3/ESP_018416_2060_RED-0193-r90.jpg',
            '/home/venkat/OSU/ML_challenges_winter23/Project/MLProjectNASA/data/map-proj-v3/ESP_028196_1840_RED-0411-r270.jpg',
            '/home/venkat/OSU/ML_challenges_winter23/Project/MLProjectNASA/data/map-proj-v3/ESP_011425_1775_RED-0060-r90.jpg ']
for img_file in img_files:
    # print(utils.histogramPlotter(img_file))
    # print(utils.edgeSobel(img_file))
    print(utils.sift(img_file))