import glob,random
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from utils.data_utils.transforms import transform_test
import matplotlib.pyplot as plt

def visualizer(pretrained_model):
    """Visualize the confidence score for 3 random images in test data

    Args:
        pretrained_model (model): pretrained model
    """
    dev =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_img_labels = pd.read_csv('../data/AllDatasets/test_labels.txt', delimiter=' ',header=None,names=['name','class'])
    test_images = test_img_labels['name'].sample(n=3, random_state=1).tolist()
    test_imgs = []

    for img_path in test_images:
        image = cv2.imread('../data/AllDatasets/AllDatasets/test/'+img_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_imgs.append(gray_image)

    validation_batch = torch.stack([transform_test(img).to(dev)
                                    for img in test_imgs])
    pred_logits_tensor = pretrained_model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    fig, axs = plt.subplots(1, len(test_imgs), figsize=(20, 5))
    for i, img in enumerate(test_imgs):
        ax = axs[i]
        ax.axis('off')
        ax.set_title("{:.0f}% Class 0, {:.0f}% 1, {:.0f}% 2, {:.0f}% 3, {:.0f}% 4, {:.0f}% 5, {:.0f}% 6, {:.0f}% 7 ".format(100*pred_probs[i,0],
                                                                100*pred_probs[i,1],
                                                                100*pred_probs[i,2],
                                                                100*pred_probs[i,3],
                                                                100*pred_probs[i,4],
                                                                100*pred_probs[i,5],
                                                                100*pred_probs[i,6],
                                                                100*pred_probs[i,7],))
        ax.imshow(img)