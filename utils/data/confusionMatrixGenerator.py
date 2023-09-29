from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

def confusion_matrix_plotter(model, test_dataloader):
    """Plots confusion matrix

        Args:
            model (Pytorch model): Pytorch model
            test_dataloader (dataLoader): test dataloader
    """
    y_pred = []
    y_true = []
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # iterate over test data
    for inputs, labels, _ in test_dataloader:
            inputs = inputs.float().to(dev)
            labels = labels.to(dev)
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ('0', '1', '2', '3', '4',
            '5', '6', '7')

    # Build confusion matrix
    logger.info('BUILDING CONFUSION MATRIX')
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('../results/cfmatrices/output.png')