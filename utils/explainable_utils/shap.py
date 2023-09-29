import shap
import numpy as np
import torch
from loguru import logger

def shap_explainer(model, test_iterator):
    """SHAP Explainer for a set of images in shap explainer

    Args:
        model (model): trained model
        test_iterator (DataLoader): Test data loader
    """
    dev =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('INITIATING SHAP DEEPEXPLAINER')
    batch = next(iter(test_iterator))
    images, _, _ = batch
    images = images.view(-1, 1, 227, 227)
    background = images[5:65]
    test_images= images[0:5]
    e = shap.DeepExplainer(model, images.float().to(dev))
    shap_values = e.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    shap.image_plot(shap_numpy, -test_numpy)