import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch import nn
from models.gradcamModel import GradCamModel
from loguru import logger

def gradcam_image_gen(test_gradcam_iterator, image_num):
    """Generate gradcam images for one image from the test gradcam dataloader

    Args:
        test_gradcam_iterator (DataLoader): Gradcam dataloader with batch size = 1
        image_num (int): Number of the image within the size of gradcam test dataloader
    """
    dev =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('INITIATING GRADCAM PLOTTER')
    gcmodel = GradCamModel().to(dev)
    images_list = []
    for img,_,_ in test_gradcam_iterator:
        images_list.append(img)
    inpimg = images_list[image_num]

    out, acts = gcmodel(inpimg.float().to('cuda'))
    acts = acts.detach().cpu()
    loss = nn.CrossEntropyLoss()(out,torch.from_numpy(np.array([2])).to('cuda'))
    loss.backward()
    grads = gcmodel.get_act_grads().detach().cpu()
    pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()
    for i in range(acts.shape[1]):
        acts[:,i,:,:] *= pooled_grads[i]
    heatmap_j = torch.mean(acts, dim = 1).squeeze()
    heatmap_j_max = heatmap_j.max(axis = 0)[0]
    heatmap_j /= heatmap_j_max
    plt.matshow(heatmap_j.squeeze())
    cmap = matplotlib.cm.get_cmap('jet',256)
    heatmap_j2 = cmap(heatmap_j,alpha = 0.2)
    img = inpimg.numpy()
    img = np.resize(img, (227,227))
    fig, axs = plt.subplots(1,1,figsize = (5,5))
    axs.imshow(img)
    axs.imshow(heatmap_j2)
    plt.show()


