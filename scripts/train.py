import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from utils.data.hiriseDataset import hiriseImageDataset
from utils.data.augmentations import train_transforms, test_transforms
from utils.data.weightedSampler import weighted_random_sampler
from losses.focal_loss import FocalLoss
from models.resnet import get_resnet_model
from scripts.model_trainer import train_model
from loguru import logger


BATCH_SIZE = 64
NUM_EPOCHS = 20

def train(load_trained, pretrained_dict_path):
    """Train main funtion

    Args:
        pretrained_dict_path (path): Path of pretrained model
        load_trained (bool, optional): Load trained. Defaults to True.

    """
    dev =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('***************************************************')
    print('***************************************************')
    print('Device used : \x1b[6;30;42m'+ str(dev)+'\x1b[0m')
    print('***************************************************')
    print('***************************************************')
    logger.info('BUILDING HIRISE IMAGE DATASETS')
    train_dataset_subset = hiriseImageDataset('../data/AllDatasets/train_labels.txt', '../data/AllDatasets/train', '../data/landmarks_map-proj-v3_classmap.csv',transform=train_transforms)
    valid_dataset_subset = hiriseImageDataset('../data/AllDatasets/val_labels.txt', '../data/AllDatasets/val', '../data/landmarks_map-proj-v3_classmap.csv',transform=test_transforms)
    test_dataset = hiriseImageDataset('../data/AllDatasets/test_labels.txt', '../data/AllDatasets/test', '../data/landmarks_map-proj-v3_classmap.csv',transform=test_transforms)
    logger.success('SUCESSFULLY BUILT HIRISE IMAGE DATASETS')

    weighted_sampler = weighted_random_sampler('../data/AllDatasets/train_labels.txt', replacement=False)
    logger.info('BUILDING HIRISE IMAGE DATALOADER')
    train_iterator = DataLoader(train_dataset_subset,  shuffle=True,
                                    batch_size = BATCH_SIZE)

    # Run below line to build train dataloader based on weighted sampler
    # train_iterator = DataLoader(train_dataset_subset,  sampler=weighted_sampler,
    #                                  batch_size = BATCH_SIZE)

    valid_iterator = DataLoader(valid_dataset_subset,
                                    batch_size = BATCH_SIZE)


    test_iterator = DataLoader(test_dataset, 
                                    batch_size = BATCH_SIZE)

    test_gradcam_iterator = DataLoader(test_dataset, batch_size = 1)
    logger.success('SUCESSFULLY BUILT HIRISE IMAGE DATALOADERS')
    pretrained_model = get_resnet_model(num_classes=8)

    # Uncomment the line below for Cross Entropy Loss
    # criterion = nn.CrossEntropyLoss()

    criterion = FocalLoss()
    optimizer = optim.Adam(pretrained_model.fc.parameters())

    datasets = {'train':train_dataset_subset, 'valid':valid_dataset_subset}
    dataloaders = {'train': train_iterator, 'valid':valid_iterator}

    if load_trained == False or pretrained_dict_path is None:
        logger.info('TRAINING IN PROGRESS ...')
        model_trained = train_model(pretrained_model, criterion, optimizer, NUM_EPOCHS, dataloaders, datasets)
        torch.save(model_trained.state_dict(), '../chkpts/current_run.pth')
        return model_trained, train_iterator, valid_iterator, test_iterator, test_gradcam_iterator
    else:
        logger.info('LOADING TRAINED MODEL')
        if dev == torch.device('cpu'):
            pretrained_model.load_state_dict(torch.load('../chkpts/'+pretrained_dict_path,  map_location=torch.device('cpu')))
        else:
            pretrained_model.load_state_dict(torch.load('../chkpts/'+pretrained_dict_path))
        return pretrained_model, train_iterator, valid_iterator, test_iterator, test_gradcam_iterator

