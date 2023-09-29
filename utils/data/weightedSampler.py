import pandas as pd
from collections import Counter
import torch
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

def weightedSampler(annotations_file):
    """Calculates the class weights

    Args:
        annotations_file (file): Annotations file

    Returns:
        weights(list): List of weights
        num_samples(int): Number of samples in terms of class count
    """
    img_labels = pd.read_csv(annotations_file, delimiter=' ',header=None, names=['name', 'class'])
    labels = img_labels['class'].tolist()
    class_count = list(Counter(labels).values())
    num_samples = sum(class_count)
    class_weights = [num_samples/class_count[i] for i in range(len(class_count))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]

    return weights, num_samples

def weighted_random_sampler(annotations_file, replacement=False):
    """Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Args:
        annotations_file (file): Annotation File
        replacement (bool, optional): Replacement switch for weighted sampler. Defaults to False.

    Returns:
        pytorch sampler: weighted sampler
    """
    weights, num_samples = weightedSampler(annotations_file)
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=int(num_samples), replacement=replacement)