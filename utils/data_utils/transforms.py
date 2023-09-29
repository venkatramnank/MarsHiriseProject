from skimage import exposure
from torchvision import datasets, transforms

def apply_gamma_correction(img, cols=None, rows=None):
    """Applies Gamma correction for equalizing luminance

    Args:
        img (numpy array): Image read through cv2

    Returns:
        numpy: LUT gamma equalized image
    """
    # Apply gamma correction to the image
    gamma = 1
    return exposure.adjust_gamma(img)

transform_test = transforms.Compose([
    transforms.ToTensor()
])

def apply_equalize_histogram(img, cols=None, rows=None):
    """Applies Histogram Equalization

    Args:
        img (numpy array): Image read through cv2

    Returns:
        numpy: histogram equalized image
    """
    return exposure.equalize_hist(img)