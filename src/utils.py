#!/usr/bin/env python
"""Explaining NASA images: Project"""
""" Utilities """
__author__ = "Venkat Ramnan K"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "kalyanav@gmail.com"

from collections import Counter
import matplotlib.pyplot as plt
import cv2

def dataProfiler(dataframe):
    labels = list(dataframe['label'])
    count = Counter(labels)
    labels = []
    label_count = []
    for keys, values in count.items():
        labels.append(keys)
        label_count.append(values)
    print("The classes are  : ", labels)
    print("Number of classes : ", len(labels))
    print("The maximum class is : ", max(count, key=count.get))
    plt.bar(labels, label_count, color ='maroon',
        width = 0.4)
    plt.xlabel('Classes')
    plt.ylabel('Class Count')
    plt.title("Class count")
    plt.show()


def histogramPlotter(img_file):
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    # plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    # normalize the histogram
    hist /= hist.sum()
    # plot the normalized histogram
    plt.figure()
    plt.title("Grayscale Histogram (Normalized)")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def edgeSobel(img_file):
    img = cv2.imread(img_file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurring the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    # cv2.imwrite(str(img_file)+'sobel.jpg', sobelxy)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    cv2.imshow('Canny Edge Detection', edges)
    # cv2.imwrite(str(img_file)+'canny.jpg', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift(img_file):
    img = cv2.imread(img_file)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img)
    cv2.imshow('SIFT', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()