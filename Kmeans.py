import os   

import numpy as np
import rasterio as rio

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.exposure import equalize_adapthist

import matplotlib.pyplot as plt

path = "Data/"
complete_dataset = os.listdir(path)
complete_dataset = [path + x for x in complete_dataset]
print(complete_dataset)