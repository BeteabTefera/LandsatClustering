from skimage.exposure import equalize_adapthist
from skimage import exposure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio as rio
import re
from ClusteredBands import *

path = '/home/bear/Downloads/LC09_L2SP_024028_20230323_20230325_02_T1/'
complete_dataset = os.listdir(path)
# complete_dataset = [path + x for x in complete_dataset if re.match(r'.*_B\d{1,2}.TIF$', x)] # get all the band files
complete_dataset = [path + x for x in complete_dataset if re.match(r'.*_B[0-7].TIF$', x)] # get all the band files

for i in complete_dataset:
    print(i)

from skimage import exposure

def show_rgb(bands_list, red=4, green=3, blue=2):
    stack = []
    colors = [red, green, blue]
    colors = ['B' + str(x) for x in colors]
    for color in colors:
        for band in bands_list:
            if color in band:
                with rio.open(band) as src:
                    array = src.read(1)
                    stack.append(array)
                break
    stack = np.dstack(stack)
    for i in range(0, 3):
        stack[:, :, i] = exposure.rescale_intensity(stack[:, :, i], out_range=(0, 255))
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(stack)
    plt.show()

# show_rgb(complete_dataset)
clustered_models = ClusteredBands(complete_dataset)
clustered_models.set_raster_stack()

ranges = np.arange(3, 6, 1)

clustered_models.build_models(ranges)
clustered_models.show_clustered()
clustered_models.show_inertia()
clustered_models.show_silhouette_scores()
# show_rgb(complete_dataset, red=7, green=6, blue=4)




