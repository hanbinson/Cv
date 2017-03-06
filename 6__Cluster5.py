from PIL import Image
from numpy import *
import os
import hcluster

# create a list of images
path = './sunsets/'
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
# extract feature vector (8 bins per color channel)
features = zeros([len(imlist), 512])
for i, f in enumerate(imlist):
    im = array(Image.open(f))
    # multi-dimensional histogram
    h, edges = histogramdd(im.reshape(-1, 3), 8, normed=True, range=[(0, 255), (0, 255), (0, 255)])
    features[i] = h.flatten()
tree = hcluster.hcluster(features)

hcluster.draw_dendrogram(tree, imlist, filename='sunset.pdf')
