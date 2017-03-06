from PIL import Image
from pylab import *
from scipy.cluster.vq import *
from scipy.misc import imresize

steps = 50 #image is divided in steps * steps region
im = array(Image.open('empire.jpg'))
dx = int(im.shape[0] / steps)
dy = int(im.shape[1] / steps)
# compute color features for each region
features = []
for x in range(steps):
    for y in range(steps):
        R = mean(im[x * dx:(x+1) * dx, y * dy:(y+1) * dy, 0])
        G = mean(im[x * dx:(x+1) * dx, y * dy:(y+1) * dy, 1])
        B = mean(im[x * dx:(x+1) * dx, y * dy:(y+1) * dy, 2])
        features.append([R, G, B])
features = array(features,'f') # make into array

# cluster
centroids, variance = kmeans(features, 3)
code, distance = vq(features, centroids)

# create image with cluster labels
codeim = code.reshape(steps, steps)
codeim = imresize(codeim, im.shape[:2], interp='nearest')

figure()
imshow(codeim)
show()