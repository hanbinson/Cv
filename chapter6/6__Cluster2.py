from PIL import Image
from pylab import *
from numpy import *
import imtools
from scipy.cluster.vq import *
from sklearn.decomposition import PCA
from PIL import ImageDraw

# get list of images
imlist = imtools.get_imlist('selected_fontimages/')

im = array(Image.open(imlist[0])) # open one image to get size
m, n = im.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images
# create matrix to store all flattened images
immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')

pca = PCA()
X_transformed = pca.fit_transform(immatrix)

# We center the data and compute the sample covariance matrix.
mean = np.mean(immatrix, axis=0)
X_centered = immatrix - mean
cov_matrix = np.dot(X_centered.T, X_centered) / imnbr
eigenvalues = pca.explained_variance_
eigenvector = pca.components_

# project on the 40 first PCs
immean = mean.flatten()

projected = array([dot(eigenvector[:3], immatrix[i] - immean) for i in range(imnbr)])
# k-means
projected = whiten(projected)
centroids, distortion = kmeans(projected, 4)

code, distance = vq(projected, centroids)

# plot clusters
for k in range(4):
    ind = where(code==k)[0]
    figure()
    gray()
    for i in range(minimum(len(ind),40)):
        subplot(4,10,i+1)
        imshow(immatrix[ind[i]].reshape((25,25)))
        axis('off')

show()

# height and width
h, w = 1200, 1200
# create a new image with a white background
img = Image.new('RGB',(w,h),(255,255,255))
draw = ImageDraw.Draw(img)
# draw axis
draw.line((0, h/2, w, h/2), fill=(255, 0, 0))
draw.line((w/2, 0, w/2, h), fill=(255, 0, 0))
# scale coordinates to fit
scale = abs(projected).max(0)
scaled = floor(array([(p / scale) * (w/2-20, h/2-20) + (w/2, h/2) for p in projected]))
# paste thumbnail of each image
for i in range(imnbr):
    nodeim = Image.open(imlist[i])
    nodeim.thumbnail((25,25))
    ns = nodeim.size
    img.paste(nodeim, (int(scaled[i][0]-ns[0]//2), int(scaled[i][1] - ns[1]//2), int(scaled[i][0]+ns[0]//2+1), int(scaled[i][1]+ns[1]//2+1)))
img.save('pca_font.jpg')

