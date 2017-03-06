from PIL import Image
from pylab import *
from numpy import *
from scipy.cluster.vq import *
import imtools
from sklearn.decomposition import PCA

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

n = len(projected)
# compute distance matrix
S = array([[ sqrt(sum((projected[i]-projected[j]) ** 2))
for i in range(n) ] for j in range(n)],  'f')
# create Laplacian matrix
rowsum = sum(S, axis=0)
D = diag(1 / sqrt(rowsum))
I = identity(n)
L = I - dot(D, dot(S, D))
# compute eigenvectors of L
U, sigma, V = linalg.svd(L)
k = 5
# create feature vector from k first eigenvectors
# by stacking eigenvectors as columns
features = array(V[:k]).T

# k-means
features = whiten(features)
centroids, distortion = kmeans(features, k)
code, distance = vq(features, centroids)
# plot clusters
gray()
for c in range(k):
    ind = where(code==c)[0]
    figure()
    for i in range(minimum(len(ind), 39)):
        im = Image.open(imlist[ind[i]]).convert('L')
        subplot(4, 10, i+1)
        imshow(im)
        axis('equal')
        axis('off')
show()