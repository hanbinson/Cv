from pylab import *
from numpy import *
from scipy.cluster.vq import *

class1 = 3.5 * np.random.randn(100, 2)
class2 = np.random.randn(100, 2) + array([5, 5])
features = vstack((class1, class2))

print(features)

centroids, variance = kmeans(features, 2)

code, distance = vq(features, centroids)

figure()
ndx = where(code == 0)[0]
plot(features[ndx, 0], features[ndx, 1], ' * ')
ndx = where(code == 1)[0]
plot(features[ndx, 0], features[ndx, 1], 'r.')
plot(centroids[:, 0], centroids[:, 1], 'go')
axis('off')
show()

