from pylab import *
from numpy import *

import hcluster

class1 = 1.5 * np.random.randn(100, 2)
class2 = np.random.randn(100, 2) + array([5, 5])

features = vstack((class1, class2))
tree = hcluster.hcluster(features)
clusters = tree.extract_clusters(5)
print(len(clusters))
for c in clusters:
    print(c.get_cluster_elements())
