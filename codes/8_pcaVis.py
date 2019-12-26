from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


start = time.time()

# load data
mat = h5py.File('/home/cyyan/projects/ccRCC/results/feats.mat')
print(list(mat.keys()))
feats = np.array(mat.get('deepfeats')).transpose()
print(feats.shape)

mat = h5py.File('/home/cyyan/projects/ccRCC/results/prediction.mat')
print(list(mat.keys()))
pred = np.array(mat.get('pred')).transpose()[:,0]
print(pred.shape)

fh = open('/home/cyyan/projects/ccRCC/data/test.txt', 'r')
labels = np.array([])
for line in fh:
    line = line.rstrip()
    words = line.split(' ')[-1]
    labels = np.append(labels, int(words))
print(labels.shape)

TPNF = np.empty(labels.shape)
TPNF[(pred == 0) & (labels == 0)] = 0
TPNF[(pred == 1) & (labels == 0)] = 1
TPNF[(pred == 1) & (labels == 1)] = 2
TPNF[(pred == 0) & (labels == 1)] = 3
print(TPNF.shape)

# model analysis
feats_pca = PCA(n_components=3).fit_transform(feats)
feats_tsne = TSNE(learning_rate=100, perplexity=100).fit_transform(feats)


# PCA 3D visualization
fig = plt.figure(figsize=(5,10))
# ax = Axes3D(fig, rect=[0,0,1,1], elev=30, azim=20)
# plt.scatter(feats_pca[0:10136,0], feats_pca[0:10136,1], feats_pca[0:10136,2], c=labels, marker='.')

# PCA 2D visualization
plt.subplot(211)
plt.scatter(feats_pca[:,0], feats_pca[:,1], c=TPNF, marker='.')

# TSNE 2D visualization
plt.subplot(212)
plt.scatter(feats_tsne[:,0], feats_tsne[:,1], c=TPNF, marker='.')

plt.show()
plt.savefig('../results/pca_tsne.png', dpi=600, bbox_inches='tight')

print(time.time()-start)

