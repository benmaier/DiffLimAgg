import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as pl

L = 1
eps = np.linspace(0,L,9)
nTown = 5000

fig, ax = pl.subplots(3,3,figsize=(8,8))
ax = ax.flatten()

pl.figure()
x = L*np.random.rand(nTown)
dists = pdist(x.reshape(nTown,1))
pl.hist(dists, bins=300,density=True)
pl.show()


for i, e in enumerate(eps):
    x0 = L*np.random.rand(nTown) - L/2
    x1 = L*np.random.rand(nTown) + L/2 + e
    pos = np.concatenate([x0,x1]).reshape(2*nTown, 1)
    dists = pdist(pos)
    ax[i].hist(dists, bins=300,density=True)


pl.figure()
x = L*np.random.rand(nTown)
dists = pdist(pos)
pl.hist(dists, bins=300,density=True)

pl.show()
