import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as pl

L = 1
eps = np.linspace(0,L,9)
nTown = 5000

fig, ax = pl.subplots(3,3,figsize=(8,8))
ax = ax.flatten()
figL, axL = pl.subplots(3,3,figsize=(8,8))
axL = axL.flatten()

pl.figure()
x = L*np.random.rand(nTown,2)
dists = pdist(x)
pl.hist(dists, bins=300,density=True)


for i, e in enumerate(eps):
    x0 = L*np.random.rand(nTown) - L/2
    x1 = L*np.random.rand(nTown) + L/2 + e
    y = 2*L*np.random.rand(2*nTown)
    x = np.concatenate([x0,x1])
    pos = np.array([x,y]).T
    dists = pdist(pos)
    ax[i].hist(dists, bins=300,density=True,histtype='step')
    axL[i].hist(dists, bins=np.logspace(-2,-1,51,base=L*10),density=True,histtype='step')
    axL[i].set_xscale('log')
    axL[i].set_yscale('log')

fig.tight_layout()

pl.figure()
x = L*np.random.rand(nTown)
dists = pdist(pos)
pl.hist(dists, bins=300,density=True)

pl.show()
