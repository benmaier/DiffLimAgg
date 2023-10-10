import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as pl

LGiant = 1
nTown = 600
L = LGiant/3
eps = np.linspace(0,L,9)

fig, ax = pl.subplots(3,3,figsize=(8,8))
ax = ax.flatten()
figL, axL = pl.subplots(3,3,figsize=(8,8))
axL = axL.flatten()

pl.figure()
x = L*np.random.rand(nTown,2)
dists = pdist(x)
pl.hist(dists, bins=300,density=True)


for i, e in enumerate(eps):
    x = np.array([])
    y = np.array([])
    for j in range(-1,2):
        for k in range(-1,2):
            x0 = (L*np.random.rand(nTown) - L/2) + (L+e) * j
            y0 = (L*np.random.rand(nTown) - L/2) + (L+e) * k
            x = np.concatenate([x,x0])
            y = np.concatenate([y,y0])
    pos = np.array([x,y]).T
    dists = pdist(pos)
    ax[i].hist(dists, bins=300,density=True,histtype='step')
    axL[i].hist(dists, bins=np.logspace(-2,-0.5,71,base=LGiant*10),density=True,histtype='step')
    axL[i].set_xscale('log')
    axL[i].set_yscale('log')

fig.tight_layout()

pl.figure()
x = L*np.random.rand(nTown)
dists = pdist(pos)
pl.hist(dists, bins=300,density=True)

pl.show()
