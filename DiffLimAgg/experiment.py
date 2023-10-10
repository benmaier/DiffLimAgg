import numpy as np
import scipy.spatial as spc
from DiffLimAgg.walkers import Walkers2D

class Experiment:

    def __init__(self,
                 walkers,
                 walker_radius=None,
                 average_walker_distance_factor_for_radius=1,
                 ):

        self.N = walkers.N

        if walker_radius is None:
            density = self.N / walkers.Lx / walkers.Ly
            walker_radius = np.sqrt(1/density) * average_walker_distance_factor_for_radius
        self.walker_radius = walker_radius

        walkers.fix_first_walker()
        self.free_walkers = list(walkers._free_walkers)
        self.fixed_walkers = sorted(list(set(range(self.N))-set(self.free_walkers)))
        self.walkers = walkers

        self.links = []
        self.times = []
        self.this_step = 0

    def step(self):
        walkers = self.walkers

        self.this_step += 1
        walkers.step(self.free_walkers)
        fixed_pos = walkers.x[self.fixed_walkers,:]
        free_pos = walkers.x[self.free_walkers,:]

        tree = spc.KDTree(fixed_pos)

        _, nearest_fixed = tree.query(
                             free_pos,
                             k=1,
                             distance_upper_bound=2*self.walker_radius,
                         )

        ndx = np.where(nearest_fixed < len(self.fixed_walkers))[0]
        if len(ndx) > 0:
            new_fixed = [ self.free_walkers[i] for i in ndx ]
            attached_to = [ self.fixed_walkers[nearest_fixed[i]] for i in ndx ]

            self.links.extend(list(zip(attached_to, new_fixed)))
            self.times.extend([self.this_step for _ in new_fixed])

            for walker in new_fixed:
                position_in_list = self.free_walkers.index(walker)
                self.free_walkers.pop(position_in_list)
                self.fixed_walkers.append(walker)
            self.free_walkers = sorted(self.free_walkers)
            self.fixed_walkers = sorted(self.fixed_walkers)

    def simulate(self,verbose=False):

        if verbose:
            from tqdm import tqdm
            bar = tqdm(total=self.N)

        free_walkers_before = len(self.free_walkers)

        while len(self.free_walkers) > 0:
            if verbose and len(self.free_walkers) < free_walkers_before:
                bar.update(free_walkers_before-len(self.free_walkers))
            free_walkers_before = len(self.free_walkers)
            self.step()

        t, counts = np.unique(self.times[1:],return_counts=True)

        return (    (
                        t*self.walkers.dt,
                        np.array(counts,dtype=float)
                    ),
                    (
                        np.array(self.times)*self.walkers.dt,
                        np.array(self.links)
                    )
               )

if __name__=="__main__":
    import matplotlib.pyplot as pl

    N = 3_000
    dt = 0.1
    #force_field = lambda x: -(x-0.5) / (np.sqrt(((x)**2).sum(axis=1)))[:,None] / 2
    force_field = lambda x: -(x-0.5).dot(np.array([[0,1],[-1,0]])) / (np.sqrt(((x-0.5)**2).sum(axis=1)))[:,None] /1000
    #print(force_field(np.array([[1,1.0]])))
    walkers = Walkers2D(N, dt, initial_positions='mixed',position_noise_coefficient=0.0,velocity_noise_coefficient=0., force_field=force_field)
    #pl.plot(walkers.x[:,0], walkers.x[:,1],'.',markersize=1)
    #pl.axis('square')
    exp = Experiment(walkers,average_walker_distance_factor_for_radius=0.3)
    #exp.step()
    exp.simulate()
    #print(exp.links)
    #print(exp.times)

    fig, ax = pl.subplots(1,3,figsize=(8,3))

    for i, j in exp.links:
        pl.plot(walkers.x[[i,j],0], walkers.x[[i,j],1],'-',c='#aaaaaa')
    pl.plot(walkers.x[:,0], walkers.x[:,1],'.',markersize=1)
    pl.axis('square')
    pl.xlim(walkers.boxx)
    pl.ylim(walkers.boxy)

    distances = spc.distance.pdist(walkers.x)
    print(distances)
    for a in ax[:2]:
        dens, bin_edges, patches = a.hist(distances,bins=100,histtype='step',density=True)
        a.plot(bin_edges[:20], bin_edges[:20]**(2/3))
        a.plot(bin_edges[:20], bin_edges[:20]**(1))
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlim(bin_edges[1], bin_edges[-1])

    pl.show()

