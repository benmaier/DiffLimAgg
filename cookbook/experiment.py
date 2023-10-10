import numpy as np
import scipy.spatial as spc
from DiffLimAgg.walkers import Walkers2D
from DiffLimAgg.experiment import Experiment
from DiffLimAgg.boxcounting import box_counting_analysis

from scipy.optimize import curve_fit

from fast_histogram import histogram2d

powerlaw = lambda x, a, b: b*x**a
linear = lambda x, b: b*x

if __name__=="__main__":
    import matplotlib.pyplot as pl

    N = 3_000
    dt = 0.1
    #force_field = lambda x: -(x-0.5) / (np.sqrt(((x)**2).sum(axis=1)))[:,None] / 2
    force_field = lambda x: -(x-0.5).dot(np.array([[0,1],[-1,0]])) / (np.sqrt(((x-0.5)**2).sum(axis=1)))[:,None] /1000
    #print(force_field(np.array([[1,1.0]])))
    #walkers = Walkers2D(N, dt, initial_positions='mixed',position_noise_coefficient=1e-8,velocity_noise_coefficient=1e-16, force_field=force_field)
    walkers = Walkers2D(N, dt, initial_positions='mixed',position_noise_coefficient=1e-8,velocity_noise_coefficient=1e-16, force_field=force_field)
    #pl.plot(walkers.x[:,0], walkers.x[:,1],'.',markersize=1)
    #pl.axis('square')
    exp = Experiment(walkers,average_walker_distance_factor_for_radius=0.2)
    #exp.step()
    exp.simulate(verbose=True)
    #print(exp.links)
    #print(exp.times)

    fig, ax = pl.subplots(2,3,figsize=(10,6))

    pl.sca(ax[1,0])
    for i, j in exp.links:
        pl.plot(walkers.x[[i,j],0]-0.5, walkers.x[[i,j],1]-0.5,'-',c='#aaaaaa')
    pl.plot(walkers.x[:,0]-0.5, walkers.x[:,1]-0.5,'.',markersize=1)
    pl.axis('square')
    pl.xlim(np.array(walkers.boxx)-0.5)
    pl.ylim(np.array(walkers.boxy)-0.5)

    distances = spc.distance.pdist(walkers.x)
    for ihist, a in enumerate(ax[0,:2]):

        if ihist == 1:
            bins = 101
        else:
            lower = np.log(exp.walker_radius/10)/np.log(10)
            upper = np.log(exp.walker_radius*10)/np.log(10)
            bins = np.logspace(lower,
                               upper,
                               201)
            print(bins)
        dens, bin_edges, patches = a.hist(distances,bins=bins,histtype='step',density=True)
        ndx = 10
        if i == 0:
            _x = np.sqrt(bin_edges[1:]*bin_edges[:-1])
            xfit =_x
            yfit = dens
        else:
            _x = 0.5*(bin_edges[1:]+bin_edges[:-1])
            xfit = _x[:ndx]
            yfit = dens[:ndx]
        ppow, _ = curve_fit(powerlaw,xfit, yfit,p0=(1,2/3))
        plin, _ = curve_fit(linear,xfit, yfit,p0=(1,))
        a.plot(xfit, powerlaw(xfit,*ppow),'--k',label='r^a, {0:4.3f}'.format(ppow[0]))
        a.plot(xfit, linear(xfit,*plin),':k',label='r')
        a.set_xlabel('pairwise distance r/L')
        a.set_ylabel('pdf')
    #ax[0,0].set_ylim(yfit[0]/2,ax[0,0].get_ylim()[-1])
        a.legend(loc='lower center')

    dist_to_center = np.linalg.norm(walkers.x - 0.5,axis=1)
    rmean = dist_to_center.mean()
    dens, be, _ = ax[0,2].hist(dist_to_center,bins=50,histtype='step', density=True)
    _x = 0.5*(be[1:]+be[:-1])
    lam = 2/rmean
    ax[0,2].plot(_x, lam**2*_x*np.exp(-lam*_x),'-k')
        #a.plot(bin_edges[:20], bin_edges[:20]**(1))
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    #ax[0,0].set_xlim(bin_edges[1], bin_edges[-1])

    ax[0,2].set_xlabel('distance to center d/L')
    ax[0,2].set_ylabel('pdf')


    hist2 = histogram2d(walkers.x[:,0]-0.5,
                        walkers.x[:,1]-0.5,
                        bins=50,
                        range=[(-0.5,0.5)]*2,
                        )
    ax[1,1].imshow(hist2,cmap='Greens')

    fig.tight_layout()

    box_counting_analysis(walkers.x[:,0],
                          walkers.x[:,1]
                          )

    pl.show()

