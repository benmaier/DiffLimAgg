import numpy as np
import scipy.spatial as spc
from DiffLimAgg.walkers import Walkers2D
from DiffLimAgg.experiment import Experiment

import matplotlib.pyplot as pl
import matplotlib.animation as animation

def animate(experiment):

    Lx = experiment.walkers.Lx
    Ly = experiment.walkers.Ly
    fig, ax = pl.subplots(1,1,figsize=(8,12*Ly/Lx))
    pl.axis('equal')
    if experiment.walkers.Lx == experiment.walkers.Ly:
        pl.axis('square')
    ax.set_xlim(experiment.walkers.boxx)
    ax.set_ylim(experiment.walkers.boxy)

    x = experiment.walkers.x[:,0]
    y = experiment.walkers.x[:,1]

    circles, = pl.plot(x,y,'.',ls='None',markersize=1)
    pl.plot(x,y,'.',ls='None',markersize=8,c='w')

    def anim(i):
        experiment.step()
        x = experiment.walkers.x[:,0]
        y = experiment.walkers.x[:,1]

        circles.set_xdata(x)
        circles.set_ydata(y)
        return circles,

    ani = animation.FuncAnimation(fig, anim, interval=1, blit=True, save_count=50)
    pl.show()


if __name__=="__main__":

    N = 50_000
    dt = 0.01
    #force_field = lambda x: -(x-0.5).dot(np.array([[1,1],[-1,1]])) / (np.sqrt(((x)**2).sum(axis=1)))[:,None] / 10
    drift_field = lambda x: -(x-0.5).dot(0.5*np.array([[.1,1],[-1,.1]])) / (np.sqrt(((x-0.5)**2).sum(axis=1)))[:,None] /3
    #print(force_field(np.array([[1,1.0]])))
    walkers = Walkers2D(N,
                        dt,
                        box=[0,1,0,1],
                        initial_positions='boundary',
                        position_noise_coefficient=0.0001,
                        velocity_noise_coefficient=.0001,
                        #force_field=force_field,
                        drift_field=drift_field,
                        )
    #pl.plot(walkers.x[:,0], walkers.x[:,1],'.',markersize=1)
    #pl.axis('square')
    exp = Experiment(walkers,average_walker_distance_factor_for_radius=.4)
    #exp.step()
    animate(exp)

