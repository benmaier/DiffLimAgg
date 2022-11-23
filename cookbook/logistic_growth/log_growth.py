import numpy as np
import matplotlib.pyplot as pl
from DiffLimAgg.walkers import Walkers2D
from DiffLimAgg.experiment import Experiment

from genLogGrowth.analysis import iterate_all_cost_functions, get_formatted_comparison_figure, _data_kwargs
from genLogGrowth.fitting import fit
from genLogGrowth.tools import scifmt


def _fit(time,cumulative_data,which_data='incidence',which_model='generalized',data_is_log=False):

    iwhich = 0
    ilog = 0
    fig, axs = get_formatted_comparison_figure(1)

    t = np.array(time)
    cum = np.array(cumulative_data)
    inc = np.diff(cum)/np.diff(t)
    t_inc = (t[1:]+t[:-1])/2
    rate = np.diff(np.log(inc)) / np.diff(t_inc)
    t_rate = (t_inc[1:]+t_inc[:-1])/2

    if which_data == 'cumulative':
        _data = cum
        _t = t
    else:
        _data = inc
        _t = t_inc
    if data_is_log:
        ndx = np.where(_data>0)[0]
        _t = _t[ndx]
        _data = np.log(_data[ndx])

    out, model, complement_model, rate_model, ymdl, cmdl, rmdl = \
            fit(_t,
                _data,
                which_data=which_data,
                which_model=which_model,
                data_is_log=data_is_log,
               )

    #print("model:", which_model)
    #for key, val in out.params.items():
    #    print("   ", key, val)
    base = iwhich*2 + ilog
    ax = axs[base*5:(base+1)*5]

    if which_data == 'cumulative':
        icums = [0,1]
        iincs = [2,3]
    else:
        icums = [2,3]
        iincs = [0,1]

    for icum in icums:
        ax[icum].plot(t, cum, **_data_kwargs)
    for iinc in iincs:
        ax[iinc].plot(t_inc, inc, **_data_kwargs)
    ax[-1].plot(t_rate, rate,**_data_kwargs)
    ax[0].plot(_t, model(_t, **out.params))
    ax[1].plot(_t, model(_t, **out.params))
    ax[2].plot(_t, complement_model(_t, **out.params))
    ax[3].plot(_t, complement_model(_t, **out.params))
    ax[0].set_title('red. χ² = {0}'.format(scifmt(out.redchi)))

    if data_is_log:
        ax[0].set_yscale('log')
        ax[2].set_yscale('log')
    else:
        ax[1].set_yscale('log')
        ax[3].set_yscale('log')
    ax[-1].plot(_t, rate_model(_t, **out.params))

    return fig, axs

if __name__=="__main__":

    N = 20_000
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
    (t, counts), _ = exp.simulate()

    pl.figure()
    pl.plot(t, counts)
    pl.figure()
    pl.plot(t, np.cumsum(counts))
    pl.figure()
    pl.plot(t[1:], np.diff(np.log(counts))/np.diff(t))
    #pl.plot(t[1:], np.diff(counts)/np.diff(t))
    t /= t[-1]
    cumcount =  np.cumsum(counts)
    cumcount /= cumcount[-1]

    #iterate_all_cost_functions(t[20:], cumcount[20:], 'generalized')
    #iterate_all_cost_functions(t, cumcount, 'richards')
    #pl.show()
    #iterate_all_cost_functions(t, cumcount, 'generalized')



    _fit(t, cumcount,data_is_log=False)
    _fit(t, cumcount, which_model='richards',data_is_log=False)


    pl.show()

