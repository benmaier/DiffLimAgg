import numpy as np
from fast_histogram import histogram2d
from tqdm import tqdm
import matplotlib.pyplot as pl

def _compute_box_counting_dimension(nums_bins, covered_boxes):
    parms = np.polyfit(np.log(nums_bins), np.log(covered_boxes), 1)
    return parms

def compute_box_counting_dimension(x,y,verbose=True,return_all_fit_params=False):
    nums_bins, covered_boxes = compute_box_counting(x,y,verbose)
    parms = _compute_box_counting_dimension(num_bins, covered_boxes)
    if return_all_fit_params:
        return parms
    else:
        return parms[0]

def box_counting_analysis(x,y,verbose=True,ax=None):

    if ax is None:
        fig, ax = pl.subplots(1,1)

    nums_bins, covered_boxes = compute_box_counting(x,y,verbose)
    parms = _compute_box_counting_dimension(nums_bins, covered_boxes)

    ax.plot(nums_bins, covered_boxes,'s',mfc='None',markersize=6,label='measured')
    ax.plot(nums_bins, np.exp(np.polyval(parms, np.log(nums_bins))),':k',label=f'dim. = {parms[0]:3.2f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of boxes per side')
    ax.set_ylabel('number of covered boxes')
    ax.legend()

    return parms[0], ax, nums_bins, covered_boxes

def compute_box_counting(x,y,verbose=True):
    Ndata = len(x)
    assert(Ndata==len(y))
    extent = [ (x.min(), x.max()),
               (y.min(), y.max())
              ]
    box_lengths = [ np.diff(dim) for dim in extent ]
    maxL = np.max(box_lengths)
    extent = [ (dim[0], dim[0]+maxL) for dim in extent ]

    surface_area = np.prod([np.diff(dim) for dim in extent])
    ideal_gas_diam = 2*np.sqrt(surface_area/Ndata)/np.pi

    max_nbins = maxL / (ideal_gas_diam*5)
    covered_boxes = []
    nums_bins = np.arange(4, max_nbins)
    side_length = maxL / nums_bins

    if verbose:
        it = tqdm(nums_bins)
    else:
        it = nums_bins

    for nbins in it:
        hist = histogram2d(x,y,int(nbins),extent)
        ncovered = np.count_nonzero(hist.ravel())
        covered_boxes.append(ncovered)

    return nums_bins, np.array(covered_boxes)



if __name__ == "__main__":


    # grid, comes out as ~2.00
    x = np.arange(100)
    y = np.arange(100)
    x, y = np.meshgrid(x,y)
    x = x.ravel()
    y = y.ravel()

    # line, comes out as ~1.00
    x = np.arange(10000)
    y = np.zeros_like(x)

    # lattice ring, comes out as ~1.07
    n = 20000
    theta = np.linspace(2*np.pi/n,2*np.pi,n)
    x = np.cos(theta)
    y = np.sin(theta)

    # random ring, comes out as ~1.06
    theta = 2*np.pi *np.random.rand(20000)
    x = np.cos(theta)
    y = np.sin(theta)

    # random uniform, comes out as  ~2.00
    x = np.random.rand(20000)
    y = np.random.rand(20000)
    dim, ax,_, __ = box_counting_analysis(x,y)


    # random Erlang
    from scipy.stats import gamma
    theta = 2*np.pi *np.random.rand(20000)
    r = gamma.rvs(2, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    pl.show()
