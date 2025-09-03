from __future__ import division, print_function; __metaclass__ = type
import os, sys, math, itertools
import numpy as np
#import west
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper, VectorizingFuncBinMapper

import logging
log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

binbounds = [0.0,1.00] + [1.10+0.1*i for i in range(35)] + [4.60+0.2*i for i in range(10)] + [6.60+0.6*i for i in range(6)] + [float('inf')]

def func(coord):
#    print(coord) # coord[0]: RMSD wrt folded
                 # coord[1]: optimized bin allocation (separately calculated) 
    #print(int(coord[1]))
    if coord[0] < binbounds[1]:
        return len(binbounds)-2
    elif coord[0] > binbounds[-2]:
        return 0 
    else: 
        return int(coord[1])

class System(WESTSystem):
    def initialize(self):
        self.pcoord_ndim = 2 # RMSD and bin assignment 
        self.pcoord_len = 2
        self.pcoord_dtype = np.float32

        # Initial bin boundaries from unoptimized WE 
        #binbounds = [0.0,1.00] + [1.10+0.1*i for i in range(35)] + [4.60+0.2*i for i in range(10)] + [6.60+0.6*i for i in range(6)] + [float('inf')]
        self.nbins = len(binbounds) - 1
        self.bin_mapper = VectorizingFuncBinMapper(func, self.nbins)

        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), int)
        self.bin_target_counts[...] = 4 # allocation 


def coord_loader(fieldname, coord_file, segment, single_point=False):
    coord_raw = np.loadtxt(coord_file, dtype=np.float32)

    npts = len(coord_raw)
    assert coord_raw.shape[1] % 3 == 0
    ngrps = coord_raw.shape[1] // 3

    coords = np.empty((ngrps, npts, 3), np.float32)
    for igroup in range(ngrps):
        for idim in range(3):
            coords[igroup,:,idim] = coord_raw[:,igroup*3+idim]
    # convert to Angstroms
    #coords *= 10

    segment.data[fieldname] = coords


