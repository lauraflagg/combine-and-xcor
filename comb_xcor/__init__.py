
import sys
from stats_lf import xcor, xcor_fast, chisq, log_likelihood_zucker, gaussian2D
import numpy as np
from astro_lf import findbests2n, vel2wl, c_kms,wl2vel,veltodeltawl, getorbitpars
from readwrite_lf import read2cols
from PyAstronomy import pyasl
from scipy import stats
import matplotlib.pyplot as plt
import v_curve
import spectres
import pandas as pd
import pickle
import time
import astro_lf
import radvel_rv
from astropy.io import fits

from matplotlib import patches
import matplotlib.colors as mpl
from lmfit import Model
#from lmfit.lineshapes import gaussian2d
#from lmfit.models import Gaussian2dModel

import logging
logfile='debug_log.txt'
logging.basicConfig(level=logging.INFO,filename=logfile)
#logging.basicConfig(level=logging.DEBUG,filename=logfile)


'''encorporates _upgrade'''

Ms=1.989*10.0**33
Mj=1.8986*10.0**30

from comb_xcor.spectrumset import SpectrumSet 
from comb_xcor.ccfmatrix import CCFMatrix
from comb_xcor.ccfmatrix import planetrvshift
