
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
from scipy.signal import butter, sosfilt, sosfreqz, ellip, sosfiltfilt, filtfilt, freqz
#from lmfit.lineshapes import gaussian2d
#from lmfit.models import Gaussian2dModel

import logging
logfile='debug_log.txt'
logging.basicConfig(level=logging.INFO,filename=logfile)
#logging.basicConfig(level=logging.DEBUG,filename=logfile)

from comb_xcor.ccfmatrix import planetrvshift

def ButterFilter_quiet(x,freq,type='highpass',order=5):
    """
    x - data
    freq - cutoff frequency/ies, where 1 corresponds to nyquist frequency (ie half the
    sampling rate)
    """
    freq = np.array(freq) #allows lists for bandpass arrays print "Assuming sampling of 1Hz:"
    #print(" Nyquist frequency: {} Hz (2 pixels)".format(0.5) )
    #print (" Min frequency: {:.7f} Hz ({} pixels, i.e. full  length)".format(1./x.shape[-1],x.shape[-1]))
    #print( " Cutoff frequency: {} Hz ({} pixels)".format(freq * 0.5,2./freq)) #create the filter 
    sos = butter(order, freq, analog=False, btype=type, output='sos')
    #apply the filter
    y = sosfiltfilt(sos, x)
    return y


'''encorporates _upgrade'''

Ms=1.989*10.0**33
Mj=1.8986*10.0**30

class SpectrumSet:
    def _read_flux_(self):
        if self.filename[-8:]=='dict.pic':
            self.byorder=True
            ob = open(self.loc, "rb")
            dat=pickle.load(ob)
            self.dates=dat['dates'].astype(np.float)
            self.data=dat['fluxes']
            wls=dat['wls']        
        
        elif self.filename[-3:]=='pic':
            ob = open(self.loc, "rb")
            d,dat0=pickle.load(ob)
            dat=dat0.transpose()
            self.dates=np.array(d[1:]).astype(np.float)
            self.data=dat[1:,:].astype(np.float)
            wls=dat[0]
        else:
            flux=pd.read_csv(self.loc,header=None)
            fluxes=flux.values
            #fluxes=np.loadtxt(filename,delimiter=',',dtype=str)
            self.dates=np.array(fluxes[0][1:])
            #fluxes=np.loadtxt(filename,delimiter=',',skiprows=1)
            data1=np.transpose(np.array(fluxes[1:]))
            self.data=data1[1:,:].astype(np.float)
            
            wls=data1[:][0]
        self.wls=wls.astype(np.float)     

    def _read_template_(self):
        if self.template_fn[-4:]=='fits':
            '''mantis templatess'''
            with fits.open(self.template_fn) as f:
                f0=f[0].data
            f1=f0[1]
            fl=f1[::-1]
            wn=np.arange(len(f0[1]))*0.01+4000
            wl=10000./wn[::-1]*1e4
        else:
            df=pd.read_csv(self.template_fn)
            data_df=np.transpose(df.values)
            wl=data_df[0]
            fl=data_df[1]
            
        if self.template_wl_unit!=None:
            self.template_wl_unit=astro_lf.find_unit(template_wl_unit)
        else:
            self.template_wl_unit=astro_lf.guess_unit(wl.flatten()[0])
        
        conversion_fac=astro_lf.wl_unit_choices[self.template_wl_unit].conversion/astro_lf.wl_unit_choices[self.spectrum_wl_unit].conversion
        wl_co_temp_0=wl*conversion_fac

        return wl_co_temp_0,fl
    

    def __init__(self,filename,folder,bad_dates=None,maskfile=None,template_fn='template_width0p2_CO.csv',subset='',badphases=[],
                 period=8.9891, scale=1.,day0=2453367.805,template_wl_unit=None,spectrum_wl_unit=None,wllims=[0.0,1e20],
                 subtractone=True,transit_midpoint=None,tp=None,printphases=False,oddeven='both'):
        #only need to worry abot day0, period if badphases!=[]
        #subbtractone added to deal with transmission spectra
        self.byorder=False
        self.idnum=filename[-12:-4]
        self.filename=filename
        self.folder=folder
        self.bad_date=bad_dates
        self.oddeven=oddeven
        self.maskfile=maskfile
        self.template_fn=template_fn
        self.subset=subset
        self.badphases=badphases
        self.period=period
        self.scale=scale
        self.day0=day0
        self.ccarrs=[]
        self.loc=folder+filename
        self.template_wl_unit=template_wl_unit
        self.wllims=wllims
        self._read_flux_()
        
        if tp==None:
            self.tp=day0
        else:
            self.tp=tp
        if transit_midpoint==None:
            self.transit_midpoint=day0
        else:
            self.transit_midpoint=transit_midpoint
        
        

            
        if spectrum_wl_unit!=None:
            self.spectrum_wl_unit=astro_lf.find_unit(spectrum_wl_unit)
        else:
            self.spectrum_wl_unit=astro_lf.guess_unit(self.wls.flatten()[0])
        

        if bad_dates==None:
            hjdall=self.dates
            bad_dates=[]
        else:			
            temp = pd.read_csv(folder+'hhjd'+subset+'.csv')
            temp2=np.transpose(temp.values)
            hjdall=temp2[1]
        
        

        if badphases!=[]:
            i=0
            
            while i<len(self.dates):
                ba=False
                logging.debug('day: ',hjdall[i])
   

                pha=((hjdall[i]-day0)/period) % 1.0 

                    
 
                if badphases[4](badphases[0](pha,badphases[1]), badphases[2](pha,badphases[3])):    
                    bad_dates.append(str(int(self.dates[i])))
                    bad_dates.append(str(float(self.dates[i])))
                    ba=True
                if len(badphases)>5:
                    if badphases[9](badphases[5](pha,badphases[6]), badphases[7](pha,badphases[8])):
                        bad_dates.append(str(int(self.dates[i])))
                        bad_dates.append(str(float(self.dates[i])))
                        ba=True
                if printphases:
                    print(hjdall[i],pha,'bad? ',ba)

                i=i+1  
                
        if self.oddeven!='both':
            ivals=np.arange(len(hjdall))
            if self.oddeven=='odd':
                bd=ivals%2==0
            elif self.oddeven=='even':
                bd=ivals%2==1
            for item in self.dates[bd]:

                bad_dates.append(str(item))
                bad_dates.append(float(item))
                



        idnum=filename[-12:-4]

        goodlocs=[]
        l=len(self.dates)
        i=0

        bad_dates=np.array(bad_dates).astype(np.float)
        #print(bad_dates)


        while i<l:
            if (float(self.dates[i]) not in bad_dates)==True:
                goodlocs.append(i)
            i=i+1    


        #print(goodlocs)

        medstar=np.median(self.data[goodlocs],0)
        meanstar=np.mean(self.data[goodlocs],0)
        l_good=len(goodlocs)



        badwls=np.array([])
        badwls_mask=np.zeros_like(self.wls)
        if maskfile!=None:
            starts,ends=read2cols(maskfile)
            for s,e in zip(starts,ends):
                arr=np.where((self.wls>s) & (self.wls<=e))

                badwls=np.append(badwls,arr)
                badwls_mask[arr]=1
            arr=np.where(self.wls<self.wllims[0])
            badwls=np.append(badwls,arr)
            badwls_mask[arr]=1
            arr=np.where(self.wls>self.wllims[1])
            badwls=np.append(badwls,arr)        
            badwls_mask[arr]=1
            badwls=badwls.astype(np.int)

        else:
            badwls=[]


        gooddat=self.data[goodlocs]


        hjd=hjdall[goodlocs] 
        self.s2narr=np.ones_like(hjd)

        #read in template file

        wl_co_temp_0, fl_co_temp=self._read_template_()
        
        ccarr=[]
        cc2arr=[]
        sigarr=[]
        sig2arr=[] 
        
        logging.debug('wl_template:',wl_co_temp_0[0:10],wl_co_temp_0[-10:])
        logging.debug('wls:',self.wls[0:10],self.wls[-10:])
        if self.byorder:
            fl_co=np.zeros_like(self.wls)
            for o, o_w in enumerate(self.wls):
                if subtractone:
                    fl_co[o]=spectres.spectres(o_w, wl_co_temp_0, fl_co_temp,verbose=False,fill=1.)-1.0
                else:
                    fl_co[o]=spectres.spectres(o_w, wl_co_temp_0, fl_co_temp,verbose=False)
        else:
        
            if subtractone:
                fl_co=spectres.spectres(self.wls, wl_co_temp_0, fl_co_temp,verbose=False,fill=1.)-1.0
            else:
                fl_co=spectres.spectres(self.wls, wl_co_temp_0, fl_co_temp,verbose=False)
        
        if self.byorder:
            fl_co[badwls_mask==1]=0
        else:
            fl_co[badwls]=0
        fl_co=fl_co*scale

        plot=1
        self.hjd=hjd
        self.badwls=badwls
        self.fl_co=np.ma.masked_equal(fl_co,0)
        self.gooddata=np.ma.masked_equal(gooddat,0)
        self.phases=((self.hjd-day0)/period) % 1.0 
        self.phases[self.phases>.5]=self.phases[self.phases>.5]-1.
        self.badwls_mask=badwls_mask
        #good data
        
    def calculate_s2ns(self):
        s2ns=[]
        i=0
        for flux in self.gooddata:

            fors2n=np.where((self.wls <2.316))        
            a=flux[fors2n]           
            #select a region that hass generally been corrected well for tellurics

            s2ntem=findbests2n(flux[fors2n]+1,edge=5,p=95)
            #find s2n of that region

            s2ns.append(s2ntem)
            i=i+1

        s2narr=np.array(s2ns)
        self.s2narr=s2narr
        return s2narr
                
    
    def filtertemplate(self,flip=False,butter_freq=False,butter_order=5,resample=1):
        '''resample is the ratio between the old resolution and the new resolution'''
        if flip:
            self.fl_co=-self.fl_co
        if butter_freq!=False:
            self.fl_co=ButterFilter_quiet(self.fl_co,butter_freq,order=butter_order)
            
    def plottemplate(self,wlrange=None):

        if wlrange!=None:
            loc=((self.wls>=wlrange[0]) & (self.wls<=wlrange[1]))
            wlplot=self.wls[loc]
            datplot=self.fl_co[:,loc]
        else:
            wlplot=self.wls
            datplot=self.fl_co
        fig=plt.figure(figsize=(8,5))
        axarr = fig.add_subplot(1,1,1)        
        x=axarr.plot(wlplot,datplot)
      
        
        axarr.set_ylabel('flux')
        axarr.set_xlabel('wavelength (microns)')
    

    def plotspecs(self,wlrange=None,orbitalpars=None,centwl=None,code='vcurve',vsys=0,lcolor='white',nontransitingphases=False):
        
        if nontransitingphases:
            self.phases[self.phases<0]=self.phases[self.phases<0]+1
        
        if wlrange!=None:
            loc=((self.wls>=wlrange[0]) & (self.wls<=wlrange[1]))
            wlplot=self.wls[loc]
            datplot=self.gooddata[:,loc]
        else:
            wlplot=self.wls
            datplot=self.gooddata
        fig=plt.figure(figsize=(8,5))
        axarr = fig.add_subplot(1,1,1)        
        x=axarr.contourf(wlplot,self.phases,datplot)
        
        if centwl!=None:
            rvshifts=-np.array([planetrvshift(date,orbitalpars,day0=self.day0,code=code)-vsys for date in self.hjd])
            wlline=astro_lf.veltodeltawl(rvshifts,centwl)+centwl

            axarr.plot(wlline,self.phases,color=lcolor,lw=2,zorder=40)
        
        
        axarr.set_ylabel('phase')
        axarr.set_xlabel('wavelength (microns)')
        temp=fig.colorbar(x)
        
        temp.ax.set_ylabel(r'relative flux')
        
    def maskwavelengths(self,wlcent,wlwid):
        '''wlcent, wlwid in microns'''
        loc=((self.wls<=(wlcent+wlwid)) & (self.wls>=(wlcent-wlwid)))
        self.fl_co=np.ma.masked_where(loc,self.fl_co)
        self.gooddata[:,loc]=0.
        self.gooddata=np.ma.masked_equal(self.gooddata,0)
        

    
    def system_properties(self, k_s,M_star,M_planet=.0001):
        '''M_star is stellar mass in solar masses
        k_s is stellar velocity amplitude in km/s
        M_planet is planet mass in Jupiter masses'''
        self.k_s=k_s
        self.M_star=M_star
        self.M_cgs=M_star*Ms
        self.M_planet=M_planet
        self.M_planet_cgs=M_planet*Mj
        
    def printorders(self):
        if self.byorder:
            for o,w in enumerate(self.wls):
                print('order ',o,' with limits of ',np.min(w),' and ',np.max(w))
        else:
            print('not in echelle format')
