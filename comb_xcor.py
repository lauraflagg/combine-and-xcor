import sys
sys.path.append('C:/Users/laura/programming/python/toimport')
from stats_lf import xcor, chisq, log_likelihood_zucker, gaussian2D
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
#in cgs, mass of sun and jupiter





def stellarrvshift(hhjd,par,day0,code='vcurve'):
    '''code = vcurve or sin or doublesin
    if code is vcurve, par has 6 items: center of mass velocity, period, asini in GM, eccentricity, argument of periastron, and phase offset
    if code is sin then par has 4 items, A*sin(x*(t-t0)+p)
    par is orbital parameters'''
    if code=='vcurve':
        per=par[1]
        pha=((hhjd-day0)/per) % 1.0
    
    if code=='vcurve' or code=='v_curve' or code=='v-curve':        
        shift_ms=v_curve.citau(pha,par)
        #inm/s
        shift=shift_ms/1000.
    elif code=='sin'or code=='sine':
        if len(par)<4:
            par3=0
        else:
            par3=par[3]
        shift=par[0]*np.sin(par[1]*(hhjd-par[2])+par3)
    elif code=='doublesin':
        shift=par[0]*np.sin(par[1]*(hhjd-par[2])+par[3])+par[4]*np.sin(par[5]*(hhjd-par[2])+par[6])
    
    
    return shift

def planetrvshift(hhjd,par,day0,code='vcurve'):
    '''code = vcurve or sin or doublesin or radvel
    if code is vcurve, par has 6 items: center of mass velocity, period, asini in GM, eccentricity, argument of periastron, and phase offset
    if code is sin then par has 4 items, A*sin(freq*(t-t0))
    par is orbital parameters'''
    if code=='vcurve' or code=='v_curve' or code=='v-curve': 
        per=par[1]
        pha=((hhjd-day0)/per) % 1.0
    
    if code=='vcurve' or code=='v_curve' or code=='v-curve':        
        shift_ms=v_curve.citau(pha,par)
        #inm/s
        shift=shift_ms/1000.
    elif code=='sin'or code=='sine':
        shift=par[2]*np.sin(par[0]*(hhjd-par[1]))
    elif code=='radvel':
        shift=radvel_rv.rv_drive(np.array([hhjd]),par)
    return shift    


class SpectrumSet:
    def _read_flux_(self):
        if self.filename[-3:]=='pic':
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
    
    def __init__(self,filename,folder,bad_dates=None,maskfile=None,template_fn='template_width0p2_CO.csv',subset='',badphases=[],
                 period=8.9891, scale=1.,day0=2453367.805,template_wl_unit=None,spectrum_wl_unit=None,wllims=[0.0,1e20],subtractone=True,transit_midpoint=None,tp=None,printphases=False):
        #only need to worry abot day0, period if badphases!=[]
        #subbtractone added to deal with transmission spectra
        self.idnum=filename[-12:-4]
        self.filename=filename
        self.folder=folder
        self.bad_date=bad_dates
        self.maskfile=maskfile
        self.template_fn=template_fn
        self.subset=subset
        self.badphases=badphases
        self.period=period
        self.scale=scale
        self.day0=day0
        self.ccarrs=[]
        self.loc=folder+filename
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
            self.spectrum_wl_unit=astro_lf.guess_unit(self.wls[0])
        

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
        starts,ends=read2cols(maskfile)
        for s,e in zip(starts,ends):
            arr=np.where((self.wls>s) & (self.wls<=e))
            badwls=np.append(badwls,arr)
        arr=np.where(self.wls<self.wllims[0])
        badwls=np.append(badwls,arr)
        arr=np.where(self.wls>self.wllims[1])
        badwls=np.append(badwls,arr)        

        badwls=badwls.astype(np.int)    


        gooddat=self.data[goodlocs]


        hjd=hjdall[goodlocs]   

        #read in template file

        df=pd.read_csv(template_fn)
        data_df=np.transpose(df.values)
        
        if template_wl_unit!=None:
            self.template_wl_unit=astro_lf.find_unit(template_wl_unit)
        else:
            self.template_wl_unit=astro_lf.guess_unit(data_df[0][0])
        
        conversion_fac=astro_lf.wl_unit_choices[self.template_wl_unit].conversion/astro_lf.wl_unit_choices[self.spectrum_wl_unit].conversion
        wl_co_temp_0=data_df[0]*conversion_fac

        fl_co_temp=data_df[1]
        ccarr=[]
        cc2arr=[]
        sigarr=[]
        sig2arr=[] 
        
        logging.debug('wl_template:',wl_co_temp_0[0:10],wl_co_temp_0[-10:])
        logging.debug('wls:',self.wls[0:10],self.wls[-10:])
        if subtractone:
            fl_co=spectres.spectres(self.wls, wl_co_temp_0, fl_co_temp,verbose=False,fill=1.)-1.0
        else:
            fl_co=spectres.spectres(self.wls, wl_co_temp_0, fl_co_temp,verbose=False)
        
        
        fl_co[badwls]=0
        fl_co=fl_co*scale

        plot=1
        self.hjd=hjd
        self.badwls=badwls
        self.fl_co=np.ma.masked_equal(fl_co,0)
        self.gooddata=np.ma.masked_equal(gooddat,0)
        self.phases=((self.hjd-day0)/period) % 1.0 
        self.phases[self.phases>.5]=self.phases[self.phases>.5]-1.
        #good data


    def plotspecs(self,wlrange=None,orbitalpars=None,centwl=None,code='vcurve',vsys=0,lcolor='white'):
        
        if wlrange!=None:
            loc=((self.wls>=wlrange[0]) & (self.wls<=wlrange[1]))
            wlplot=self.wls[loc]
            datplot=self.gooddata[:,loc]
        else:
            wlplot=self.wls
            datplot=self.gooddat
        fig=plt.figure(figsize=(8,5))
        axarr = fig.add_subplot(1,1,1)        
        x=axarr.contourf(wlplot,self.phases,datplot)
        
        if centwl!=None:
            rvshifts=np.array([planetrvshift(date,orbitalpars,day0=self.day0,code=code)+vsys for date in self.hjd])
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
        self.gooddata=np.ma.masked_where(loc,self.goodata)


    
    def system_properties(self, k_s,M_star,M_planet=.0001):
        '''M_star is stellar mass in solar masses
        k_s is stellar velocity amplitude in km/s
        M_planet is planet mass in Jupiter masses'''
        self.k_s=k_s
        self.M_star=M_star
        self.M_cgs=M_star*Ms
        self.M_planet=M_planet
        self.M_planet_cgs=M_planet*Mj


        
class CCFMatrix():
    def __init__(self,velocities=np.arange(0, 110),low=-50,up=50,method='median',func=xcor,order='cx',disp=2.04,fromfile=''):
        self.func=func
        self.allccs=[]
        self.allcc_sigs=[]
        #can't pickle a func so must be initizialized every time
        if fromfile!='':
            self.readfile(fromfile)       
        
        else:    
            self.disp=disp
            self.velocities=velocities
            self.low=low
            self.up=up
            self.method=method
            self.order=order

            self.rvs=np.arange(low,up+1,1)*disp          

    def creatematrix(self, spectrumset,orbitalpars,indstart=0,indend=-1,day0=2453367.8,write=True,code='vcurve'):

        self.orbitalpars=orbitalpars
        self.idnum=spectrumset.idnum
        self.day0=day0


        #indend and indstart is if you want to subset a section of consecutive dates  
        
        if indend==-1:
            indend=len(spectrumset.hjd)
    
    
        ccarr=[]
        cc2arr=[]
        sigarr=[]
        sig2arr=[]    
        s2ns=[]

        maxkp=np.max(self.velocities)
    
        l_good=len(spectrumset.hjd)
    
        starrvs=[]
        
        if code=='sin'or code=='sine':
            pars2use=np.zeros(3)
            pars2use[2]=maxkp
            pars2use[0:2]=orbitalpars[0:2]
        else:
            #print(orbitalpars)
            pars2use=orbitalpars.copy()
            maxvtheory=getorbitpars(m_s=spectrumset.M_star,period=orbitalpars[1],m_p=spectrumset.M_planet)['v_p']#np.pi*2*orbitalpars[2]*constants.au.value/(orbitalpars[1]*sec_in_day)*1e-3
            #print(maxkp,maxvtheory)
            pars2use[2]=maxkp/maxvtheory.value*orbitalpars[2]
            #print(pars2use)
        maxplanetrvs=np.array([planetrvshift(date,pars2use,day0=self.day0,code=code) for date in spectrumset.hjd])
        
        for date in spectrumset.hjd:
            #starrvs.append(stellarrvshift(date,orbitalpars,day0=self.day0,code=code))
            starrvs=maxplanetrvs
        #print(maxplanetrvs)
    
    
        if self.method=='weighted':
            i=0
            while i<l_good:
                flux=spectrumset.gooddata[i]
    
                fors2n=np.where((spectrumset.wls <2.316))        
                a=flux[fors2n]           
                #select a region that hass generally been corrected well for tellurics
    
                s2ntem=findbests2n(flux[fors2n]+1,edge=5,p=95)
                #find s2n of that region
    
                s2ns.append(s2ntem)
                i=i+1
    
            s2narr=np.array(s2ns)
    
    
        #print('rvlen',len(rvs))
        #print(rvs)
        #
    
        for vtemp in self.velocities:
            if vtemp==0:
                vmax=0.001
            else:
                vmax=vtemp
            m_p=spectrumset.M_cgs*spectrumset.k_s/vmax
            #m_p is in grams, because Mc is in grams; vmax, k_s are both in km/s
    
            massratio=vmax/spectrumset.k_s
            Jupmasses=m_p/Mj
    
            #print 'v=',vmax
            specs=[]
            #the mass of the planet in cgs
    
            planetfluxes=np.zeros((l_good,len(spectrumset.wls)))
    
    
            i=0
            while i<l_good:
                flux=spectrumset.gooddata[i]
    
                flux[spectrumset.badwls]=0    
    
                starrv=starrvs[i]
                #in km/s
    
                #momentum of star=
                p=spectrumset.M_cgs*starrv
    
                #angular momentum is conserved and seperation is the same so M_p*v_p=-M_star*v_star
                #v_p=-p/m_p
                v_p=-starrv*massratio
                v_p=maxplanetrvs[i]*vtemp/maxkp
                #print(v_p)
                #velocity of planet in km/s
    
                nf_1, wl_1 = pyasl.dopplerShift(spectrumset.wls*10000, flux, -v_p, edgeHandling='firstlast', fillValue=None)
    
                specs.append(nf_1)
                #since I'm using the shifted flux, the wavelengths stay the same so I don't have to interpolate again
    

                i=i+1
    
            specsarr=np.array(specs)
    
            #trying something diff
            #
            #OR
    
            if self.order=='cx':
    
                if self.method=='median':
    
                    medspec=np.median(specsarr[indstart:indend],0)
                elif self.method=='mean':
                    medspec=np.mean(specsarr[indstart:indend],0)
    
                elif self.method=='weighted':
                    medspec=np.average(specsarr[indstart:indend],0,weights=s2narr[0:])
                else:
                    medspec=stats.trim_mean(specsarr[indstart:indend], 0.05,axis=0)
    
                #never mind, try replacing nan with 0
                medspec=np.nan_to_num(medspec)
    
    
    
    
                cc=self.func(medspec,spectrumset.fl_co,self.low,self.up)
                le=int((len(cc)-1)/2)
    
                cc=np.array(cc)
    
    
    
                ccarr.append(cc)
    
    
            else:
                ccs=[]
                for spectrum in specsarr:
    
                    sp=np.nan_to_num(spectrum)
                    cc_temp=self.func(sp,spectrumset.fl_co,self.low,self.up)
                    ccs.append(np.nan_to_num(cc_temp))
                
                self.ccfs_shifted=ccs
    
                if self.method=='median':
    
                    medspec=np.median(ccs[indstart:indend],0)
                elif self.method=='mean':
                    medspec=np.mean(ccs[indstart:indend],0)
    
                elif self.method=='weighted':
                    medspec=np.average(ccs[indstart:indend],0,weights=s2narr[0:])
                else:
                    medspec=stats.trim_mean(ccs[indstart:indend], 0.05,axis=0)
                    
    
                ccarr.append(medspec)
    
    
        self.ccarr=np.array(ccarr)
        if write:
            self.writefile()
        self.allccs.append(self.ccarr)
    def createcoadd(self, spectrumset,orbitalpars,indstart=0,indend=-1,day0=2453367.8,write=True,code='vcurve'):
        self.orbitalpars=orbitalpars
        self.idnum=spectrumset.idnum
        self.day0=day0
        self.wls=spectrumset.wls


        #indend and indstart is if you want to subset a section of consecutive dates  
        
        if indend==-1:
            indend=len(spectrumset.hjd)
    
    
        ccarr=[]
        cc2arr=[]
        sigarr=[]
        sig2arr=[]    
        s2ns=[]
        meds=[]
        maxkp=np.max(self.velocities)
    
        l_good=len(spectrumset.hjd)
    
        starrvs=[]
        
        if code=='sin'or code=='sine':
            pars2use=np.zeros(3)
            pars2use[2]=maxkp
            pars2use[0:2]=orbitalpars[0:2]
        elif code=='radvel':
            pars2use=np.zeros(5)
            pars2use[4]=maxkp
            pars2use[0:4]=orbitalpars[0:4]        
        else:
            #print(orbitalpars)
            pars2use=orbitalpars.copy()
            maxvtheory=getorbitpars(m_s=spectrumset.M_star,period=orbitalpars[1],m_p=spectrumset.M_planet)['v_p']#np.pi*2*orbitalpars[2]*constants.au.value/(orbitalpars[1]*sec_in_day)*1e-3
            #print(maxkp,maxvtheory)
            pars2use[2]=maxkp/maxvtheory.value*orbitalpars[2]
            #print(pars2use)
        maxplanetrvs=np.array([planetrvshift(date,pars2use,day0=self.day0,code=code) for date in spectrumset.hjd])
        
        for date in spectrumset.hjd:
            #starrvs.append(stellarrvshift(date,orbitalpars,day0=self.day0,code=code))
            starrvs=maxplanetrvs
        #print(maxplanetrvs)    
    

    
        if self.method=='weighted':
            i=0
            while i<l_good:
                flux=spectrumset.gooddata[i]
    
                fors2n=np.where((spectrumset.wls <2.316))        
                a=flux[fors2n]           
                #select a region that hass generally been corrected well for tellurics
    
                s2ntem=findbests2n(flux[fors2n]+1,edge=5,p=95)
                #find s2n of that region
    
                s2ns.append(s2ntem)
                i=i+1
    
            s2narr=np.array(s2ns)
    
    
        #print('rvlen',len(rvs))
        #print(rvs)
        #
    
        for vtemp in self.velocities:
            if vtemp==0:
                vmax=0.001
            else:
                vmax=vtemp
            m_p=spectrumset.M_cgs*spectrumset.k_s/vmax
            #m_p is in grams, because Mc is in grams; vmax, k_s are both in km/s
    
            massratio=vmax/spectrumset.k_s
            Jupmasses=m_p/Mj
    
            #print 'v=',vmax
            specs=[]
            #the mass of the planet in cgs
    
            planetfluxes=np.zeros((l_good,len(spectrumset.wls)))
    
    
            i=0
            while i<l_good:
                flux=spectrumset.gooddata[i]
    
                flux[spectrumset.badwls]=0    
    
                starrv=starrvs[i]
                #in km/s
    
                #momentum of star=
                p=spectrumset.M_cgs*starrv
    
                #angular momentum is conserved and seperation is the same so M_p*v_p=-M_star*v_star
                #v_p=-p/m_p
                v_p=-starrv*massratio
                v_p=maxplanetrvs[i]*vtemp/maxkp
                #velocity of planet in km/s
    
                nf_1, wl_1 = pyasl.dopplerShift(spectrumset.wls*10000, flux, -v_p, edgeHandling='firstlast', fillValue=None)
    
                specs.append(nf_1)
                #since I'm using the shifted flux, the wavelengths stay the same so I don't have to interpolate again
    

                i=i+1
    
            specsarr=np.array(specs)
    
            #trying something diff
            #
            #OR
    

    
            if self.method=='median':

                medspec=np.median(specsarr[indstart:indend],0)
            elif self.method=='mean':
                medspec=np.mean(specsarr[indstart:indend],0)

            elif self.method=='weighted':
                medspec=np.average(specsarr[indstart:indend],0,weights=s2narr[0:])
            else:
                medspec=stats.trim_mean(specsarr[indstart:indend], 0.05,axis=0)

                #never mind, try replacing nan with 0
            medspec=np.nan_to_num(medspec)
            meds.append(medspec)
        self.coadds=np.array(meds)
    
    


    def createccvsphase(self, spectrumset,indstart=0,indend=-1,write=False):
        
        self.idnum=spectrumset.idnum

        #indend and indstart is if you want to subset a section of consecutive dates  
        
        if indend==-1:
            indend=len(spectrumset.hjd)
    
    
        ccarr=[]
        cc2arr=[]
        sigarr=[]
        sig2arr=[]    
        s2ns=[]
    
        l_good=len(spectrumset.hjd)

    
    
        if self.method=='weighted':
            i=0
            while i<l_good:
                flux=spectrumset.gooddata[i]
    
                fors2n=np.where((spectrumset.wls <2.316))        
                a=flux[fors2n]           
                #select a region that hass generally been corrected well for tellurics
    
                s2ntem=findbests2n(flux[fors2n]+1,edge=5,p=95)
                #find s2n of that region
    
                s2ns.append(s2ntem)
                i=i+1
    
            s2narr=np.array(s2ns)
 

        specs=[]

        planetfluxes=np.zeros((l_good,len(spectrumset.wls)))


        i=0
        while i<l_good:
            flux=spectrumset.gooddata[i]

            flux[spectrumset.badwls]=0    


            specs.append(flux)
            #since I'm using the shifted flux, the wavelengths stay the same so I don't have to interpolate again


            i=i+1

        specsarr=np.array(specs)
        
        ccs=[]
        for spectrum in specsarr:
            sp=np.nan_to_num(spectrum)
            cc_temp=self.func(sp,spectrumset.fl_co,self.low,self.up)
            ccs.append(np.nan_to_num(cc_temp))
    
        self.ccvsphase=np.array(ccs)

    
    def writefile(self):
        self.ccf_filename='ccfs/'+self.idnum+'_ccfarr.pickle'
        f = open(self.ccf_filename,'wb') 
        pickle.dump((self.rvs,self.velocities,self.ccarr,self.low,self.up,self.method,self.order,self.disp),f)
        #pickle.dump((dist_poisson_full,dist_cdf_full,dist_cdf_add),f)
        f.close()  
        
        with open('log.txt', 'a') as fd: 
            now = time.strftime("%c")
            nstr='comb_xcor_class.py  '+now+'  ID: '+self.idnum+'   func: '+str(self.func)+'  method:'+self.method 
            fd.write(nstr)        
    
    def readfile(self,ccf_filename):
        self.ccf_filename=ccf_filename
        f = open(self.ccf_filename,'rb') 
        self.rvs,self.velocities,self.ccarr,self.low,self.up,self.method,self.order,self.disp=pickle.load(f)
        f.close()        
    
    def plotcoaddmatrix(self,wlrange=None):
        if wlrange!=None:
            loc=((self.wls>=wlrange[0]) & (self.wls<=wlrange[1]))
            wlplot=self.wls[loc]
            coaddplot=self.coadds[:,loc]
        else:
            wlplot=self.wls
            coaddplot=self.coadds
        fig=plt.figure(figsize=(8,5))
        axarr = fig.add_subplot(1,1,1)        
        x=axarr.contourf(wlplot,self.velocities,coaddplot)
        axarr.set_ylabel('k$_p$ (km/s)')
        axarr.set_xlabel('wavelength (microns)')
        temp=fig.colorbar(x)
        temp.ax.set_ylabel(r'relative flux')        
   
    
    def plotccfvsphase(self,spectrumset, orbitalpars,rv_lim=100,levels=10,cm='viridis',kp_val=None,showplot=True,lcolor='white',lalpha=0.5,phasestart=0,phaseend=0,day0=2453367.8,code='vcurve',per='3.',vsys=0.):

        phases=spectrumset.phases
        #print(phases)
        fig=plt.figure(figsize=(8,5))
        axarr = fig.add_subplot(1,1,1)       
        x=axarr.contourf(np.round(self.rvs,1),phases,self.ccvsphase,levels,cmap=cm,zorder=0)
        temp=fig.colorbar(x)
        if self.func==xcor:
            temp.ax.set_ylabel(r'CCF height')
        else:
            temp.ax.set_ylabel(r'$\chi^{2}$')        
        if kp_val!=None:
            massratio=kp_val/spectrumset.k_s
            rvshift=np.array([planetrvshift(date,orbitalpars,day0=day0,code=code)+vsys for date in spectrumset.hjd])

            axarr.plot(rvshift[phases<phasestart],phases[phases<phasestart],color=lcolor,alpha=lalpha,lw=2,zorder=40)
            axarr.plot(rvshift[phases>phaseend],phases[phases>phaseend],color=lcolor,alpha=lalpha)
        axarr.set_ylim(np.min(phases),np.max(phases))
        axarr.set_xlim(-rv_lim,rv_lim)
        axarr.set_xlabel(r'systemic velocity (km s$^{-1}$)',fontsize=16)
        axarr.set_ylabel(r'phase',fontsize=16)
        plt.tight_layout()
        if showplot:
            plt.show() 
        plt.close()
        return fig,axarr        
            
    def findcenter(self,xguess,yguess,xlims=None,ylims=None,xcenlims=None,ycenlims=None,negamp=False):
        if xlims!=None:
            yesrv=np.where((self.rvs<=xlims[1]) & (self.rvs>=xlims[0]))
        else:
            yesrv=np.where(self.rvs<=1e6)
            
        if ylims!=None:
            yesvel=np.where((self.velocities<=ylims[1]) & (self.velocities>=ylims[0]))
        else:
            yesvel=np.where(self.velocities<=1e6)        
        
        temp2=self.ccarr[yesvel[0]]
        ccarr_plot=temp2[:,yesrv[0]]
        rvs_plot=self.rvs[yesrv]
        velocities_plot=self.velocities[yesvel]
        #print(ccarr_plot.shape,rvs_plot.shape,velocities_plot.shape)
        
        mod = Model(gaussian2D, independent_vars=['x', 'y'])
        params = mod.make_params()
        if xcenlims==None:
            params['xcen'].set(xguess,min=np.min(rvs_plot),max=np.max(rvs_plot))
        else:
            params['xcen'].set(xguess,min=xcenlims[0],max=xcenlims[1])
        if ycenlims==None:
            params['ycen'].set(yguess,min=np.min(velocities_plot),max=np.max(velocities_plot))
        else:
            params['ycen'].set(yguess,min=ycenlims[0],max=ycenlims[1])
            
        if negamp:
            params['amp'].set(np.min(ccarr_plot),max=0)
        else:
            params['amp'].set(np.max(ccarr_plot),min=0)
        params['xwid'].set(1,min=0)
        params['ywid'].set(2,min=0)
        
        #print(params)
        result = mod.fit(ccarr_plot, x=rvs_plot, y=velocities_plot, xcen=params['xcen'],ycen=params['ycen'],bas=0,amp=params['amp'],xwid=params['xwid'],ywid=params['ywid'])
        #print(result.fit_report())
        print(result.best_values)
        return result.best_values['xcen'],result.best_values['ycen']
    
    def plotccf(self,levels=10,cm='ocean',rv_line=None,kp_line=None,showplot=True,lcolor='gray',lalpha=0.5):
        fig=plt.figure(figsize=(16,5))
        axarr = fig.add_subplot(1,1,1)       
        x=axarr.contourf(np.round(self.rvs,1),self.velocities,self.ccarr,levels,cmap=cm)
        if rv_line!=None:
            axarr.axvline(rv_line,ymin=-1000,ymax=1000,color=lcolor,linestyle='--',alpha=lalpha)
        if kp_line!=None:
            axarr.axhline(kp_line,xmin=np.min(self.rvs)-1,xmax=np.max(self.rvs)+1,color=lcolor,linestyle='--',alpha=lalpha)
        axarr.set_xlabel(r'systemic velocity (km s$^{-1}$)',fontsize=16)
        axarr.set_ylabel(r'planet velocity (km s$^{-1}$)',fontsize=16)
        axarr.set_ylim(np.min(self.velocities),np.max(self.velocities))
        temp=fig.colorbar(x)
        if self.func==xcor:
            temp.ax.set_ylabel(r'CCF height')
        else:
            temp.ax.set_ylabel(r'$\chi^{2}$')
        if showplot:
            plt.show() 
        plt.close()
        return fig,axarr

    def plotccfline(self,kp_val,rv_line=None,showplot=True,lcolor='gray',lalpha=0.5,cm='black',rv_lim=None):
        dif=np.abs(self.velocities-kp_val)
        loc=np.where(dif==np.min(dif))        
        
        #print(loc)
        #print(plotvals[loc])
        fs=16
        m0=np.min(self.ccarr[loc])
        m1=np.max(self.ccarr[loc])
        print(m0,m1)
        plt.figure(figsize=(8,3))
        plt.plot(self.rvs,self.ccarr[loc].flatten(),color=cm,lw=2)
        if rv_line!=None:
            plt.vlines(rv_line,ymin=m0*1.1,ymax=m1*1.1,color=lcolor,linestyle='--') 
        plt.ylim(m0*1.1,m1*1.1)
        if rv_lim!=None:
            plt.xlim(-rv_lim,rv_lim)
        
        plt.xlabel('systemic velocity (km s$^{-1}$)',fontsize=fs)

        plt.ylabel(r'CCF Height',fontsize=fs)
        #if rv_lim!=None or plotlim!=None:
            #if plotlim!=None:
                #plt.xlim(-plotlim,plotlim)
            #else:
                #plt.xlim(-rv_lim*.95,rv_lim*.95)        
        if showplot:
            plt.show() 
        plt.close()        

        return self.rvs,self.ccarr[loc].flatten()
        #plt.close()

    def plotcoadd(self,kp_val,wl0,dw=2,rv_line=None,wlunits='$\mathrm{\AA}$',lcolor='gray',lalpha=1.,showplot=True):

        def vel2wl2(wl):
            return vel2wl(wl,wl0)
        
        
        if wlunits!='microns':
            if wlunits=='$\mathrm{\AA}$':
                wls=self.wls*1e4
            elif wlunits=='nm':
                wls=self.sls*1e3
        else:
            wls=self.wls
        #vticklocs=np.round(wl2vel(np.array(ticklocs),cenwl),0)

        fig,axarr=plt.subplots(figsize=(16,5))
        #axarr = fig.add_subplot(1,1,1)
        
        dif=np.abs(self.velocities-kp_val)
        loc=np.where(dif==np.min(dif))
        axarr.plot(wls,self.coadds[loc].flatten())
        ax2=axarr.secondary_xaxis("top", functions=(wl2vel, vel2wl2))
        
        plotvals=self.coadds[loc].flatten()[np.where((wls<(wl0+dw))&(wls>(wl0-dw)))]
        minpv=np.min(plotvals)
        maxpv=np.max(plotvals)
        if rv_line!=None:
            lineloc=veltodeltawl(rv_line,wl0)+wl0
            axarr.vlines(lineloc,ymin=minpv*1.2,ymax=maxpv*1.2,color=lcolor,linestyle='--',alpha=lalpha)

        axarr.set_xlabel(r'wavelength ('+wlunits+')',fontsize=16)
        ax2.set_xlabel(r'velocity (km/s)',fontsize=16)

        axarr.set_ylabel(r'coadd',fontsize=16)
        axarr.set_xlim(wl0-dw,wl0+dw)
        axarr.set_ylim(np.min(plotvals)*1.1,np.max(plotvals)*1.1)

        if showplot:
            plt.show()
        plt.close()
        return wls,self.coadds[loc].flatten()

    def _createsigmatrixdata_(self,kp_lim=100,neg_kp=False,rv_lim=None,block=[],tonorm=True,negsig=False):
        if rv_lim==None:
            rv_lim=np.max(self.rvs)
        
        if kp_lim>np.max(self.velocities):
            kp_lim=np.max(self.velocities)
            
        #cut out data to plot    
        if neg_kp==False:
            yesvel=np.where((self.velocities >= 0) & (self.velocities <=kp_lim))
        else:    
            yesvel=np.where((self.velocities >= -kp_lim) & (self.velocities <=kp_lim)) 
            
        yesrv=np.where((self.rvs <=rv_lim) & (self.rvs >= -1*rv_lim))
       
        temp2=self.ccarr[yesvel[0]]
        ccarr_plot=temp2[:,yesrv[0]]
        rvs_plot=self.rvs[yesrv]
        velocities_plot=self.velocities[yesvel]
        
        
        #remove mask region from stats
        if block!=[]:
            newarrl=[]
            i=0
            while i<len(rvs_plot):
                j=0
                while j<len(velocities_plot):
                    #print rv0[i], vel0[j]
                    if (rvs_plot[i]<block[0][0] or rvs_plot[i] > block[0][1]) and (velocities_plot[j]<block[1][0] or velocities_plot[j]>block[1][1]):
                        newarrl.append(ccarr_plot[j,i])
                    j=j+1
                i=i+1        
            
            newarr=np.array(newarrl)
            #newarr=newarr.flatten()
            sd=np.std(newarr,ddof=1)
            avg=np.mean(newarr)
            med=np.median(newarr)
            #print(avg2, med2, sd2)
        
        else:
            sd=np.std(ccarr_plot,ddof=1)
            avg=np.mean(ccarr_plot)
        if tonorm==True:
            plotvals=(ccarr_plot-avg)/sd
        else:
            plotvals=(ccarr_plot)/sd
            
        if negsig:
            plotvals=plotvals*-1    
        
        return plotvals,rvs_plot,velocities_plot


    def plotccf_sig(self,kp_lim=100,neg_kp=False,rv_lim=None,block=[],tonorm=True,levels=10,cm='mine',fs=16,rv_line=None,kp_line=None,negsig=False,showplot=True,lcolor='gray',lalpha=0.5):
        #fs is fontsize
        #block is 2x2 arr
        #block=[[xmin,xmax],[ymin,ymax]] for blocked region
        #good choices for cm -- 'mine', 'ocean', or 'tab20' with 20 levels
        plotvals,rvs_plot,velocities_plot=self._createsigmatrixdata_(kp_lim=kp_lim,neg_kp=neg_kp,rv_lim=rv_lim,block=block,tonorm=tonorm,negsig=negsig)

        if cm=='mine':
            C=[[140,70,0],[170,70,10],[200,60,20],[240,60,20],[255,60,20],[255,110,60],[255,140,80],[255,160,120],[255,180,160],[255,255,255]]               
            cm = mpl.ListedColormap(np.array(C)/255.0)
        print(np.max(plotvals))    
        fig=plt.figure(figsize=(16,5))
        axarr = fig.add_subplot(1,1,1)
        x=axarr.contourf(np.round(rvs_plot,1),velocities_plot,plotvals,levels,cmap=cm)
        if rv_line!=None:
            axarr.axvline(rv_line,ymin=-1100,ymax=1001,color=lcolor,linestyle='--',alpha=lalpha)
        if kp_line!=None:
            axarr.axhline(kp_line,xmin=np.min(self.rvs)-1,xmax=np.max(self.rvs)+1,color=lcolor,linestyle='--',alpha=lalpha)
        axarr.set_xlabel(r'systemic velocity (km s$^{-1}$)',fontsize=fs)
        axarr.set_ylabel(r'planet velocity (km s$^{-1}$)',fontsize=fs)
        axarr.set_ylim(np.min(velocities_plot),np.max(velocities_plot))
        temp=fig.colorbar(x)
        temp.ax.set_ylabel(r'significance ($\sigma$)',fontsize=fs)
        self.ccarr_sig=plotvals
        if showplot:
            plt.show() 
        plt.close()
        #sself.allcc_sigs.append(plotvals)
        return fig,axarr
    
    def findccf_sig(self,kp_lim=240,neg_kp=False,rv_lim=None,block=[],tonorm=True,levels=10,cm='mine',fs=16,rv_loc=[-1000,1000],kp_loc=[-1000,1000],negsig=False):
        #fs is fontsize
        #block is 2x2 arr
        #block=[[xmin,xmax],[ymin,ymax]] for blocked region
        #good choices for cm -- 'mine', 'ocean', or 'tab20' with 20 levels
        plotvals,rvs_plot,velocities_plot=self._createsigmatrixdata_(kp_lim=kp_lim,neg_kp=neg_kp,rv_lim=rv_lim,block=block,tonorm=tonorm,negsig=negsig)
        rvs,kps=np.meshgrid(rvs_plot,velocities_plot)
        loc=np.where((rvs<rv_loc[1]) & (rvs>rv_loc[0]) & (kps<kp_loc[1]) & (kps>kp_loc[0]))
        return plotvals[loc]
        #plt.close()    
        
    def plotccfline_sig(self,kp_val,kp_lim=300,neg_kp=False,rv_lim=None,plotlim=None,block=[],sig3=False,tonorm=True,levels=10,cm='black',fs=16,rv_line=None,kp_line=None,showplot=True,lcolor='gray',negsig=False):
        #fs is fontsize
        #block is 2x2 arr
        #block=[[xmin,xmax],[ymin,ymax]] for blocked region
        #good choices for cm -- 'mine', 'ocean', or 'tab20' with 20 levels
        if rv_lim==None:
            rv_lim=np.max(self.rvs)
        
        if kp_lim>np.max(self.velocities):
            kp_lim=np.max(self.velocities)
            
        #cut out data to plot    
        if neg_kp==False:
            yesvel=np.where((self.velocities >= 0) & (self.velocities <=kp_lim))
        else:    
            yesvel=np.where((self.velocities >= -kp_lim) & (self.velocities <=kp_lim)) 
            
        yesrv=np.where((self.rvs <=rv_lim) & (self.rvs >= -1*rv_lim))
       
        temp2=self.ccarr[yesvel[0]]
        ccarr_plot=temp2[:,yesrv[0]]
        rvs_plot=self.rvs[yesrv]
        velocities_plot=np.array(self.velocities[yesvel])
        
        
        #remove mask region from stats
        if block!=[]:
            newarrl=[]
            i=0
            while i<len(rvs_plot):
                j=0
                while j<len(velocities_plot):
                    #print rv0[i], vel0[j]
                    if (rvs_plot[i]<block[0][0] or rvs_plot[i] > block[0][1]) and (velocities_plot[j]<block[1][0] or velocities_plot[j]>block[1][1]):
                        newarrl.append(ccarr_plot[j,i])
                    j=j+1
                i=i+1        
            
            newarr=np.array(newarrl)
            #newarr=newarr.flatten()
            sd=np.std(newarr,ddof=1)
            avg=np.mean(newarr)
            med=np.median(newarr)
            #print(avg2, med2, sd2)
        
        else:
            sd=np.std(ccarr_plot,ddof=1)
            avg=np.mean(ccarr_plot)
        if tonorm==True:
            plotvals=(ccarr_plot-avg)/sd
        else:
            plotvals=(ccarr_plot)/sd
            

        #print(np.max(plotvals))
        dif=np.abs(self.velocities-kp_val)
        loc=np.where(dif==np.min(dif))        
        
        #print(loc)
        #print(plotvals[loc])
        if negsig:
            plotvals=plotvals*-1
        m0=np.min(plotvals[loc])
        m1=np.max(plotvals[loc])
        print(m0,m1)
        plt.figure(figsize=(8,3))
        plt.plot(rvs_plot,plotvals[loc].flatten(),color=cm,lw=2)
        if rv_line!=None:
            if m0>-3:
                m0=-3.
            if m1<3:
                m1=3.
            plt.vlines(rv_line,ymin=m0*1.1,ymax=m1*1.1,color=lcolor,linestyle='--') 
        plt.ylim(m0*1.1,m1*1.1)
        if sig3:
            plt.hlines(-3,-rv_lim,rv_lim,color='darkgray',linestyle=':') 
            plt.hlines(3,-rv_lim,rv_lim,color='darkgray',linestyle=':') 
        
        plt.xlabel('systemic velocity (km s$^{-1}$)',fontsize=fs)

        plt.ylabel(r'CCF Significance ($\sigma$)',fontsize=fs)
        if rv_lim!=None or plotlim!=None:
            if plotlim!=None:
                plt.xlim(-plotlim,plotlim)
            else:
                plt.xlim(-rv_lim*.95,rv_lim*.95)        
        if showplot:
            plt.show() 
        plt.close()
        
        
        #plt.close()            


def domanysigplots(plotret,nrows,ncols,figsize=(16,5),levels=10,cm='mine',fs=16,rv_line=None,kp_line=None,negsig=False,lcolor='gray',lalpha=0.5):
    
    if isinstance(levels,int):
        ps=[]
        for item in plotret:
            ps.append(np.array(item[0]))
        t=np.array(ps)
        print(t.shape)
        maxv=np.max(t)
        minv=np.min(t)
        
        levs=np.arange(levels+1)/(levels+1)*(maxv-minv)+minv
        levs=np.round((levs*10),0)/10
        #print(levs)
    else:
        levs=levels
    
    
    if cm=='mine':
        C=[[140,70,0],[170,70,10],[200,60,20],[240,60,20],[255,60,20],[255,110,60],[255,140,80],[255,160,120],[255,180,160],[255,255,255]]               
        cm = mpl.ListedColormap(np.array(C)/255.0)    
    l=len(plotret)
    f= plt.figure(figsize=figsize,facecolor='white') 
    ax = f.add_subplot(111)
    a=f.subplots(nrows,ncols,sharex=True,sharey=True)
    #print(a.shape)
    for i in range(l):
        plotvals,rvs_plot,velocities_plot=plotret[i]
        col=int(np.floor(np.float(i)/nrows))
        row=i % nrows
        #print(row,col)
        
        x=a[row,col].contourf(np.round(rvs_plot,1),velocities_plot,plotvals,levels=levs,cmap=cm,extend='both')
        if rv_line!=None:
            a[row,col].axvline(rv_line,ymin=np.min(velocities_plot)-1,ymax=np.max(velocities_plot)+1,color=lcolor,linestyle='--',alpha=lalpha)
        if kp_line!=None:
            a[row,col].axhline(kp_line,xmin=np.min(rvs_plot)-1,xmax=np.max(rvs_plot)+1,color=lcolor,linestyle='--',alpha=lalpha)
        
        #if row==(nrows-1):
            
        #if col==0:
            
    temp= f.colorbar(x, ax=a.ravel().tolist(), shrink=0.95)
    
    if np.max(plotvals)>1.0:
        temp.ax.set_ylabel(r'CCF Significance ($\sigma$)',fontsize=fs)  
    else:
        temp.ax.set_ylabel(r'CCF Height',fontsize=fs)     
    #ax = f.add_subplot(111) 
    ax.set_ylabel(r'planet velocity (km s$^{-1}$)',fontsize=fs)
    ax.set_xlabel(r'systemic velocity (km s$^{-1}$)',fontsize=fs)
    #ax.set_xticks(color='w')
    #ax.set_yticks(color='w')
    ax.tick_params(colors='white',axis='both')
    
def domanylinesigplots(plotret,kp_val=80,nrows=2,ncols=1,figsize=(16,5),levels=10,cm='darkviolet',fs=16,rv_line=None,rv_lim=None,sig3=False,plotlim=None,negsig=False,lcolor='gray',lalpha=0.5):
    

    l=len(plotret)
    f= plt.figure(figsize=figsize,facecolor='white') 
    ax = f.add_subplot(111)
    a=f.subplots(nrows,ncols,sharex=True,sharey=True)
    #print(a.shape)
    for i in range(l):
        plotvals,rvs_plot,velocities_plot=plotret[i]
        col=int(np.floor(np.float(i)/nrows))
        row=i % nrows
        #print(row,col)
        loc=np.where(velocities_plot==kp_val)
        #print(loc)
        #print(plotvals[loc])
        if negsig:
            plotvals=plotvals*-1
        m0=np.min(plotvals[loc])
        m1=np.max(plotvals[loc])
        #print(m0,m1)
        #plt.figure(figsize=(8,3))
        a[row,col].plot(rvs_plot,plotvals[loc].flatten(),color=cm,lw=2)
        if rv_line!=None:
            if m0>-3:
                m0=-3.
            if m1<3:
                m1=3.
            a[row,col].vlines(rv_line,ymin=m0*1.1,ymax=m1*1.1,color=lcolor,linestyle='--') 
        a[row,col].set_ylim(m0*1.1,m1*1.1)
        if sig3:
            a[row,col].hlines(-3,-rv_lim,rv_lim,color='darkgray',linestyle=':') 
            a[row,col].hlines(3,-rv_lim,rv_lim,color='darkgray',linestyle=':') 
        

        if rv_lim!=None or plotlim!=None:
            if plotlim!=None:
                a[row,col].set_xlim(-plotlim,plotlim)
            else:
                a[row,col].set_xlim(-rv_lim*.95,rv_lim*.95)              
         
        #if col==0:
            
    if np.max(plotvals)>1.0:
        ax.set_ylabel(r'CCF Significance ($\sigma$)',fontsize=fs)  
    else:
        ax.set_ylabel(r'CCF Height',fontsize=fs) 
    #ax = f.add_subplot(111) 
    #ax.set_ylabel(r'planet velocity (km s$^{-1}$)',fontsize=fs)
    ax.set_xlabel(r'systemic velocity (km s$^{-1}$)',fontsize=fs)
    
    #plt.xlabel('systemic velocity (km s$^{-1}$)',fontsize=fs)
    
      
    #ax.set_xticks(color='w')
    #ax.set_yticks(color='w')
    ax.tick_params(colors='white',axis='both')