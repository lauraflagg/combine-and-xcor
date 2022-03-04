import sys
sys.path.append('C:/Users/laura/programming/python/toimport')
from stats_lf import xcor, xcor_fast, chisq, log_likelihood_zucker, gaussian2D
import numpy as np
from astro_lf import findbests2n, vel2wl, c_kms,wl2vel,veltodeltawl, getorbitpars, createwlscale
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




        
class CCFMatrix():
    def __init__(self,velocities=np.arange(0, 110),low=-50,up=50,method='median',func=xcor,order='cx',disp=2.04,fromfile='',trim=5):
        '''trim is % trimmed if method is trimmed mean'''
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
        self.init_data=False
        self.trim=trim
    
    def _addspecdata_(self,spectrumset,orbitalpars,day0,code):
        self.init_data=True
        self.orbitalpars=orbitalpars
        self.idnum=spectrumset.idnum
        self.day0=day0        
        self.wls=spectrumset.wls
        self.byorder=spectrumset.byorder
        maxkp=np.max(self.velocities)
        
        
    
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
        

        #print(maxplanetrvs)
        self.l_good=len(spectrumset.hjd)
        
        self.orbitinfo=(maxplanetrvs,maxkp)
        
        return maxplanetrvs,maxkp
    
    def _shiftbykp_(self,spectrumset,vtemp,maxplanetrvs,maxkp):
        if spectrumset.byorder:
            specs=np.zeros_like(spectrumset.gooddata)
        else:
            specs=[]
            
        i=0
        while i<self.l_good:
            flux=spectrumset.gooddata[i]

            if spectrumset.byorder:
                flux[spectrumset.badwls_mask==1]=0
            else:
                flux[spectrumset.badwls]=0              


            v_p=maxplanetrvs[i]*vtemp/maxkp
            #print(v_p)
            #velocity of planet in km/s
            
            if spectrumset.byorder:
                for o, f_o in enumerate(flux):
                    nf_1, wl_1 = pyasl.dopplerShift(spectrumset.wls[o]*10000, f_o, -v_p, edgeHandling='firstlast', fillValue=None)
                    specs[i,o]=nf_1
                
            else:
                nf_1, wl_1 = pyasl.dopplerShift(spectrumset.wls*10000, flux, -v_p, edgeHandling='firstlast', fillValue=None)

                specs.append(nf_1)
            #since I'm using the shifted flux, the wavelengths stay the same so I don't have to interpolate again


            i=i+1    
        return specs
    

        

    def creatematrix(self, spectrumset,orbitalpars,indstart=0,indend=-1,day0=2453367.8,write=True,code='vcurve',orders=[]):
        
        if self.init_data==False:
            maxplanetrvs,maxkp=self._addspecdata_(spectrumset,orbitalpars,day0,code)
        else:
            maxplanetrvs,maxkp=self.orbitinfo
            

        #indend and indstart is if you want to subset a section of consecutive dates  
        
        if indend==-1:
            indend=len(spectrumset.hjd)
    
    
        ccarr=[]
        cc2arr=[]
        sigarr=[]
        sig2arr=[]    


        
    
        if self.method=='weighted':
            s2narr=spectrumset.s2narr

    
        for vtemp in self.velocities:
            specs=self._shiftbykp_(spectrumset,vtemp,maxplanetrvs,maxkp)

    
            planetfluxes=np.zeros((self.l_good,len(spectrumset.wls)))
    

    
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
                    medspec=stats.trim_mean(specsarr[indstart:indend], self.trim/100.,axis=0)
    
                #never mind, try replacing nan with 0
                medspec=np.nan_to_num(medspec)
    
    
    
                if spectrumset.byorder:
                    cc_o=[]
                    for o,medspec_o in enumerate(medspec):
                        
                        if orders==[] or o in orders:
                            cc=self.func(medspec_o,spectrumset.fl_co[o],self.low,self.up)
                            cc_o.append(np.nan_to_num(cc))
                    cc=np.mean(np.array(cc_o),0)
                else:
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
    def createcoadd(self, spectrumset,orbitalpars,indstart=0,indend=-1,day0=2453367.8,write=True,code='vcurve',orders=[]):
        if orders!=[]:
            if len(orders)>1:
                print('coadds doesnt currently work for multiple orders')
                return None
                
        self.orders=orders
        
        
        if self.init_data==False:
            maxplanetrvs,maxkp=self._addspecdata_(spectrumset,orbitalpars,day0,code)     
        else:
            maxplanetrvs,maxkp=self.orbitinfo


        #indend and indstart is if you want to subset a section of consecutive dates  
        
        if indend==-1:
            indend=len(spectrumset.hjd)
    

        sigarr=[]
        sig2arr=[]    
        meds=[]

    
        if self.method=='weighted':
            s2narr=spectrumset.s2narr


    
        for vtemp in self.velocities:
            specs=self._shiftbykp_(spectrumset,vtemp,maxplanetrvs,maxkp)
    
            planetfluxes=np.zeros((self.l_good,len(spectrumset.wls)))
    
    

            specsarr=np.array(specs)
    
            #trying something diff
            #
            #OR
    

    
            if self.method=='median':

                medspec=np.median(specsarr[indstart:indend],0)
            elif self.method=='mean' or self.method=='average':
                medspec=np.mean(specsarr[indstart:indend],0)

            elif self.method=='weighted':
                print('s2ns=',s2narr)
                medspec=np.average(specsarr[indstart:indend],0,weights=s2narr[0:])
            else:
                medspec=stats.trim_mean(specsarr[indstart:indend], self.trim/100., axis=0)

                #never mind, try replacing nan with 0
            medspec=np.nan_to_num(medspec)
            meds.append(medspec)
        meds=np.array(meds)
        if self.byorder:
            meds_out=np.zeros((meds.shape[0],meds.shape[2]))
            for o in orders:
                meds_out=meds_out+meds[:,o,:]
            meds=meds_out
            self.wls=self.wls[o]
           
        self.coadds=meds
    
    


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
    
        self.l_good=len(spectrumset.hjd)

    
    
        if self.method=='weighted':
            i=0
            while i<self.l_good:
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

        planetfluxes=np.zeros((self.l_good,len(spectrumset.wls)))


        i=0
        while i<self.l_good:
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
    
    def plotcoaddmatrix(self,wlrange=None,wl_line=[],vsys=None,kp_lim=None,kp_line=None,binned=1.):
        '''wl_line in microns'''
        
      
        
        if wlrange!=None:
            loc=((self.wls>=wlrange[0]) & (self.wls<=wlrange[1]))
            wlplot=self.wls[loc]
            coaddplot=self.coadds[:,loc]
        else:
            wlplot=self.wls
            coaddplot=self.coadds
        print('mean=',np.mean(coaddplot),' sd=',np.std(coaddplot))
        print('min=',np.min(coaddplot),' max=',np.max(coaddplot))
            
        if binned!=1:
            dv0=2*(wlplot[1]-wlplot[0])/(wlplot[1]+wlplot[0])*c_kms
            wls0=wlplot.copy()
            #print(dv0,(wl0-dw),(wl0+dw))
            wlplot=createwlscale(dv0*binned,wls0[0],wls0[-1])
            #print(wlplot,wls0)
            coaddplot0=coaddplot.copy()
            coaddplot=spectres.spectres(wlplot,wls0,coaddplot0,verbose=False)
    
            print('mean=',np.mean(coaddplot),' sd=',np.std(coaddplot))
            print('min=',np.min(coaddplot),' max=',np.max(coaddplot))

        fig=plt.figure(figsize=(8,5))
        axarr = fig.add_subplot(1,1,1)        
        x=axarr.contourf(wlplot,self.velocities,coaddplot)
        axarr.set_ylabel('k$_p$ (km/s)')
        axarr.set_xlabel('wavelength (microns)')
        temp=fig.colorbar(x)
        temp.ax.set_ylabel(r'relative flux')
        if wl_line!=[]:
            for item in wl_line:
                w=item+veltodeltawl(vsys,item*1e4)*1e-4
                axarr.axvline(w,ymin=-1000,ymax=1000,color='white',linestyle='--',zorder=20,alpha=.5)
        if kp_line!=None:
            axarr.axhline(kp_line,xmin=np.min(self.rvs)-1,xmax=np.max(self.rvs)+1,color='white',linestyle='--',alpha=.5)        
        if kp_lim==None:
            plt.ylim(0,np.max(self.velocities))
        else:
            plt.ylim(0,kp_lim)
        if binned!=1:
            plt.xlim(wlplot[1],wlplot[-3])
    
    def plotccfvsphase(self,spectrumset, orbitalpars,rv_lim=100,levels=10,cm='viridis',kp_val=None,showplot=True,lcolor='white',lalpha=0.5,phasestart=0,phaseend=0,day0=2453367.8,code='vcurve',per='3.',vsys=0.):

        phases=spectrumset.phases
        #print(phases)
        fig=plt.figure(figsize=(8,5))
        axarr = fig.add_subplot(1,1,1)       
        x=axarr.contourf(np.round(self.rvs,1),phases,self.ccvsphase,levels,cmap=cm,zorder=0)
        temp=fig.colorbar(x)
        if self.func==xcor or self.func==xcor_fast:
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
        '''def findcenter(self,xguess,yguess,xlims=None,ylims=None,xcenlims=None,ycenlims=None,negamp=False)'''
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
        if self.func==xcor or self.func==xcor_fast:
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

    def plotcoadd(self,kp_val,wl0,dw=2,rv_line=None,wlunits='$\mathrm{\AA}$',lcolor='gray',lalpha=1.,showplot=True,binned=1):

        def vel2wl2(wl):
            return vel2wl(wl,wl0)
    
        dif=np.abs(self.velocities-kp_val)
        loc=np.where(dif==np.min(dif))        

        
        if wlunits!='microns':
            if wlunits=='$\mathrm{\AA}$':
                wls=self.wls*1e4
            elif wlunits=='nm':
                wls=self.wls*1e3
        else:
            wls=self.wls
            
        if binned!=1:
            locwl=np.where((wls<(wl0+dw))&(wls>(wl0-dw)))
            dv0=2*(wls[1]-wls[0])/(wls[1]+wls[0])*c_kms
            wls0=wls
            #print(dv0,(wl0-dw),(wl0+dw))
            wls=createwlscale(dv0*binned,(wl0-dw),(wl0+dw))
            
            
        #vticklocs=np.round(wl2vel(np.array(ticklocs),cenwl),0)

        fig,axarr=plt.subplots(figsize=(16,5))
        #axarr = fig.add_subplot(1,1,1)
        


        locwl=np.where((wls<(wl0+dw))&(wls>(wl0-dw)))

            
        plotvals=self.coadds[loc].flatten()[locwl]
        if binned==1:
            wls=wls[locwl]
        
        
        
        
        if binned!=1:
            plotvals=spectres.spectres(wls,wls0,self.coadds[loc].flatten())
        axarr.plot(wls,plotvals)
        ax2=axarr.secondary_xaxis("top", functions=(wl2vel, vel2wl2))            
        
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
        return wls,wl2vel(wls,wl0),plotvals
    
    def coadd_coadds(self,lines,kp_val,dw=2,rv_line=None,wlunits='$\mathrm{\AA}$',lcolor='gray',lalpha=1.,showplot=True,binned=1):
        coadd_res=[self.plotcoadd(kp_val,line,dw,rv_line,wlunits,lcolor,lalpha,showplot=False,binned=binned) for line in lines]
        vels_arr=[item[1] for item in coadd_res]
        #print(coadd_res)
        dv0=vels_arr[0][1]-vels_arr[0][0]
        dv=binned*dv0
        #print(len(vels_arr))
        minmaxs_t=np.array([np.array([np.min(item),np.max(item)]) for item in vels_arr])

        minmaxs=minmaxs_t.transpose()
        minv=np.max(minmaxs[0])
        maxv=np.min(minmaxs[1])
        nbins=np.floor((maxv-minv)/dv)
        vels=np.arange(nbins)*float(dv)+minv

        fluxes=np.array([np.interp(vels,item[1],item[2]) for item in coadd_res])

        coadded_flux=np.sum(fluxes,axis=0)
        if showplot:
            plt.plot(vels,coadded_flux)
            plt.xlabel('velocity (km/s)')
            plt.ylabel('flux')
            plt.show()
        return vels,coadded_flux
            
                         
        


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