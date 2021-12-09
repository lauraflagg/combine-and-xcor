
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 5)
plt.rcParams.update({'errorbar.capsize': 3})
plt.rcParams['xtick.labelsize']= 13
plt.rcParams['ytick.labelsize']= 13
plt.rcParams.update({'axes.labelsize': 16})


# In[10]:


import sys
sys.path.append('../../toimport')


# In[11]:


from astro_lf import createwlscale, coadd_echelle


# In[12]:


import os, pickle
import astropy.constants
import spectres
from PyAstronomy import pyasl
#from wasp49_pars import *

def getphase(mjd,day0,per):
    return ((mjd-day0)/per) % 1.0
def intransit(mjd,per,td_days,day0):


    pha=getphase(mjd,day0,per)

    td_perc=td_days/per
    td_perc_half=td_perc/2
    #print(pha,td_perc_half)
    if pha<td_perc_half or pha>(1-td_perc_half):
        return True
    else:
        return False


def vac2air(vacwave):
    """
    Converts a wavelength in vacuum to a wavelength in air.
    Input must be in Angstroms (though would be nice to introduce
    astropy units flexibility).
    Parameters
    ----------
    vacwave: array
        input wavelength axis in vacuum
    Returns
    ----------
    airwave: array
        output wavelength axis in air
    """

    s = 1e4 / vacwave # in AA
    n = 1. + 8.34254e-5 + 2.406147e-2 / (130. - s**2) + 1.5998e-4 / (38.9 - s**2)
    airwave = vacwave / n

    return airwave



# In[13]:
def convert_GRACES(per,day0,td_days,filestart='graces',fileend='.npz',sigcut=None,orders=np.arange(70),minlen=5903):

    ls=len(filestart)
    le=len(fileend)
    files=os.listdir('../data/processed/orders/')
    
    #print(files)
    orders=list(orders)
    orders.append(10000)
    dats_w=[]
    dats_m=[]
    dats_e=[]
    for item in files:
        fo=item.find('order-')
        #print(fo)
        if fo!=-1:
            test=item[fo+6:fo+8]
            #print(test)
            order=int(test)
        else:
            order=10000
        if item[-le:]==fileend and item[:ls]==filestart and order in orders:
            data = np.load('../data/processed/orders/'+item)
            l=data['wave'].shape[1]
            #print(l)
            more=l-minlen
            if more<2:
                start=0
            elif more<4:
                start=1
            elif more<6:
                start=2
            elif more<8:
                start=3
            else:
                start=4
            dats_w.append(data['wave'][:,start:(start+minlen)])
            dats_e.append(data['error'][:,start:(start+minlen)])
            dats_m.append(data['sysr'][:,start:(start+minlen)])

    
    dats_ma=np.array(dats_m)
    dats_fl0=10.**(dats_ma/-2.5)
    dats_w=np.array(dats_w)
    dats_u=np.array(dats_e)




    dats_fl=dats_fl0.copy()
    for i,item in enumerate(dats_fl0):
        #print(i)
        if sigcut!=None:
            
            avg=np.average(item)
            sd=np.std(item, ddof=1)
            loc=(item>(sigcut*sd+avg)) ^ (item<(-sigcut*sd+avg))
            dats_fl[i][loc]=1





    wls=createwlscale(2.15,low=np.ceil(np.min(dats_w)*10),up=np.floor(np.max(dats_w)*10))/10




    wls_a=vac2air(wls*10)/10




    norders=dats_fl.shape[0]
    ndates=dats_fl.shape[1]

    

    fl_out=np.zeros((ndates,len(wls)))
    
    for i in range(ndates): #for all frames
        fl_out[i,:],u_out=coadd_echelle(dats_w[:,i,:],dats_fl[:,i,:],dats_u[:,i,:],wls_a)
        #print(dats_w[i,:,:].shape)
   #     f0=spectres.spectres(wls_a,dats_w[i,0,:],dats_fl[i,:,:],fill=0,verbose=False)
   #     fl_out[np.where(f0!=0)]=f0[np.where(f0!=0)]


    # In[31]:


    fl_out[np.where(fl_out==0)]=1


    # In[32]:


    phases=np.load('../data/processed/orbital.npy',allow_pickle=True)


    # In[33]:


    berv=np.load('../data/processed/berv.npy',allow_pickle=True)



    fl_out1=np.zeros_like(fl_out)
    for i,item in enumerate(fl_out):
        nf_1, wl_1 = pyasl.dopplerShift(wls*10, item, berv[i], edgeHandling='firstlast', fillValue=None)
        fl_out1[i]=nf_1




    mjds=phases*per+day0




    intransit_arr=[]
    for item in mjds:
        #print(getphase(item))
        intransit_arr.append(intransit(item,per=per,td_days=td_days,day0=day0))


    # In[44]:


    med_spec=np.zeros_like(wls)
    med_spec[:]=np.median(fl_out1[~np.array(intransit_arr)],axis=0)




    # In[46]:


    f=fl_out1/med_spec-1







    sigfn='_sigcut'+str(sigcut)
    if sigcut==None:
        sigfn=''
        
    fnbase=filestart+'_'+fileend[:-5]+sigfn+'.pic'
    # In[53]:


    num=len(mjds)
    toprint=np.zeros((num+1,len(wls)))
    toprint[0]=wls/1e3 #to convert to microns
    for i in range(num):
        toprint[i+1]=f[i]
    d=['wavelength_(microns)']
    d=d+list(mjds.astype(np.str))

    fn='../data/'+fnbase
    print(fn)

    pickle.dump((d,np.transpose(toprint)),open(fn[:-3]+'pic','wb'),protocol=2)


    # In[54]:


    num=len(mjds)
    toprint=np.zeros((num+1,len(wls)))
    toprint[0]=wls/1e3 #to convert to microns
    for i in range(num):
        toprint[i+1]=fl_out1[i]-1.
    d=['wavelength_(microns)']
    d=d+list(mjds.astype(np.str))


    fn='../data/nodivide'+fnbase
    pickle.dump((d,np.transpose(toprint)),open(fn[:-3]+'pic','wb'),protocol=2)



