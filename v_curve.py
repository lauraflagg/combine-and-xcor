import numpy as np
import matplotlib.pyplot as plt
#pro vcurve,g,pd,ai,e,w,fout

# This procedure creates a 2 x 360 array containing the 
# radial velocity data as a function of time for a spectroscopic
# binary.  Input g (center of mass velocity of system), pd
# (period of orbit in days), ai (asin(i) of orbit in giga-meters),
# e (eccentricity of orbit), w (longitude of periastron), and
# fout (name of output file).
# Formulae from Heintz (1978), Danby (1990), and math tables.
# LAP 9.1.96

# print statement to prompt input:

#print,'type g(km/s),pd(days),ai(Gm),e(ecc), w(deg) & output (filename)'

#port from IDL by Laura Flagg
def v_curve(g,pd,ai,e,w):
   fout=np.zeros((2,360))
   out=np.zeros((2,360))
   
   # define pi and convert to useful units
   
   pi=np.pi
   # convert to seconds:
   p=pd*8.64e4
   # convert to km:
   asi=ai*1e6
   # convert to radians:
   wr=(w/360.0)*2*pi
   
   #stop
   
   t1=g
   t2=2*pi/p #units of s^-1, angular frequency
   t3=asi #distaance
   ed=(1-e**2)
   t4=np.sqrt(1/ed)
   t5=e*np.cos(wr) #large if wr is close to 0, dimensionless
   erat=(np.sqrt((1-e)/(1+e)))
   
   # print,sqrt((1-e)/(1+e))
   #stop
   for ji in range(0,360):
   
      j=ji+540
      jr=(j/360.0)*2*pi #the angle in radians
   
      t6=np.cos(jr+wr) #combine the positiona angle with the angle of periastron
   
      v=t1+(t2*t3*t4*(t5+t6))
   
      prt1=(e*np.sin(jr)*np.sqrt(ed)/(1+e*np.cos(jr)))
      prt2=2*np.arctan(erat*np.tan(jr/2.0))
      if ji == 0:
         prt2=-pi      
         #because of weird idl, they get -pi, while python gets pi, so correcting that
      jj=j-540
      t=(1/t2)*(prt2-prt1)
   
      out[0,jj]=t/p
      out[1,jj]=v
    #  if ji == 180:
     #    print prt1, prt2, t, t2, t6,p, out[0,jj], jj, jr, np.tan(jr/2.0)

         
   
   #   print,j,jr,t6,v,out(0,jj),out(1,jj)
   
      
   if out[0,0] < 0.0:
      out[0,]=out[0,] - out[0,0]
      
   
   fout[0,0:180]=out[0,180:]-out[0,180]
   fout[0,180:]=out[0,0:180]+out[0,180]
   fout[1,0:180]=out[1,180:]
   fout[1,180:]=out[1,0:180]
   
   return fout

def citau(phase,par):#
   #function func_citau,phase,par
   #
   #  This function computes the radial velocity amplitude for a single
   #  line spectroscopic binary, using the code vcurve.pro supplied by
   #  Lisa Prato.  
   #  INPUTS:
   #    phase - The orbital phase for the desired points
   #    par(0) - g: center of mass velocity of system in m/s
   #    par(1) - pd: period of orbit in days
   #    par(2) - ai: asin(i) of orbit in giga-meters
   #    par(3) - e: eccentricity of orbit
   #    par(4) - w: longitude of periastron
   #    par(5) - ph0: Phase offset    
#  OUTPUTS:
#    function returns the velocity of the star in m/s
#
#  HISTORY:
#    10-Apr-2014 CMJ - Written, based on func420.pro for XO-3b
#    25-Feb-2007 CMJ - Written
#    13-Mar-2007 CMJ - Added Phase offset term
#

   # Set up variables
   g = par[0]/1000.
   pd = par[1]
   par[2] = abs(par[2])
   ai = par[2]
   par[3] = abs(par[3])
   e = par[3]
   par[4] = par[4] % 360.
   w = par[4]
   par[5] = (par[5]+20.) % 1.
   ph0 = par[5]
   #if ph0 lt 0. then ph0 = 0.
   #if ph0 gt 1. then ph0 = 1.
   
   fout=v_curve(g,pd,ai,e,w)
   
   # Interpolate onto phases and return
   #
   #vel = 1.d3*interpol(reform(fout(1,*)),reform(fout(0,*)),((phase+ph0) mod 1.)) 
   a=fout[1]
   b=fout[0]
   c=((phase+ph0) % 1.)
   vel=np.interp(c,b,a)*1000.
   #in m/s
   #vel(10:20) = vel(10:20) + par(6)             # adjust HET velocities
   

   
   return vel

if 1==2:   
   par=np.zeros(6)
   par[0] = -134.70961 #center of mass velocity in m/s
   par[1] = 8.9891005 #period
   par[2] = 0.11257813 #  asin(i) of orbit in giga-meters
   par[3] = 0.25086000 #eccentricity
   par[4] = 31.342030 ;#arg of periastron
   par[5] = 0.51032202 # phase offset
   
   phases=np.arange(0,101)/100.
   
   a_0=citau(phases,par)/1000.
   #velocities now in km/s
   