# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:39:39 2022

@author: ICER
"""

import os
import math as m
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from IPython import get_ipython;   
get_ipython().magic('reset -sf')
plt.close()

clear = lambda: os.system('cls')
Pr=0.744

def conv(f,t):
    return (f[1],f[2],-0.5*f[0]*f[2],f[4],-0.5*Pr*((f[0]*f[4])-(f[1]*f[3])))

    
eta=np.linspace(0,10,100)

def sol(x,ts):
    f0=[0.,0.,x[0],x[1],-1.]
    f=odeint(conv,f0,eta)
    return [1.-f[-1,1],f[-1,3]]
    

s=scipy.optimize.fsolve(sol,(1.,-1.),args=(eta))
f0=[0,0,s[0],s[1],-1.]
f=odeint(conv,f0,eta)


f0=f[:,0].tolist()
f1=f[:,1].tolist()
f2=f[:,2].tolist()
f3=f[:,3].tolist()
f4=f[:,4].tolist()

plot1=plt.figure(1)
plt.plot(eta,f1, label='f')
plt.xlabel(" \u03B7 ")
plt.ylabel(" f' ")

plot1=plt.figure(2)
plt.plot(eta,f3, label='f')
plt.xlabel(" \u03B7 ")
plt.ylabel(" \u03B8 ")
clear()


