# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:58:33 2022

@author: ICER
"""
import math as m
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize




def blasius(f,t):
    return (f[1],f[2],-0.5*f[0]*f[2])

    
eta=np.linspace(0,10,100)

def blas0(x,ts):
    f0=[0.,0.,x]
    f=odeint(blasius,f0,eta)
    return 1.-f[-1,1]
        

xpp0=scipy.optimize.fsolve(blas0,x0=0.1,args=(eta))
f0=[0,0,xpp0]
f=odeint(blasius,f0,eta)    
 
f0=f[:,0].tolist()
f1=f[:,1].tolist()
f2=f[:,2].tolist()
print(f0)
print(eta)

for i in range((len(f1))):
    if f1[i]>0.99: 
        print('The value of eta at boundary layer thickness = ',eta[i]) 
        print('The value of f1 at boundary layer thickness = ', f1[i]) 
        break

headerlist=['eta_th','f2_th','f1_th','f0_th']
df = pd.read_csv (r'E:\Fenil Course\Convective Heat Transfer\Assignment\Cengel data.csv', header=None, names=headerlist)

uinf=0.5
nu=14.9*1e-06
xloc=[0.1,0.2,0.3,0.4,0.5]

u=[0]*len(f1)
y=[[0]*len(eta)]*len(xloc)

for i in range (len(xloc)):
    y[i]=eta/(m.sqrt(uinf/(nu*xloc[i])))

u=uinf*f[:,1]    


#print(f1_th)     
eta_th=df['eta_th'].iloc[1:25].astype(float).array
f2_th=df['f2_th'].iloc[1:25].astype(float).array
f1_th=df['f1_th'].iloc[1:25].astype(float).array
f0_th=df['f0_th'].iloc[1:25].astype(float).array
eta_th=eta_th.to_numpy()
f1_th=f1_th.to_numpy()



plot1=plt.figure(1)
plt.plot(f0,eta, label='f')
plt.plot(f1,eta, label='f')
plt.plot(f2,eta, label='f"')
plt.xlim([0,2])
plt.ylim([0,7])
plt.xlabel(" f,f',f'' ")
plt.ylabel("Eta")
plt.legend()

plot2=plt.figure(2)
plt.plot(f1,eta, label='present code')
plt.plot(f1_th,eta_th,label='cengel.soln')
##plt.xlim([0,5])
plt.ylim([0,7])
plt.xlabel("f'")
plt.ylabel("Eta")
plt.legend()

plot3=plt.figure(3)
plt.plot(y[0],u,y[1],u,y[2],u,y[3],u,y[4],u)
plt.xlabel('Y (m)')
plt.ylabel("U (m/s)")
plt.legend(['x=0.1 m', 'x=0.2 m','x=0.3 m', 'x=0.4 m','x=0.5 m'])
plt.show()


