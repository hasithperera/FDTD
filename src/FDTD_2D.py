# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:30:27 2021

@author: HasithPerera
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import time

domain_x = 201;
domain_y = 201;

src_loc = [0,0]


Ez = np.zeros([domain_x,domain_y])
# Ez[100:102,25:27]= 1 
# Ez[98:102,98:102]=1;


Hx = np.zeros([domain_x,domain_y])
Hy = np.zeros([domain_x,domain_y])

dy = 1
dx = 1
dz = 1.;


# match CFL condition
mu = 2;
epsilon = 2;
dt = 1;


def src_gaussian_2D(x,mu=20,sig=10):   
    return 70*np.exp(-1*(np.power(x-mu,2))/np.power(sig,2))
    

def update_FDTD_2D(t,run_n):
    '''
    Main update 2D ftdt function
    
    t       : time step start
    run_n   : times steps before update
    
    '''
    
    print("Start:",t)
    # print(time)    
    # Ez[100,100] = 1
    
    for tt in range(t,t+run_n):
      
        Ez[70,70] = src_gaussian_2D(tt)
        
        #equation 3   
        Ez[1:domain_x,1:domain_y] = ( Ez[1:domain_x,1:domain_y] 
                                          - (dt/(epsilon*dy))*(Hx[1:domain_x,1:domain_y] 
                                                              - Hx[1:domain_x,0:domain_y-1]) 
                                          + (dt/(epsilon*dx))*(Hy[1:domain_x,1:domain_y]
                                                             -Hy[0:domain_x-1,1:domain_y]))
    
        #equation 1
        Hx[:,0:domain_y-1] = Hx[:,0:domain_y-1] - (Ez[:,1:domain_y]-Ez[:,0:domain_y-1])*(dt/(mu*dy))
        
        
        
        #equation 2
        Hy[0:domain_x-1,:] = Hy[0:domain_x-1,:] +(Ez[1:domain_x,:]-Ez[0:domain_x-1,:])*(dt/(dx*mu))
        
       
    
    print("End:",t+run_n)
    im.set_array(Ez)   
    return im,
    
    

def init():
    print("Clear Ez")
    return im,



data = []
data_src = []
fig = plt.figure( figsize=(8,8) )


if __name__=='__main__':
    
    print("Start")
    im = plt.imshow(Ez)
    
    plot_tstep = 10
    t_end = 1000 
   
    ani = FuncAnimation(fig, update_FDTD_2D, frames=np.arange(0, t_end, plot_tstep),
                    init_func=init, blit=True,interval=200,fargs=[plot_tstep],repeat = False)
    
    
    
    im = plt.imshow(Ez,vmin=0,vmax=1)
            