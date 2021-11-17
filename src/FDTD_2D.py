# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:30:27 2021

@author: HasithPerera
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import time

domain_y = 500;
domain_x = 500;

src_loc = [0,0]


Ez = np.zeros([domain_y,domain_x])
# Ez[100:102,25:27]= 1 
# Ez[98:102,98:102]=1;


Hx = np.zeros([domain_y,domain_x])
Hy = np.zeros([domain_y,domain_x])


#design custom region
epsilon = np.ones([domain_y,domain_x])*8;

dy = 1
dx = 1
dz = 1


# match CFL condition
mu = 2;
# epsilon = 2;
dt = 1;

data = []
data2 = []
probe_loc = [250,250]





def src_gaussian_2D(x,mu=100,sig=35):   
    return 10*np.exp(-1*(np.power((x-mu)/sig,2)))   

def src_sin(time,A,freq=.08):
    return A*np.sin(time*freq)


def FDTD_2D(time):
    
       
        #equation 3   
        Ez[1:domain_y,1:domain_x] = ( Ez[1:domain_y,1:domain_x] 
                                          - (dt/(epsilon[1:domain_y,1:domain_x]*dy))*(Hx[1:domain_y,1:domain_x] 
                                                              - Hx[1:domain_y,0:domain_x-1]) 
                                          + (dt/(epsilon[1:domain_y,1:domain_x]*dx))*(Hy[1:domain_y,1:domain_x]
                                                             -Hy[0:domain_y-1,1:domain_x]))
        
        #source 
        # Ez[225:275,1] = src_gaussian_2D(time,sig=25)
        Ez[100:400,2] = src_sin(time,2,.1)
        
        
        #equation 1
        Hx[:,0:domain_x-1] = Hx[:,0:domain_x-1] - (Ez[:,1:domain_x]-Ez[:,0:domain_x-1])*(dt/(mu*dy))
        
       
          
        #equation 2
        Hy[0:domain_y-1,:] = Hy[0:domain_y-1,:] +(Ez[1:domain_y,:]-Ez[0:domain_y-1,:])*(dt/(dx*mu))
        
        #probe fields - debug
        
        data.append(Ez[probe_loc[0],probe_loc[1]])
        data2.append(Ez[probe_loc[0],probe_loc[1]+50])
        
    

def update_FDTD_2D(t,run_n):
    '''
    Main update 2D ftdt function
    
    t       : time step start
    run_n   : times steps before update
    
    '''
    
    print("Start:",t)   
    for tt in range(t,t+run_n):
        FDTD_2D(tt)
       
    
    print("End:",t+run_n)
    im.set_array(Ez)   
    im.set_cmap('bwr')
    return im,
    
def init():
    print("Clear Ez")
    return im,


def edge(x,y,x0,y0,angle):
    if y<np.tan(angle)*(x-x0)+y0:
        return 1
    return 0


def draw_lens(lens):
    y0 = [(lens[2]+lens[3])/2,lens[3],lens[2]]
    x0 = [lens[1],lens[0],lens[0]]
    return np.polyfit(y0,x0,2)

def plano_convex(x,y,p):
    return np.ones_like(y)*x<np.polyval(p,y)



def sim_optics1():
    '''Place optical elements in the beam path'''
    eps_2 = 1
    angle = 40*np.pi/180
    
    for x in range(1,domain_x):
        for y in range(1,domain_y):
            if edge(x,y,250,250,angle):
                epsilon[y,x] = eps_2

def sim_optics2():
    
    '''
    Plano-convex lens n=2 
    background n=1
    '''

    eps_2 = 4

    
    p = draw_lens([50,150,0,500])
    for x in range(1,domain_x):
        # print(x)
        # for y in range(1,domain_y):
        epsilon[:,x] = plano_convex(x,np.arange(0,domain_y),p)*eps_2 + 1

        
    epsilon[:,0:50]=1
    print("Lens generated")

def sim_optics3():
    '''
    Plano-convex lens n=1
    background n=2
    '''
    
    eps_2 = -3
    p = draw_lens([50,150,0,500])
    for x in range(1,domain_x):
        epsilon[:,x] = plano_convex(x,np.arange(0,domain_y),p)*eps_2 +4

    epsilon[:,0:50]=4


data = []
data_src = []
fig = plt.figure( figsize=(8,8) )


if __name__=='__main__':
    
    print("Start")
    
    #define optics
    
    sim_optics2()
    
    

    
    im0 = plt.imshow(epsilon,cmap='binary',vmin=1,vmax=10)
    im = plt.imshow(Ez,cmap='bwr',vmin=-1,vmax=1,alpha=0.4)
    # plt.gcf().set_size_inches(14,7)
    # plt.ylim([0,350])
    
    plot_tstep = 10
    t_end = 700
   
    ani = FuncAnimation(fig, update_FDTD_2D, frames=np.arange(0, t_end, plot_tstep),
                    init_func=init, blit=True,interval=10,fargs=[plot_tstep],repeat = False,save_count=70,)
    
    # plt.colorbar()  
    # ani.save("FDTD_2D_lens.gif", writer='imagemagick',fps=60)
            