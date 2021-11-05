# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:30:27 2021

@author: HasithPerera
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import time

domain_x = 200;
domain_y = 100;

src_loc = [0,0]


Ez = np.zeros([domain_x,domain_y])


Hx = np.zeros([domain_x,domain_y])
Hy = np.zeros([domain_x,domain_y])

dy = 1
dx = 1
dz = 1.;
mu = 1;
epsilon = 1;
dt = 1;



def src_gaussian_2D(x,mu=50,sig=15):   
    return np.exp(-1*(np.power(x-mu,2))/np.power(sig,2))
    

def update_FDTD_2D(time):
    print(time)
    
    # print(time,src_gaussian_2D(time),Ez[50,40])
    
    Ez[50,50] = src_gaussian_2D(time)
    
    # data.append(Ez[40,50])
    # data_src.append(Ez[50,1])
    #equation 1
    Hx[:,0:domain_y-1] = Hx[:,0:domain_y-1] + (Ez[:,1:domain_y]-Ez[:,0:domain_y-1])*(dt/(mu*dy))
    
    #equation 2
    Hy[0:domain_x-1,:] = Hy[0:domain_x-1,:] - (Ez[1:domain_x,:]-Ez[0:domain_x-1,:])*(dt/(dx*mu))
    
    #equation 3   
    Ez[0:domain_x-1,0:domain_y-1] = ( Ez[0:domain_x-1,0:domain_y-1] 
                                      - (dt/(epsilon*dy))*(Hx[0:domain_x-1,1:domain_y] 
                                                          - Hx[0:domain_x-1,0:domain_y-1]) 
                                      + (dt/(epsilon*dx))*(Hy[1:domain_x,0:domain_y-1]
                                                          -Hy[0:domain_x-1,0:domain_y-1]))
    if time%50 == 0:
        print("Plot")
        plt.imshow(Ez)
        plt.show()
        
    
    # im.set_array(Ez/np.max(np.max(Ez)))
    
    # return [im]


def init():
    return [im]


fig, ax = plt.subplots()
im = plt.imshow(Ez)
data = []
data_src = []

if __name__=='__main__':
    
    print("Start")
    
    
   
    # ani = FuncAnimation(fig, update_FDTD_2D, frames=np.linspace(0, 100, 101),
                    # init_func=init, blit=True,interval=100)
    
    
    # # plt.show()
    for i in range(1,300):
        # print(i)
        update_FDTD_2D(i)
        #print(Ez[50])
        
    # plt.clf()
    # plt.plot(data)
    # plt.plot(data_src,'o')
    # plt.show()
    
    # xx = np.arange(1,100)
    # plt.clf()
    # plt.plot(xx,src_gaussian_2D(xx))
    
    