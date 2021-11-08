""" FDTD simulation
PHYS 611

Hasith Perera
"""


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

domain_x =5000;

Ez = np.zeros([domain_x,1])
Hy = np.zeros([domain_x,1])

Jz = np.zeros([domain_x,1])


epsilon_r = np.ones([domain_x,1])

#define medium
epsilon_r[2000:3000] = 4


#source location
src_loc = 1;

time_end = 5000;
dt = 1
dx = 1
mu_0 = 1
epsilon_0 = 1


gauss_sigma = .01;
gauss_mu = 100

detector_log = []

detector_locs = [100,250,350]

#define realtime animation
animation = 0
 

def src_gauss(time,mu,sigma):
    '''Generate a gaussian source'''
    
    return np.exp(-sigma*np.power(time-mu,2))

def src_step(time,start_t,width):
    
    if time>start_t:
        if time<start_t+width:
            return 1
        
    return 0
    
def src_sin(time,A,f):
    '''src_sin(time,A,F)
    '''
    
    # #sudden cut off gives discontinous edges
    # if (2*f*time > 2*np.pi):
    #     return 0
    
    return A*np.sin(2*f*time)


def dump_detector_data(Ez,time,detectors):
    
    detector_log.append(Ez[detectors])
    print(Ez[detectors])
    
    print("{},".format(time),end='')
    for detector in detectors:   
       
        print("{:.4e},".format(Ez[detector][0]),end='')
    print('')

    
def write_to_file(Ez,time,detectors,file='log.dat'):
    '''log to file at intervals'''
    print(detectors)
    
    with open(file,'a+') as fp:
        fp.write("{},".format(time))
        for detector in detectors:       
            
            fp.wrtie("{},".format(Ez[detector][1]))
        fp.write('\n')
    
    
    
def FDTD_update(time):
    
    # #inject a source to Ez[0]      
    # Ez[100] = np.exp(-gauss_sigma*np.power(i-gauss_mu,2))
    
    # #gaussian source
    # Jz[src_loc] = src_gauss(time, gauss_mu, gauss_sigma)
    
    #sine wave
    Jz[src_loc] = src_sin(time, 1, .01)
    
    #probe values
    # dump_detector_data(Ez,time,detector_locs)

    #update eq from ampher's law
    Ez[1:len(Ez)] = Ez[1:len(Ez)]+(dt/(dx*epsilon_0*epsilon_r[1:len(Ez)])) * (Hy[1:len(Hy)]-Hy[0:len(Hy)-1])
    
    #current term
    Ez[1:len(Ez)] = Ez[1:len(Ez)] + dt/(epsilon_0*epsilon_r[1:len(Ez)]) * Jz[1:len(Ez)]

    #PBC to the right
    Hy[len(Hy)-1]=Hy[len(Hy)-2]
 
    #update faraday's law
    Hy[0:len(Hy)-1] = Hy[0:len(Hy)-1]+(dt/dx*mu_0)*Ez[1:len(Ez)]-Ez[0:len(Ez)-1]    
        
    #PBC for the Left 
    Ez[0]=Ez[1]   
    print(time)
    
    
def fdtd_animate(time):
    
    FDTD_update(time)
    ln.set_data(np.arange(0,domain_x),Ez)
    return ln,


def init():
    ax.set_xlim(0, domain_x)
    ax.set_ylim(-1, 1)
    # Ez = np.zeros([domain_x,1])
    print("Clear Ez")
    return ln,

if __name__=='__main__':
    
    
    print("Start FDTD 1D simulation")     
    if animation:
        print("With Realtime animation")
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = plt.plot([], [], 'r.')
    
        ani = FuncAnimation(fig, fdtd_animate, frames=np.linspace(1, time_end, time_end),
                    init_func=init, blit=True,interval=10,repeat=False)
    
        plt.show()
        print("Exit")
    else:
        for i in range(0,time_end):
            FDTD_update(i)
            
    
    # log = np.array(detector_log)
    # plt.plot(log[:,0])
    # plt.plot(log[:,1])
    # plt.plot(log[:,2])
    