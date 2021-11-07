""" FDTD simulation
PHYS 611

Hasith Perera
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

domain_x =500;

Ez = np.zeros([domain_x,1])
Hy = np.zeros([domain_x,1])

Jz = np.zeros([domain_x,1])
epsilon_r = np.ones([domain_x,1])

epsilon_r[250:] = 8



#source location
src_loc = 1;

time_end = 200;
dt = 1
dx = 1
mu_0 = 1
epsilon_0 = 1


gauss_sigma = .01;
gauss_mu = 100


detector = []

detector_location_x = 50
 

def src_gauss(time,mu,sigma):
    return np.exp(-sigma*np.power(time-mu,2))

def src_step(time,start_t,width):
    
    if time>start_t:
        if time<start_t+width:
            return 1
        
    return 0
    
def src_sin(time,A,f):
    if (2*f*time > 4*np.pi):
        return 0
    return A*np.sin(2*f*time)
    

def FDTD_update(time):
    
    #inject a source to Ez[0]      
    #Ez[100] = np.exp(-gauss_sigma*np.power(i-gauss_mu,2))
    
    # Jz[src_loc] = src_gauss(time, gauss_mu, gauss_sigma)
    Jz[src_loc] = src_sin(time, 1, .02)
    # print(Jz[src_loc])
    #probe values
    detector.append(Ez[detector_location_x][0])

    
    #update faraday's law
    Hy[0:len(Hy)-1] = Hy[0:len(Hy)-1]+(dt/dx*mu_0)*Ez[1:len(Ez)]-Ez[0:len(Ez)-1]
    
    
    
    
    #PBC for the Left 
    Ez[0]=Ez[1]
    
    #update eq from ampher's law
    Ez[1:len(Ez)] = Ez[1:len(Ez)]+(dt/(dx*epsilon_0*epsilon_r[1:len(Ez)])) * (Hy[1:len(Hy)]-Hy[0:len(Hy)-1])
    
    #current term
    Ez[1:len(Ez)] = Ez[1:len(Ez)] + dt/(epsilon_0*epsilon_r[1:len(Ez)]) * Jz[1:len(Ez)]
    
    #PBC to the right
    Hy[len(Hy)-1]=Hy[len(Hy)-2]
    
    
    ## adding propergation in dielectric medium
    
    
    
    ln.set_data(np.arange(0,domain_x),Ez)
    
    return ln,


def init():
    ax.set_xlim(0, domain_x)
    ax.set_ylim(-1, 1)
    Ez = np.zeros([domain_x,1])
    print("Clear Ez")
    return ln,

if __name__=='__main__':
    
    print("Start")    
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')
    
    
    ani = FuncAnimation(fig, FDTD_update, frames=np.linspace(1, 600, 600),
                    init_func=init, blit=True,interval=10,repeat=False)
    plt.show()
    print("Exit")
    
