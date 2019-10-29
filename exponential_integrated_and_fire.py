#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:37:43 2019

@author: anaclaudiacosta
BASED ON
https://github.com/computational-neuroscience/Computational-Neuroscience-UW/blob/master/week-05/Simplified%20models.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt

R = 40.0  # Mohm
C = 1.0   # nF
tau = R*C # Membrane time constant

V_thres    = -40.0  # Threshold potential [mV]
V_max      =  40.0 # Spike potential     [mV]
V_reset    = -70.0 # Reset potential     [mV]
V_eq       = -60.0 # Resting potential   [mV]
max_refrac = 3     # time steps in refractory period

# define time variable and voltage trace
dt = 1     
t = np.arange(0,500,dt)
V_trace = np.zeros(len(t))
V_trace[0] = V_eq

# define a current trace 
"""
The following code defines the different parameter and prepares an additional 
current variable that is used to inject a 2.0 mA current 
between 200 and 400 ms."""
I_max = 1.0
I_trace = np.zeros(len(t))
I_trace[200:400] = I_max

"""
With the parameters defined we are now ready to run the simuluation. 
At each time step we first check if the neuron is in the refractory period. 
If so we keep the membrane potential fixed. In the case that the neuron is not
in the refractory period we update the membrane potential according to the 
differential equation. Finally, we check if the membrane potential has crossed
the threshold value and emit a spike if that is the case
"""
refrac = 0  # If larger than 0, the neuron is in the refractory period
for i in range(1,len(t)):
    # 1. Check if the neuron is in the refractory period
    if refrac > 0:
        V_trace[i] = V_reset
        refrac -= 1
        continue
    
    # 2. Update the membrane potential
    V  = V_trace[i-1]
    dv = (-(V - V_eq) + R*I_trace[i])/tau
    V_trace[i] = V + dv*dt
    
    # 3. Check for spikes
    if V_trace[i] > V_thres:
        V_trace[i] = V_max
        refrac = max_refrac

plt.plot(t,V_trace)
plt.axhline(V_thres,linestyle = '--',color='k')
plt.gca().set(xlabel='Time [ms]',ylabel='Membrane potential [mV]')
plt.title('Integrate and fire neuron')

"""
When the equations are relatively simple, it is possible to construct a phase 
plane for a given differential equation. For a one-dimensional differential 
equation this is simply a plot of the value of the membrane potential and the 
corresponding value of the derivative. In the following code we do this for 
the case where there is no external input and for the case where there is a 
positive input.
"""
V = np.arange(-80,60,1)
f_V = -(V-V_eq)/tau

plt.plot(V,f_V,'b')
plt.plot(V,f_V + R*I_max/tau,'g')
plt.axhline(0,color='k')
plt.axvline(0,color='k')
plt.plot(V_eq,0,'og')
plt.plot(V_eq + R*I_max,0,'og')
plt.plot(V_thres,0,'or')
plt.plot(V_reset,0,'ok')
plt.title('Phase plane')
plt.legend(['I = 0','I = ' + str(I_max)])
plt.gca().set(xlabel = 'Potential [mV]', ylabel ='dV/dt')


delta_T = 7.0

# Prepare voltage trace
t = np.arange(0,200,dt)
V_trace = np.zeros((2,len(t)))
V_trace[:,0] = V_eq

# define a current trace 
I_max = [1.49,1.50]
I_trace = np.zeros(len(t))

for i in range(2):
    I_trace[50:95] = I_max[i]
    refrac = 0 
    for j in range(1,len(t)):
        if refrac > 0:
            V_trace[i,j] = V_reset
            refrac -= 1
            continue
    
        V  = V_trace[i,j-1]
        dv = (-(V - V_eq) + np.exp((V-V_thres)/delta_T) + R*I_trace[j])/tau
        V_trace[i,j] = V + dv*dt
        
        if V_trace[i,j] > V_max:
            V_trace[i,j] = V_max
            refrac = max_refrac
        
# Plot voltage traces
plt.plot(t,V_trace[0,:],'--k')
plt.plot(t,V_trace[1,:],'k')
plt.title('Voltage trace')
plt.axvline(95,color='r')
plt.gca().set(ylabel = 'Potential [mV]', xlabel ='Time [ms]')

V = np.arange(-80.0,60.0,1.0)
f_V = (-(V-V_eq) + np.exp((V - V_thres)/delta_T))/tau

fp_1 = -59.94
fp_2 = -13.05

plt.subplot(1,2,2)
plt.axhline(0,color='k')
plt.axvline(0,color='k')
plt.plot(V,f_V)
plt.plot(V,f_V + R*I_max[0]/tau)
plt.plot(fp_1,0,'og')
plt.plot(fp_2,0,'or')
plt.ylim(-5,10)
plt.title('Phase plane')
plt.gca().set(xlabel = 'Potential [mV]', ylabel ='dV/dt')

