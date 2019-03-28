#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:36:51 2019

Implementação da aula de Gradiente Descent em:
    https://www.youtube.com/watch?v=XdM6ER7zTLk

@author: pisciana
"""

from numpy import *
import matplotlib.pyplot as plt


def computer_error_for_given_points(b, m, points):
    
    total_error = 0    
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]        
        total_error += (y - (m * x + b)) ** 2    
        
    return (total_error / float(len(points)))        
          

def step_gradient(b, m, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))
    
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        b_gradient += -(2/N) * (y - (m*x +b)) 
        m_gradient += -(2/N) * x * (y - (m*x +b))
    
    new_b = b - (learning_rate * b_gradient)
    new_m = m - (learning_rate * m_gradient)
        
    return [new_b, new_m]
    


def gradient_descent_runner(points, initial_b, initial_m, learning_rate,num_interations):
    m = initial_m
    b = initial_b
    
    for i in range (num_interations):
        m, b = step_gradient(b, m, array(points), learning_rate)
    
    return[b,m]
    

def run():
   points = genfromtxt("data.csv", delimiter=",")
   
   #hyperparameter
   learning_rate = 0.0003
   
   #y = mx + b   - slope formula
   initial_m = 0 #inclinação da reta
   initial_b = 0 # ponto em que Y intercepta
   num_iterations = 2000
   
   #retorna os valores otimos de b e m
   [b,m] = gradient_descent_runner(
           points, initial_b, initial_m, learning_rate,num_iterations)
   
   print(b)
   print(m)
   print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, computer_error_for_given_points(initial_b, initial_m, points))
   print "Running..."
   [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
   print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, computer_error_for_given_points(b, m, points))
   
   x = linspace(10.,80.)
   fig,ax = plt.subplots()
   ax.plot(x,m *x+b)
   ax.set_xlim((10.,80.))
   plt.scatter(points[:,0], points[:,1])
   plt.show

if __name__ == '__main__':
    run()
    