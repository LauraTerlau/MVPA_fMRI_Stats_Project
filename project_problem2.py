"""
STATS PROJECT 2 -- Minimizing Cost Function with different lambda & q parameter

Created on Tue Aug  3 11:57:37 2021

@author: tydingsmcclary
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
from sklearn import linear_model
import seaborn as sns; sns.set('paper')
import os

os.chdir('/Users/tydingsmcclary/Documents/SCAN MASTER/Semester2/Stats/Part2/Project/')


## Setting up Data
np.random.seed(10)
x = np.linspace(-3, 3, 10)
y = 1.5 * x + np.random.normal(0,1.5,10)
y_ideal = 1.5 * x
y_less = 1 * x
x_new_points = np.linspace(-3.5, 3.5, 10)
y_new_points = 0.85 * x_new_points + np.random.normal(0,1.5,10)
x = x[:, np.newaxis]

#plot1
plt.rcParams['figure.figsize'] = [12, 8]
plt.plot(x, y, 'ko')
plt.plot(x, y_ideal, color='black', linewidth=2.5)
plt.xlim((-4.5,4.5))
plt.savefig('lina.png')
plt.show()

#plot2
plt.rcParams['figure.figsize'] = [12, 8]
plt.plot(x, y, 'ko')
plt.plot(np.linspace(-3.5, 3.5, 10), y_new_points, 'cs')
plt.plot(x_new_points, y_less, color='cyan', linewidth=2.5)
plt.xlim((-4.5,4.5))
plt.savefig('linb.png')
plt.show()

plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize = 20)
plt.figure(figsize = (20,10))
lin_reg_slopes = []
lin_reg_rsq = []
for s in np.linspace(-1, 2, 200):
    y_hat = s*x.T[0]
    lin_reg_slopes.append(s)
    lin_reg_rsq.append(1/2 * sum((y-y_hat)**2))

plt.plot(lin_reg_slopes, lin_reg_rsq, color='blue', linewidth=3)
plt.plot(lin_reg_slopes[np.where(lin_reg_rsq == min(lin_reg_rsq))[0][0]], min(lin_reg_rsq), 'bo')
plt.xlabel('Slope Value')
plt.ylabel('RSS/2')
plt.show()

#defining lambda 
lmbdt = [0, 10, 20, 40, 80, 160]
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']

#Lasso
plt.rc('axes', titlesize = 25)
plt.rc('axes', labelsize=30)
plt.rc('legend', fontsize = 25)
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
plt.figure(figsize = (25,17.5))
for i in range(0,len(lmbdt)):
    lasso_slopes = []
    lasso_rsq = []
    for s in np.linspace(-1, 2, 200):
        y_hat = s*x.T[0]
        lasso_slopes.append(s)
        lasso_rsq.append(0.5 * sum((y-y_hat)**2) + (lmbdt[i]/2)*np.abs(s))
        
    plt.plot(lasso_slopes, lasso_rsq, color=colors[i], linewidth=3, label = 'lambda = '+ str(lmbdt[i]))
    plt.plot(lasso_slopes[np.where(lasso_rsq == min(lasso_rsq))[0][0]], min(lasso_rsq),'o' +  colors[i][0], ms = 12, mfc = colors[i])
plt.xlabel('Slope Value')
plt.ylabel('E(w) [with q = 1])')
plt.title('Different Optimal Slope Values depending on Lambda for Lasso Regression [q = 1]')
plt.legend()
plt.savefig('Lasso.png')
plt.show()
    

#Ridge
plt.rc('axes', titlesize = 25)
plt.rc('axes', labelsize = 30)
plt.rc('legend', fontsize = 25)
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
plt.figure(figsize = (25,17.5))
for i in range(0,len(lmbdt)):
    ridge_slopes = []
    ridge_rsq = []
    for s in np.linspace(-1, 2, 200):
        y_hat = s*x.T[0]
        ridge_slopes.append(s)
        ridge_rsq.append(0.5 * sum((y-y_hat)**2) + (lmbdt[i]/2)*(s**2))
    
    plt.plot(ridge_slopes, ridge_rsq, color=colors[i], linewidth=3, label = 'lambda = '+ str(lmbdt[i]))
    plt.plot(ridge_slopes[np.where(ridge_rsq == min(ridge_rsq))[0][0]], min(ridge_rsq),'o' +  colors[i][0], ms = 12, mfc = colors[i])

plt.xlabel('Slope Value')
plt.ylabel('E(w) [with q = 2]')
plt.title('Different Optimal Slope Values depending on Lambda for Ridge Regression [q = 2]')
plt.legend()
plt.savefig('Ridge.png')
plt.show()







