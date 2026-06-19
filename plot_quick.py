import numpy as np
import matplotlib.pyplot as plt
import sys

d = []
for i in range(len(sys.argv)-1):
    d.append(np.loadtxt(sys.argv[i+1]))

print(len(d))
Nplots = d[0].shape[1]
if Nplots > 1:
    ploth = int(np.sqrt(Nplots))
    plotw = (Nplots-1)//ploth + 1
    fig,ax = plt.subplots(ploth,plotw,sharex=True)
    twoD = ploth > 1
    oneD = True
else:
    ploth = 1
    plotw = 1
    fig = plt.figure()
    ax = plt.gca()
    twoD = False
    oneD = False
    

for i in range(ploth):
    for j in range(plotw):
        id = i*plotw+j
        if twoD:
            the_ax = ax[i,j]
        elif oneD:
            the_ax = ax[j]
        else:
            the_ax = ax
        if id < Nplots:
            for pl in range(len(d)):
                the_ax.plot(d[pl][:,id])
        else:
            the_ax.set_visible(False)
    
plt.show()
