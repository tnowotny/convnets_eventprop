import numpy as np
import matplotlib.pyplot as plt
import sys

d = np.loadtxt(sys.argv[1])
Nplots = d.shape[1]
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
            the_ax.plot(d[:,id])
        else:
            the_ax.set_visible(False)
    
plt.show()
