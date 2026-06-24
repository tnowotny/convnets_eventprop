import numpy as np
import matplotlib.pyplot as plt

def gridlines(ax, hti, wdi, split, extra_lines = None):
    k= 1
    ymn= 0
    ymx= np.prod(split[:hti])
    xmn= 0
    xmx= np.prod(split[hti:])
    for i in range(hti):
        k*= split[i]
        d= ymx // k
        lnsy= [ [j*d-0.5, j*d-0.5] for j in range(1,ymx // d) ]
        lnsx= [ [xmn-0.5, xmx-0.5] for j in range(1,ymx // d) ]
        ax.plot(np.array(lnsx).T,np.array(lnsy).T,'w',lw= hti-i)   
    k=1
    for j in range(hti, len(split)):
        k*= split[j]
        d= xmx // k
        lnsx= [ [i*d-0.5, i*d-0.5] for i in range(1,xmx // d) ]
        lnsy= [ [ymn-0.5, ymx-0.5] for i in range(1,xmx // d) ]
        ax.plot(np.array(lnsx).T,np.array(lnsy).T,'w',lw= len(split)-j)

    if extra_lines is not None:
        lnsy= []
        lnsx = []
        if "x" in extra_lines:
            for j in extra_lines["x"]["j"]:
                lnsx.append([j*d-0.5, j*d-0.5])
                lnsy.append([ymn-0.5, ymx-0.5])
            ax.plot(np.array(lnsx).T,np.array(lnsy).T,'w',lw= extra_lines["x"]["w"])   
        lnsy= []
        lnsx = []
        if "y" in extra_lines:
            for j in extra_lines["y"]["j"]:
                lnsy.append([j*d-0.5, j*d-0.5])
                lnsx.append([xmn-0.5, xmx-0.5])
            ax.plot(np.array(lnsx).T,np.array(lnsy).T,'w',lw= extra_lines["y"]["w"])   
