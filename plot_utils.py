import numpy as np
import matplotlib.pyplot as plt
import scipy
import json

def plot_with_regress(ax, x, y, clr, lims):
    lims = np.asarray(lims)
    ax.scatter(x,y,s=5.0,c=clr)
    #ax.set_xlim(lims[0,:])
    #ax.set_ylim(lims[1,:])
    regr = scipy.stats.linregress(np.squeeze(x),np.squeeze(y))
    print(x)
    print(y)
    print(f"{regr.slope} * x + {regr.intercept}, R^2 = {regr.rvalue**2}")
    lx = np.array(lims[0,:])
    ly = regr.slope*lx+regr.intercept
    ax.plot(lx,ly,clr)

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


def plot_comparison(afs,exp_name,the_col):
    """
    plot comparisons between fews shot data (afs) and pretraining data.
    """
    base = "scan_OMNI_0/J0_"
    best_val = []
    last_val = []
    sparse = []
    rot = []
    for n in range(100):
        fname = base+str(n)+"_results.txt"
        d = np.loadtxt(fname)
        best_val.append(np.max(d[:,-1]))
        last_val.append(d[-1,-1])
        sparse.append(np.mean(d[-1,:-2:4]))
        with open(base+str(n)+".json","r") as f:
            p = json.load(f)
        rot.append(p["TRAINING_ROTATION"])

    best_val = np.asarray(best_val)
    last_val = np.asarray(last_val)
    sparse = np.asarray(sparse)
    rot = np.asarray(rot)

    # exclude any runs where the final results were bad (unstable training) (bottom 10%)
    bv75= np.percentile(last_val[rot],15)
    bv75b= np.percentile(last_val[np.logical_not(rot)],15)
    bvr75 = last_val[rot] >= bv75
    bvnr75 = last_val[np.logical_not(rot)] >= bv75b

    best_val_r = best_val[rot][bvr75]
    best_val_nr = best_val[np.logical_not(rot)][bvnr75]
    sparse_r = sparse[rot][bvr75]
    sparse_nr = sparse[np.logical_not(rot)][bvnr75]

    #print(bv75)
    #print(bv75b)
    abvrot = np.argmax(best_val_r)
    abvnr = np.argmax(best_val_nr)

    for way in [ 5, 20 ]:
        for shot in [ 1, 5]:
            fix,ax = plt.subplots(1,3,figsize=(10,3),sharey=True)
            x = afs[np.logical_and(afs[:,0] == way, afs[:,1] == shot),the_col]
            #print(x)
            xr = x[rot][bvr75]
            xnr = x[np.logical_not(rot)][bvnr75]
            #print(f"Max validation (rot): {np.max(best_val_r)} at {abvrot}, {way}-way, {shot}-shot test at {abvrot}: {xr[abvrot]}")
            #print(f"Max validation (non-rot): {np.max(best_val_nr)} at {abvnr}, {way}-way, {shot}-shot test at {abvnr}: {xnr[abvnr]}")
            plot_with_regress(ax[0],sparse_r,best_val_r,'C0',[[0.1,0.9],[0.7,1.0]])
            plot_with_regress(ax[0],sparse_nr,best_val_nr,'C1',[[0.1,0.9],[0.5,1.0]])
            plot_with_regress(ax[1],sparse_r,xr,'C0',[[0.1,0.9],[0.1,1.0]])
            plot_with_regress(ax[1],sparse_nr,xnr,'C1',[[0.1,0.9],[0.1,1.0]])
            plot_with_regress(ax[2],best_val_r,xr,'C0',[[0.1,1.0],[0.1,1.0]])
            plot_with_regress(ax[2],best_val_nr,xnr,'C1',[[0.1,1.0],[0.1,1.0]])
            ax[0].set_xlabel("spikes per neuron per trial")
            ax[1].set_xlabel("spikes per neuron per trial")
            ax[2].set_xlabel("validation accuracy pre-training")
            ax[0].set_ylabel("validation accuracy pre-training")
            ax[1].set_ylabel(f"{way}-way, {shot}-shot test accuracy")
            ax[2].set_ylabel(f"{way}-way, {shot}-shot test accuracy")
            for i in range(3):
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(f"{base}{exp_name}_{way}-way_{shot}-shot.png")
    plt.show()

def plot_alone(afs,asp,exp_name,afs_col,asp_col):
    """
    plot direct "fromscratch" few shot data (afs) versus sparsity data (asp)
    """

    base = "scan_OMNI_0/J0_"
    for way in [ 5, 20 ]:
        for shot in [ 1, 5]:
            plt.figure()
            x = asp[np.logical_and(asp[:,0] == way, asp[:,1] == shot),asp_col]
            y = afs[np.logical_and(afs[:,0] == way, afs[:,1] == shot),afs_col]
            bv75= np.percentile(y,15)
            idx = y > 1.0/way*1.2
            yr = y[idx]
            xr = x[idx]
            plot_with_regress(plt.gca(),xr,yr,'C0',[[0,5],[0,1]])
            plt.xlabel("spikes per neuron per trial")
            plt.ylabel(f"{way}-way, {shot}-shot test accuracy")
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(f"{base}{exp_name}_{way}-way_{shot}-shot.png")
    plt.show()
