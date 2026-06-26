import matplotlib.pyplot as plt
import numpy as np
import json
import scipy

def plot_with_regress(ax, x, y, clr, lims):
    lims = np.asarray(lims)
    ax.scatter(x,y,s=5.0,c=clr)
    ax.set_xlim(lims[0,:])
    ax.set_ylim(lims[1,:])
    regr = scipy.stats.linregress(np.squeeze(x),np.squeeze(y))
    print(f"{regr.slope} * x + {regr.intercept}, R^2 = {regr.rvalue**2}")
    lx = np.array(lims[0,:])
    ly = regr.slope*lx+regr.intercept
    ax.plot(lx,ly,clr)


name_ext = ["", "e", "f", "g", "h", "i", "j", "k"]
base = "scan_OMNI_0/J0_"

afs = []
best_val = []
last_val = []
sparse = []
rot = []
for n in range(100):
    fname = base+str(n)+"_fewshot_results.txt"
    d = np.loadtxt(fname)
    afs.append(d)
    fname = base+str(n)+"_results.txt"
    d = np.loadtxt(fname)
    best_val.append(np.max(d[:,-1]))
    last_val.append(d[-1,-1])
    sparse.append(np.mean(d[:-2:4,-1]))
    with open(base+str(n)+".json","r") as f:
        p = json.load(f)
    rot.append(p["TRAINING_ROTATION"])

afs = np.asarray(afs)
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

print(bv75)
print(bv75b)
abvrot = np.argmax(best_val_r)
abvnr = np.argmax(best_val_nr)


for way in [ 5, 20 ]:
    for shot in [ 1, 5]:
        fix,ax = plt.subplots(1,3,figsize=(10,3),sharey=True)
        x = afs[:,np.logical_and(afs[n][:,0] == way, afs[n][:,1] == shot),2]
        xr = x[rot][bvr75]
        xnr = x[np.logical_not(rot)][bvnr75]
        print(f"Max validation (rot): {np.max(best_val_r)} at {abvrot}, {way}-way, {shot}-shot test at {abvrot}: {xr[abvrot]}")
        print(f"Max validation (non-rot): {np.max(best_val_nr)} at {abvnr}, {way}-way, {shot}-shot test at {abvnr}: {xnr[abvnr]}")
        plot_with_regress(ax[0],sparse_r,best_val_r,'C0',[[0.1,0.9],[0.7,1.0]])
        plot_with_regress(ax[0],sparse_nr,best_val_nr,'C1',[[0.1,0.9],[0.5,1.0]])
        plot_with_regress(ax[1],sparse_r,xr,'C0',[[0.1,0.9],[0.5,1.0]])
        plot_with_regress(ax[1],sparse_nr,xnr,'C1',[[0.1,0.9],[0.5,1.0]])
        plot_with_regress(ax[2],best_val_r,xr,'C0',[[0.5,1.0],[0.5,1.0]])
        plot_with_regress(ax[2],best_val_nr,xnr,'C1',[[0.5,1.0],[0.5,1.0]])
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
        plt.savefig(f"{base}fewshot_{way}-way_{shot}-shot.png")
plt.show()
