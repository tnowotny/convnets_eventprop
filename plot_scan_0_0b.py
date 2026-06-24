import numpy as np
import matplotlib.pyplot as plt
import sys
import plot_utils

shape1 = [ 2, 2, 5, 5 ]
shape2 = [ 2, 2, 5, 5 ]
full_shape = [ 2, 2, 2, 5, 5 ]
plotshape = [ 8, 25 ]

plotcols = [ -2, -1 ]
name1 = "scan_OMNI_0/J0"
name2 = "scan_OMNI_1/J1"

all_res1 = []
for i in range(np.prod(shape1)):
    all_res1.append(np.loadtxt(f"{name1}_{i}_results.txt"))

all_res2 = []
for i in range(np.prod(shape2)):
    all_res2.append(np.loadtxt(f"{name2}_{i}_results.txt"))

for c in plotcols:
    ad = []
    for shape, all_res in zip([shape1,shape2],[all_res1,all_res2]): 
        d = np.zeros(shape)
        it = np.nditer(d, flags=["c_index"], op_flags=["writeonly"])
        with it:
            while not it.finished:
                it[0] = np.max(all_res[it.index][:,c])
                it.iternext()
        ad.append(d)
    dd = np.asarray([ ad[0], ad[1] ])
    dd = np.reshape(dd, plotshape)
    plt.figure(figsize=(10,2.9))
    plt.imshow(dd)
    plot_utils.gridlines(plt.gca(),3,2,full_shape)
    plt.colorbar()
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"scan_0_1_overview_col{c}.pdf")
    print(f"max: {np.max(dd.flatten())} at index {np.argmax(dd)}")
plt.show()
    
