import matplotlib.pyplot as plt
import numpy as np
import json
from plot_utils import plot_with_regress, plot_comparison


base = "scan_OMNI_0/J0_"

afs = []
for n in range(100):
    fname = base+str(n)+"_fewshot_results.txt"
    d = np.loadtxt(fname)
    afs.append(d)

afs = np.asarray(afs)
afs = np.reshape(afs,(afs.shape[0]*afs.shape[1],-1))
print(afs.shape)
plot_comparison(afs, "prototype_Euclid",2)
plot_comparison(afs, "prototype_Cosine",4)
