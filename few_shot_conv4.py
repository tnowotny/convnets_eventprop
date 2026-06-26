import matplotlib.pyplot as plt
import numpy as np
import sys
import json

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, ConnVarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Conv2D, Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire,  LatencyInput
from ml_genn.optimisers import Adam
from ml_genn.optimisers import Adam
from ml_genn.regularisers import SpikeCount
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from time import perf_counter

import utils

p = {
    "NUM_OUTPUT": 964,
    "BATCH_SIZE": 32,
    "EXAMPLE_TIME": 25.0,
    "DT": 1.0,
    "KERNEL_PROFILING": False,
    "HIDDEN_LAYERS": 4,
    "NUM_HIDDEN": [ 64, 64, 64, 64 ],
    "KERNEL_SZ": [ 3, 3, 3, 3 ],
    "CONV_STRIDES": [ 2, 1, 1, 1 ],
    "HID_MEAN": [ 1.0, 0.0, 0.0, 0.0 ],
    "HID_SD": [3.0, 1.0, 1.0, 1.0 ],
    "N_WAY": 20,
    "K_SHOT": 5,
    "NAME": "scan_OMNI_0/J0_16",
    "TRAINING_ROTATION": False
}

if len(sys.argv) > 1:
    fname= f"{sys.argv[1]}.json"
    with open(fname,"r") as f:
        p0= json.load(f)

    for (name,value) in p0.items():
        p[name]= value

if p["TRAINING_ROTATION"]:
    p["NUM_OUTPUT"] *= 4

p["EMBEDDING_NAME"] = p["NAME"]+"_checkpoints"

images, test_labels, alph_ids, char_ids = utils.load_omniglot("test")
test_img = utils.rescale(images)

serialiser = Numpy(p["EMBEDDING_NAME"])
print(f"serialiser: {p['EMBEDDING_NAME']}")
network = SequentialNetwork()

with network:
    # Populations
    input = InputLayer(LatencyInput("linear", p["EXAMPLE_TIME"] - (2.0 * p["DT"]), 2.0 * p["DT"], 1, False),
                       (28, 28, 1), name="input",record_spikes= True)   
    hidden = []
    for nh in range(p["HIDDEN_LAYERS"]):
        initial_hidden_weight = Normal(mean= p["HID_MEAN"][nh], sd= p["HID_SD"][nh])
        flatten =  nh == p["HIDDEN_LAYERS"]-1 
        hidden.append(Layer(Conv2D(initial_hidden_weight, p["NUM_HIDDEN"][nh],
                                   p["KERNEL_SZ"][nh], flatten, conv_strides=p["CONV_STRIDES"][nh]),
                                   LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                                   synapse=Exponential(5.0), name=f"hidden{nh}", record_spikes= True))
    output = Layer(Dense(Normal(mean=0.0, sd=0.1)),
                   LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),    
                   p["NUM_OUTPUT"], Exponential(5.0), name="output")

max_example_timesteps = int(np.ceil(p["EXAMPLE_TIME"] / p["DT"]))
network.load((0,), serialiser)
compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                             reset_in_syn_between_batches=True,
                             batch_size=p["BATCH_SIZE"])
compiled_net = compiler.compile(network)

embeddings = utils.extract_embeddings(compiled_net, input, output, test_img,"output",p["BATCH_SIZE"],"v")
embeddings = utils.l2_normalise(embeddings)
res = []
for i in range(1000):
    res.append(utils.run_episode(embeddings, test_labels, p["N_WAY"], p["K_SHOT"]))
mn = np.mean(res)
std = np.std(res)
print(f"Euclidean nearest centroid: {mn*100}+/-{std*100}% correct.")
print(f"output file: {p['NAME']+'_fewshot_results.txt'}")
f = open(p["NAME"]+"_fewshot_results.txt","a")
f.write(f"{p['N_WAY']} {p['K_SHOT']} {mn} {std} ")

#res = []
#for i in range(1000):
#    res.append(utils.run_episode_linear(embeddings, test_labels, p["N_WAY"], p["K_SHOT"]))
#mn = np.mean(res)
#std = np.std(res)
#print(f"Logistic regression: {mn*100}+/-{std*100}% correct.")
    
res = []
for i in range(1000):
    res.append(utils.run_episode_cosine(embeddings, test_labels, p["N_WAY"], p["K_SHOT"]))
mn = np.mean(res)
std = np.std(res)
print(f"Cosine nearest centroid: {mn*100}+/-{std*100}% correct.")
f.write(f"{mn} {std}\n")
f.close()
