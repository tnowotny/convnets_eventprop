import matplotlib.pyplot as plt
import numpy as np
import sys, json

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
    "NUM_INPUT": 784,
    "BATCH_SIZE": 2,
    "NUM_EPOCHS": 200,
    "EXAMPLE_TIME": 25.0,
    "DT": 1.0,
    "KERNEL_PROFILING": False,
    "PLOT_EPOCHS": 2000,
    "HIDDEN_LAYERS": 4,
    "NUM_HIDDEN": [ 64, 64, 64, 64 ],
    "KERNEL_SZ": [ 3, 3, 3, 3 ],
    "CONV_STRIDES": [ 2, 1, 1, 1 ],
    "HID_MEAN": [ 1.0, 0.0, 0.0, 0.0 ],
    "HID_SD": [3.0, 1.0, 1.0, 1.0 ],
    "LR": [5e-3, 5e-3, 5e-3, 5e-3, 2e-3 ],
    "ALPHABETS": None,
    "AUG": {"zoom": (0.8,1.2),
       "rotate": (-10.0,10.0),
       "shift": (-15, 15)
       },
    "SHOW_AUGMENTATION_EXAMPLE": False,
    "REG_STRENGTH":(1e-4,1e-6),
    "REG_TARGET": 0.1,
    "NAME": "scan_OMNI_0/J0_16",
    "SHUFFLE": False,
    "RECORD_CONFUSION": True,
    "HEADLESS": True,
    "TRAINING_ROTATION": True,
    "SIGNED": False,
    "NUM_REPS": 50,
    "N_WAY": 5,
    "K_SHOT": 5
}

if len(sys.argv) > 1:
    fname= f"{sys.argv[1]}.json"
    with open(fname,"r") as f:
        p0= json.load(f)

    for (name,value) in p0.items():
        p[name]= value

p["BATCH_SIZE"] = 2               # necessary as few-shot has very few examples and batch 0
                                  # doesn't learn in Eventprop
p["NAME"] = p["NAME"] + "_fromscratch"
serialiser2 = Numpy(f"{p['NAME']}_checkpoints")
print(p)
with open(f"{p['NAME']}_run.json","w") as f:
    json.dump(p,f,indent=4)

images, labels, alph_ids, char_ids = utils.load_omniglot("test") 
if p["SIGNED"]:
    images = np.minimum((images-127.5)*2,255)
    
network = SequentialNetwork()
NUM_OUTPUT = p["N_WAY"]

with network:
    # Populations
    input = InputLayer(LatencyInput("linear", p["EXAMPLE_TIME"] - (2.0 * p["DT"]), 2.0 * p["DT"], 1, p["SIGNED"]),
                       (28, 28, 1), name="input",record_spikes= True)   
    hidden = []
    for nh in range(p["HIDDEN_LAYERS"]):
        initial_hidden_weight = Normal(mean= p["HID_MEAN"][nh], sd= p["HID_SD"][nh])
        flatten =  nh == p["HIDDEN_LAYERS"]-1 
        hidden.append(Layer(Conv2D(initial_hidden_weight, p["NUM_HIDDEN"][nh],
                                   p["KERNEL_SZ"][nh], flatten, conv_strides=p["CONV_STRIDES"][nh]),
                                   LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                                   synapse=Exponential(5.0), name=f"hidden{nh}", record_spikes= True))
    output = Layer(Dense(Normal(mean=0.0, sd=0.0)),
                   LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),    
                   NUM_OUTPUT, Exponential(5.0), name="output")

max_example_timesteps = int(np.ceil(p["EXAMPLE_TIME"] / p["DT"]))
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             max_spikes = 15,
                             losses="sparse_categorical_crossentropy",
                             batch_size=p["BATCH_SIZE"], kernel_profiling=p["KERNEL_PROFILING"])

optimisers = {hidden[nh]: {"weight": Adam(p["LR"][nh])} for nh in range(p["HIDDEN_LAYERS"])}
optimisers[output] = {"weight": Adam(p["LR"][-1])}
Inferencecompiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                      reset_in_syn_between_batches=True,
                                      batch_size=p["BATCH_SIZE"])
all_test  = []
for rep in range(p["NUM_REPS"]):
    train_img, train_lbl, qry_img, qry_lbl = utils.sample_episode(images, labels, p["N_WAY"], p["K_SHOT"])
    qry_img = utils.rescale(qry_img)
    compiled_net = compiler.compile(network, name=p["NAME"], optimisers=optimisers, regularisers={"all_hidden_populations": SpikeCount(strength=p["REG_STRENGTH"],target=p["REG_TARGET"])})
    with compiled_net:
        start_time = perf_counter()
        callbacks = [ SpikeRecorder(input, "sin", example_filter= [ 1, 500, 1000, 1500 ])
        ]
        if p["RECORD_CONFUSION"]:
            callbacks.append(VarRecorder(output,"vAvg","outvavg"))
        for nh in range(p["HIDDEN_LAYERS"]):
            callbacks.append(SpikeRecorder(hidden[nh], f"shid{nh}",example_filter=[ 1, 500, 1000, 1500 ]))
        
        for e in range(p["NUM_EPOCHS"]):
            the_img = utils.augment(train_img, p["AUG"])
            the_img = utils.rescale(the_img)
            if not p["HEADLESS"] and p["SHOW_AUGMENTATION_EXAMPLE"]:
                for i in range(min(10,len(train_img))):
                    fig,ax = plt.subplots(1,2)
                    ax[0].imshow(train_img[i])
                    ax[1].imshow(the_img[i])
                plt.show()

            metrics,cb_data = compiled_net.train({input: the_img},
                                                 {output: train_lbl},
                                                 num_epochs=1, start_epoch=e,
                                                 shuffle=p["SHUFFLE"],
                                                 callbacks=callbacks)
            if not p["HEADLESS"] and e % p["PLOT_EPOCHS"] == 0:
                if p["RECORD_CONFUSION"]:
                    n = len(np.unique(train_lbl))
                    conf= np.zeros((n,n))
                    preds= utils.predictions_from_vAvg(cb_data["outvavg"])
                    for i,j in zip(train_lbl,preds):
                        conf[i,j] += 1
                    plt.figure()
                    plt.imshow(conf)
                    plt.gca().yaxis.set_inverted(False)
                w = utils.get_conn_var(compiled_net, hidden[0], "weight")
                fig0= utils.show_filters(w, (p["KERNEL_SZ"][0], p["KERNEL_SZ"][0]))
                grad = []
                utils.spike_raster(cb_data, f"sin")
                for nh in range(p["HIDDEN_LAYERS"]):
                    grad.append(utils.get_conn_var(compiled_net, hidden[nh], "weightGradient"))
                    utils.plot_var(grad[-1]) 
                    utils.spike_raster(cb_data, f"shid{nh}")
                plt.show()
            f = open(f"{p['NAME']}_trainresults.txt","a")
            f.write(f"{e} {p['N_WAY']} {p['k_SHOT']} ")
            for nh in range(p["HIDDEN_LAYERS"]):
                smean,ssig,sallmean,sallsig = utils.spike_stats(hidden[nh],cb_data,f"shid{nh}")
                f.write(f"{smean} {ssig} {sallmean} {sallsig} ")
            f.write(f"{metrics[output].result}\n")
            f.close()
        compiled_net.save((0,),serialiser2)
    network.load((0,), serialiser2)
    compiled_net = Inferencecompiler.compile(network,name=p["NAME"])
    with compiled_net:
        metrics, cb_data = compiled_net.evaluate({input: qry_img},{output: qry_lbl},callbacks=[])
    print(f"Test accuracy: {metrics[output].result}")
    all_test.append(metrics[output].result)
    end_time = perf_counter()
    print(f"Time = {end_time - start_time}s")

    if p["KERNEL_PROFILING"]:
        print(f"Neuron update time = {compiled_net.genn_model.neuron_update_time}")
        print(f"Presynaptic update time = {compiled_net.genn_model.presynaptic_update_time}")
        print(f"Gradient batch reduce time = {compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')}")
        print(f"Gradient learn time = {compiled_net.genn_model.get_custom_update_time('GradientLearn')}")
        print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")
        print(f"Softmax1 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax1')}")
        print(f"Softmax2 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax2')}")
        print(f"Softmax3 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax3')}")

mn = np.mean(all_test)
std = np.std(all_test)
f = open(p["NAME"]+"_results.txt","a")
f.write(f"{p['N_WAY']} {p['K_SHOT']} {mn} {std}\n")
f.close()
print(f"Avg Test: {mn} +/- {std}")
