import matplotlib.pyplot as plt
import numpy as np
import sys, json

from ml_genn import Network, Population, Connection
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
from ml_genn.readouts import AvgVarExpWeight

from time import perf_counter

import utils

p = {
    "INPUT_SIZE": (84,84),
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 100,
    "EXAMPLE_TIME": 75.0,
    "DT": 1.0,
    "KERNEL_PROFILING": False,
    "PLOT_EPOCHS": 2000,
    "RES_BLOCKS": 4,
    "N_FILTER": [ 64, 128, 256, 512 ],
    "HID_MEAN": 0.0,
    "HID_SD": 1.0,
    "SKIP_MEAN": 4.0,
    "SKIP_SD": 1.0,
    "LR": 5e-3,
    "AUG": {"rotate": (-15.0,15.0), "shift": (-25,25), "zoom": (0.8,1.2)},
    "SHOW_AUGMENTAION_EXAMPLE": False,
    "REG_STRENGTH":(1e-10,1e-10),
    "REG_TARGET": 0.1,
    "NAME": "resnet12_test_3",
    "SHUFFLE": True,
    "RECORD_CONFUSION": False,
    "HEADLESS": True,
    "TRAINING_ROTATION": False,
    "PLOT_RASTER": ["input", "pop1"],
}

if len(sys.argv) > 1:
    fname= f"{sys.argv[1]}.json"
    with open(fname,"r") as f:
        p0= json.load(f)

    for (name,value) in p0.items():
        p[name]= value
    
#print(p)
with open(f"{p['NAME']}_run.json","w") as f:
    json.dump(p,f,indent=4)


"""
makes resnet block, connecting to all prev_pop (list of pops) incoming pops
and returns the update list of pops, conns, and the "output pops"
"""
def resblock(pops, conns, n_filter, prev_pop, pool=1, flatten=False):
    convs= []
    if pool > 1:
        shape = tuple((x+1)//pool for x in prev_pop[0].shape[:2])+(n_filter,)
    else:
        shape = prev_pop[0].shape[:2]+(n_filter,)    
    the_shape= int(np.prod(shape)) if flatten else shape
    r = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,), shape=the_shape, name=f"pop{len(pops)}", record_spikes= True)
    pops.append(r)
    c = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,), shape=shape, name=f"pop{len(pops)}", record_spikes= True)
    pops.append(c)
    convs.append(c)
    for i in range(2):
        shape = convs[-1].shape[:2]+(n_filter,)
        if i == 1 and flatten:
            shape = int(np.prod(shape))
        c = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,),shape=shape, name=f"pop{len(pops)}", record_spikes= True)
        pops.append(c)
        convs.append(c)
    for pop in prev_pop:
        print(pop.shape)
        print(n_filter)
        initial_hidden_weight = Normal(mean= p["SKIP_MEAN"], sd= p["SKIP_SD"])
        cr = Connection(pop,r,Conv2D(initial_hidden_weight,n_filter,(1,1), flatten=flatten, conv_strides=pool),Exponential(5.0),name=f"c_{pop.name}_{r.name}")
        conns.append(cr)
        initial_hidden_weight = Normal(mean= p["HID_MEAN"], sd= p["HID_SD"])
        cc = Connection(pop,convs[0],Conv2D(initial_hidden_weight,n_filter,(3,3),flatten=False,conv_strides=pool,conv_padding="same"),Exponential(5.0),name=f"c_{pop.name}_{convs[0].name}")
        conns.append(cc)
    flat = False
    for i in range(2):
        if i == 1 and flatten:
            flat = True
        cc = Connection(convs[i],convs[i+1],Conv2D(initial_hidden_weight,n_filter,(3,3),flatten=flat, conv_strides=1,conv_padding="same"),Exponential(5.0),name=f"c_{convs[i].name}_{convs[i+1].name}")
        conns.append(cc)
    return pops, conns, [ r, convs[-1] ]

train_img, train_labels = utils.load_mini_imagenet("train") 
val_img, val_labels = utils.load_mini_imagenet("val")

images = np.concatenate((train_img,val_img), axis=0)
labels = np.concatenate((train_labels,val_labels), axis=0)

train_img, train_labels, val_img, val_labels = utils.simple_strat_split(images, labels)
val_img = utils.rescale_3(val_img,p["INPUT_SIZE"])

#train_img = train_img[:32]
#train_labels = train_labels[:32]
#val_img = val_img[:32]
#val_labels = val_labels[:32]

serialiser = Numpy(f"{p['NAME']}_checkpoints")
network = Network()
NUM_OUTPUT = len(np.unique(train_labels))

pops = []
conns = []
with network:
    # Populations
    input = Population(LatencyInput("linear", p["EXAMPLE_TIME"] - (2.0 * p["DT"]),
                                    2.0 * p["DT"], 1, False),
                       tuple(p["INPUT_SIZE"])+(3,), name="input",record_spikes= True)
    pops.append(input)
    prev = [ input ]
    pool = 1
    for nh in range(p["RES_BLOCKS"]):
        flatten =  nh == p["RES_BLOCKS"]-1 
        pops, conns, prev = resblock(pops, conns, p["N_FILTER"][nh], prev, pool, flatten)
        pool = 2
    ro = AvgVarExpWeight(window_start = 20, window_end = 75)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout=ro),    
                   NUM_OUTPUT, name="output")
    pops.append(output)
    for pop in prev:
        initial_hidden_weight = Normal(mean= p["HID_MEAN"], sd= p["HID_SD"])
        c = Connection(pop,output,Dense(initial_hidden_weight),Exponential(5.0),name=f"c_{pop.name}_{output.name}")
        conns.append(c)

    for pop in pops:
        print(f"Population {pop.name} of shape {pop.shape}")

    for co in conns:
        print(f"Connection {co}")

max_example_timesteps = int(np.ceil(p["EXAMPLE_TIME"] / p["DT"]))
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             max_spikes = 15,
                             losses="sparse_categorical_crossentropy",
                             batch_size=p["BATCH_SIZE"], kernel_profiling=p["KERNEL_PROFILING"])
optimisers = {}
for co in conns:
    if "output" in co.name:
        optimisers[co] = {"weight": Adam(p["LR"]/10)}
    else:
        optimisers[co] = {"weight": Adam(p["LR"])}
        
compiled_net = compiler.compile(network, name=p["NAME"], optimisers=optimisers, regularisers={"all_hidden_populations": SpikeCount(strength=p["REG_STRENGTH"],target=p["REG_TARGET"])})
with compiled_net:
    start_time = perf_counter()
    #callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
    callbacks = []
    val_callbacks = []
    if p["RECORD_CONFUSION"]:
        callbacks.append(VarRecorder(output,"vAvg","outvavg"))
        val_callbacks.append(VarRecorder(output,"vAvg","outvavg"))
    for pop in pops:
        if pop.name != "output":
            callbacks.append(SpikeRecorder(pop, pop.name, example_filter=[ 0, 1 ]))
    val_best = 0.0
    for e in range(p["NUM_EPOCHS"]):
        the_img = utils.augment_mini_imagenet(train_img, p["AUG"])
        the_img = utils.rescale_3(the_img,p["INPUT_SIZE"])
        if not p["HEADLESS"] and p["SHOW_AUGMENTAION_EXAMPLE"]:
            for i in range(10):
                fig,ax = plt.subplots(1,2)
                ax[0].imshow(train_img[i])
                ax[1].imshow(the_img[i])
            plt.show()
        metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train_validate({input: the_img},
                                                                                  {output: train_labels},
                                                                                  validation_x= {input: val_img},
                                                                                  validation_y = {output: val_labels},
                                                                                  num_epochs=1, start_epoch=e,
                                                                                  shuffle=p["SHUFFLE"],
                                                                                  callbacks=callbacks,
                                                                                  validation_callbacks= val_callbacks)
        print(f"{metrics[output].result} {val_metrics[output].result}")
        if val_metrics[output].result > val_best:
            # save network with best validation accuracy
            val_best = val_metrics[output].result
            compiled_net.save((0,),serialiser)
        if not p["HEADLESS"] and e % p["PLOT_EPOCHS"] == 0:
            if p["RECORD_CONFUSION"]:
                for (cbdata, lbls) in zip([cb_data, val_cb_data],[train_labels, val_labels]):
                    n = len(np.unique(lbls))
                    conf= np.zeros((n,n))
                    preds= utils.predictions_from_vAvg(cbdata["outvavg"])
                    for i,j in zip(lbls,preds):
                        conf[i,j] += 1
                    print(np.sum(conf.flatten()))
                    plt.figure()
                    plt.imshow(conf)
                    plt.gca().yaxis.set_inverted(False)
            #w = utils.get_conn_var(compiled_net, hidden[0], "weight")
            #fig0= utils.show_filters(w, (3,3))
            grad = []
            for pop in pops:
                if pop.name != "output" and pop.name in p["PLOT_RASTER"]:
                    utils.spike_raster(cb_data, pop.name)
            for co in conns:
                var = utils.get_conn_var(compiled_net, co, "weightGradient")
                grad.append(var)
                print(f"{co.name}: {var.shape}")
                #utils.plot_var(grad[-1]) 
            plt.show()
        with open(f"{p['NAME']}_results.txt","a") as f:
            f.write(f"{e} ")
            for pop in pops:
                if pop.name != "output":
                    smean,ssig,sallmean,sallsig = utils.spike_stats(pop,cb_data,pop.name)
                f.write(f"{smean} {ssig} {sallmean} {sallsig} ")
            f.write(f"{metrics[output].result} {val_metrics[output].result}\n")
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
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
