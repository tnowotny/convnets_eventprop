import matplotlib.pyplot as plt
import numpy as np
import mnist

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

NUM_INPUT = 784
BATCH_SIZE = 32
NUM_EPOCHS = 2000
EXAMPLE_TIME = 25.0
DT = 1.0
TRAIN = True
KERNEL_PROFILING = False
PLOT_EPOCHS = 50
HIDDEN_LAYERS = 4
NUM_HIDDEN = [ 128, 64, 64, 64 ]
KERNEL_SZ = [ 3, 3, 3, 3 ]
CONV_STRIDES = [ 2, 1, 1, 1 ]
HID_MEAN = [ 1.0, 0.0, 0.0, 0.0 ]
HID_SD = [3.0, 1.0, 1.0, 1.0 ]
LR = [2e-2, 2e-3, 2e-3, 2e-3, ]
ALPHABETS = None
AUG = {"rotate": (-10.0,10.0),
       "shift": (-15, 15)
       }
SHOW_AUGMENTAION_EXAMPLE = False

images, labels, alph_ids, char_ids = utils.load_omniglot("train") if TRAIN else load_omniglot("test")
if TRAIN:
    train_img, train_labels, val_img, val_labels = utils.stratified_split(images, labels, alph_ids, ALPHABETS)
    val_img = utils.rescale(val_img)
else:
    test_img = utils.rescale(images)
    
serialiser = Numpy("latency_omniglot_conv4_checkpoints")
network = SequentialNetwork()
NUM_OUTPUT = len(np.unique(train_labels))
print(NUM_OUTPUT)
with network:
    # Populations
    input = InputLayer(LatencyInput("linear", EXAMPLE_TIME - (2.0 * DT), 2.0 * DT, 1, False),
                       (28, 28, 1), name="input",record_spikes= True)   
    hidden = []
    for nh in range(HIDDEN_LAYERS):
        initial_hidden_weight = Normal(mean= HID_MEAN[nh], sd= HID_SD[nh])
        flatten =  nh == HIDDEN_LAYERS-1 
        hidden.append(Layer(Conv2D(initial_hidden_weight, NUM_HIDDEN[nh],
                                   KERNEL_SZ[nh], flatten, conv_strides=CONV_STRIDES[nh]),
                                   LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                                   synapse=Exponential(5.0), name=f"hidden{nh}", record_spikes= True))
    output = Layer(Dense(Normal(mean=0.0, sd=0.1)),
                   LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),    
                   NUM_OUTPUT, Exponential(5.0), name="output")

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 max_spikes = 15,
                                 losses="sparse_categorical_crossentropy",
                                 batch_size=BATCH_SIZE, kernel_profiling=KERNEL_PROFILING)
    optimisers = {hidden[nh]: {"weight": Adam(LR[nh])} for nh in range(HIDDEN_LAYERS)}
    optimisers[output] = {"weight": Adam(5e-4)}
    compiled_net = compiler.compile(network, optimisers=optimisers, regularisers={"all_hidden_populations": SpikeCount(strength=(1e-7,1e-5),target=1)})

    with compiled_net:
        start_time = perf_counter()
        #callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        callbacks = [ SpikeRecorder(input, "sin", example_filter= [ 1, 500, 1000, 1500 ]),VarRecorder(output,"vAvg","outvavg")
        ]
        val_callbacks = [VarRecorder(output,"vAvg","outvavg")]
        for nh in range(HIDDEN_LAYERS):
            callbacks.append(SpikeRecorder(hidden[nh], f"shid{nh}",example_filter=[ 1, 500, 1000, 1500 ]))
        for e in range(NUM_EPOCHS):
            the_img = utils.augment(train_img, AUG)
            #the_img = train_img
            the_img = utils.rescale(the_img)
            if SHOW_AUGMENTAION_EXAMPLE:
                for i in range(10):
                    fig,ax = plt.subplots(1,2)
                    ax[0].imshow(train_img[i])
                    ax[1].imshow(the_img[i])
                plt.show()
            metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train_validate({input: the_img},
                                                                                      {output: train_labels},
                                                                                      validation_x= {input: val_img},
                                                                                      validation_y = {output: val_labels},
                                                                                      num_epochs=1, start_epoch=e, shuffle=False,
                                                                                      callbacks=callbacks,validation_callbacks= val_callbacks)
            print(f"{metrics[output].result} {val_metrics[output].result}")
            compiled_net.save((0,),serialiser)
            if e % PLOT_EPOCHS == 0:
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
                w = utils.get_conn_var(compiled_net, hidden[0], "weight")
                fig0= utils.show_filters(w, (KERNEL_SZ[0], KERNEL_SZ[0]))
                grad = []
                utils.spike_raster(cb_data, f"sin")
                for nh in range(HIDDEN_LAYERS):
                    grad.append(utils.get_conn_var(compiled_net, hidden[nh], "weightGradient"))
                    utils.plot_var(grad[-1]) 
                    utils.spike_raster(cb_data, f"shid{nh}")
                plt.show()
            with open("results.txt","a") as f:
                for nh in range(HIDDEN_LAYERS):
                    smean,ssig,sallmean,sallsig = utils.spike_stats(hidden[nh],cb_data,f"shid{nh}")
                    f.write(f"{smean} {ssig} {sallmean} {sallsig} ")
                f.write(f"{metrics[output].result} {val_metrics[output].result}\n")
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")

        if KERNEL_PROFILING:
            print(f"Neuron update time = {compiled_net.genn_model.neuron_update_time}")
            print(f"Presynaptic update time = {compiled_net.genn_model.presynaptic_update_time}")
            print(f"Gradient batch reduce time = {compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')}")
            print(f"Gradient learn time = {compiled_net.genn_model.get_custom_update_time('GradientLearn')}")
            print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")
            print(f"Softmax1 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax1')}")
            print(f"Softmax2 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax2')}")
            print(f"Softmax3 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax3')}")
else:
    # Load network state from latest checkpoint
    network.load((0,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                 reset_in_syn_between_batches=True,
                                 batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network)
    w = utils.get_weights(compiled_net, hidden[0])
    fig0= utils.show_filters(w, (5, 5))
    plt.show()
    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        metrics, _  = compiled_net.evaluate({input: test_img},
                                            {output: labels})
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
