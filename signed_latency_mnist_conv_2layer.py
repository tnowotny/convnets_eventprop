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
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from time import perf_counter

import utils

NUM_INPUT = 784
NUM_HIDDEN = 128
NUM_OUTPUT = 10
BATCH_SIZE = 32
NUM_EPOCHS = 20
EXAMPLE_TIME = 25.0
DT = 1.0
TRAIN = True
KERNEL_PROFILING = False
NUM_EX = 60000
PLOT_EPOCHS = 2000

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
images = mnist.train_images() if TRAIN else mnist.test_images()
images = np.asarray(images).astype(float)
images = np.minimum((images-127.5)*2,255)
if TRAIN:
    train_img = images[:50000,:,:]
    val_img = images[50000:,:,:]
    train_img = train_img[:NUM_EX,:,:]
    train_labels = labels[:50000]
    val_labels = labels[50000:]
    train_labels = train_labels[:NUM_EX]
    train_img = np.expand_dims(train_img, axis=3)
    val_img = np.expand_dims(val_img, axis=3)
else:
    #labels = mnist.train_labels()
    #images = mnist.train_images()
    #images = np.asarray(images).astype(float)
    #images = np.minimum((images-127.5)*2,255)
    test_img = np.expand_dims(images, axis=3)
    
serialiser = Numpy("latency_mnist_2l_checkpoints")
network = SequentialNetwork()
with network:
    # Populations
    input = InputLayer(LatencyInput("linear", EXAMPLE_TIME - (2.0 * DT), 2.0 * DT, 1, True),
                       (28, 28, 1), name="input",record_spikes= True)   
    initial_hidden1_weight = Normal(mean= 1.0, sd=1.0)
    hidden1 = Layer(Conv2D(initial_hidden1_weight, NUM_HIDDEN, 5, False, conv_strides=3),
                    LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                    synapse=Exponential(5.0), name="hidden1", record_spikes= True)
    initial_hidden2_weight = Normal(sd=0.1)
    hidden2 = Layer(Conv2D(initial_hidden2_weight, 64, 5, True),
                    LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                    synapse=Exponential(5.0), name="hidden2", record_spikes= True)
    output = Layer(Dense(Normal(mean=0.0, sd=0.1)),
                   LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),    
                   NUM_OUTPUT, Exponential(5.0), name="output")

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 reg_lambda= (0,0), reg_nu_upper= 1.0, 
                                 batch_size=BATCH_SIZE, kernel_profiling=KERNEL_PROFILING)
    compiled_net = compiler.compile(network, optimisers={hidden1: {"weight": Adam(1e-1)}, hidden2: {"weight": Adam(5e-4)}, output: {"weight": Adam(5e-4)}})

    with compiled_net:
        start_time = perf_counter()
        #callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        callbacks = [ SpikeRecorder(input, "sin", example_filter= [ 1, 500, 1000, 1500 ]),
                      SpikeRecorder(hidden1, "shid1",example_filter=[ 1, 500, 1000, 1500 ]),
                      SpikeRecorder(hidden2, "shid2",example_filter=[ 1, 500, 1000, 1500 ]),
                      VarRecorder(hidden2,"Lambdav", "lbdv1", example_filter=[1, 500, 1000, 1500 ]),
                      #ConnVarRecorder(hidden1.connection(), "Gradient", "hid1_g", example_filter=[1])
        ]
        for e in range(NUM_EPOCHS):
            metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train_validate({input: train_img},
                                                                                      {output: train_labels},
                                                                                      validation_x= {input: val_img},
                                                                                      validation_y = {output: val_labels},
                                                                                      num_epochs=1, start_epoch=e, shuffle=True,
                                                                                      callbacks=callbacks,validation_callbacks= [])
            with open("results.txt","a") as f:
                f.write(f"{metrics[output].result} {val_metrics[output].result}\n")
            print(f"{metrics[output].result} {val_metrics[output].result}")
            compiled_net.save((0,),serialiser)
            if e % PLOT_EPOCHS == 0:
                w = utils.get_conn_var(compiled_net, hidden1, "weight")
                grad1 = utils.get_conn_var(compiled_net, hidden1, "weightGradient")
                grad2 = utils.get_conn_var(compiled_net, hidden2, "weightGradient")
                fig0= utils.show_filters(w, (5, 5))
                fig1= utils.spike_raster(cb_data, "sin")
                fig2= utils.spike_raster(cb_data, "shid1")
                fig3= utils.spike_raster(cb_data, "shid2")
                fig4= utils.plot_var(grad1)    
                fig5= utils.plot_var(grad2)
                fig6= utils.plot_var(cb_data["lbdv1"])
                plt.show()
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
    w = utils.get_weights(compiled_net, hidden1)
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
