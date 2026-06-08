import matplotlib.pyplot as plt
import numpy as np

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
KERNEL_PROFILING = False
HIDDEN_LAYERS = 4
NUM_HIDDEN = [ 128, 64, 64, 64 ]
KERNEL_SZ = [ 3, 3, 3, 3 ]
CONV_STRIDES = [ 2, 1, 1, 1 ]
HID_MEAN = [ 1.0, 0.0, 0.0, 0.0 ]
HID_SD = [3.0, 1.0, 1.0, 1.0 ]
LR = [2e-2, 2e-3, 2e-3, 2e-3, ]
N_WAY = 5
K_SHOT = 5

images, test_labels, alph_ids, char_ids = utils.load_omniglot("test")
test_img = utils.rescale(images)

serialiser = Numpy("latency_omniglot_conv4_checkpoints")
network = SequentialNetwork()

NUM_OUTPUT = 964

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
network.load((0,), serialiser)
compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                             reset_in_syn_between_batches=True,
                             batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

embeddings = utils.extract_embeddings(compiled_net, input, test_img, output)
embeddings = l2_normalise(embeddings)
res = utils.run_episode(embeddings, test_labels, N_WAY, K_SHOT)
print(f"Euclidean nearest centroid: {res*100}% correct.")
    
