import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

def show_filters(w, fshape):
    fh, fw = fshape
    w = w.copy().reshape(fh,fw,-1)
    Nplots = w.shape[2]
    glb_min = np.min(w.flatten())
    glb_max = np.max(w.flatten())
    print(f"Global min filter value: {glb_min}, GLobal max filter value: {glb_max}")
    if Nplots > 1:
        ploth = int(np.sqrt(Nplots))
        plotw = (Nplots-1)//ploth + 1
        fig,ax = plt.subplots(ploth,plotw,sharex=True,sharey=True)
        twoD = ploth > 1
        oneD = True
    else:
        ploth = 1
        plotw = 1
        fig = plt.figure()
        ax = plt.gca()
        twoD = False
        oneD = False

    for i in range(ploth):
        for j in range(plotw):
            id = i*plotw+j
            if twoD:
                the_ax = ax[i,j]
            elif oneD:
                the_ax = ax[j]
            else:
                the_ax = ax
            if id < Nplots:
                img = the_ax.imshow(w[:,:,id], vmin = glb_min, vmax = glb_max)
                the_ax.set_xticks([])
                the_ax.set_yticks([])
                #print(f"Min: {np.min(w[:,:,id])}, Max: {np.max(w[:,:,id])}")
                if id == Nplots-1:
                    fig.colorbar(img)
            else:
                the_ax.set_visible(False)
    return fig

def get_conn_var(compiled_net, layer, var):
    conn = layer.connection
    pop = compiled_net.connection_populations[conn()]
    pop.vars[var].pull_from_device()
    w = pop.vars[var].values.copy()
    return w

def spike_raster(cb_data, key, trials= None):
    t = cb_data[key][0]
    ids = cb_data[key][1]
    if trials is None:
        trials= list(range(len(t)))
    Nplots = len(trials)
    if Nplots > 1:
        ploth = int(np.sqrt(Nplots))
        plotw = (Nplots-1)//ploth + 1
        fig,ax = plt.subplots(ploth,plotw,sharex=True,sharey=True)
        twoD = ploth > 1
        oneD = True
    else:
        ploth = 1
        plotw = 1
        fig = plt.figure()
        ax = plt.gca()
        twoD = False
        oneD = False

    for i in range(ploth):
        for j in range(plotw):
            id = i*plotw+j
            if twoD:
                the_ax = ax[i,j]
            elif oneD:
                the_ax = ax[j]
            else:
                the_ax = ax
            if id < Nplots:
                the_ax.scatter(t[id],ids[id],marker="|",s=4)
            else:
                the_ax.set_visible(False)
                
    fig.suptitle(f"layer {key}")
    return fig

def plot_var(d, trials= None):
    if trials is None:
        trials= list(range(len(d)))
    Nplots = len(trials)
    if Nplots > 1:
        ploth = int(np.sqrt(Nplots))
        plotw = (Nplots-1)//ploth + 1
        fig,ax = plt.subplots(ploth,plotw,sharex=True,sharey=True)
        twoD = ploth > 1
        oneD = True
    else:
        ploth = 1
        plotw = 1
        fig = plt.figure()
        ax = plt.gca()
        twoD = False
        oneD = False

    for i in range(ploth):
        for j in range(plotw):
            id = i*plotw+j
            if twoD:
                the_ax = ax[i,j]
            elif oneD:
                the_ax = ax[j]
            else:
                the_ax = ax
            if id < Nplots:
                the_ax.plot(d[id])
            else:
                the_ax.set_visible(False)
                
    #fig.suptitle(f"layer {key}")
    return fig


def load_omniglot(split="train"):
    images = []
    labels = []
    alph_ids = []
    char_ids = []

    # Load full dataset (split == "train" or == "test")
    ds = tfds.load("omniglot", split=split, shuffle_files=False)

    # Convert TFDS tensors to numpy
    for x in ds.as_numpy_iterator():
        alph_id = int(x["alphabet"])
        img = x["image"]
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize(img, [28, 28])
        mn, mx = tf.reduce_min(img), tf.reduce_max(img)
        img = tf.where(mx > mn, (img - mn) / (mx - mn), img)
        img = tf.clip_by_value(1.0-img, 0.0, 1.0)*255
        images.append(img)
        labels.append(x["label"])
        alph_ids.append(alph_id)
        char_ids.append(int(x["alphabet_char_id"]))

    print(f"[Omniglot] Loaded {len(images)} samples from {split}")
    print(f"           Classes={len(np.unique(labels))}")
    print(images[0].shape)
    return images, labels, alph_ids, char_ids


def stratified_split(images, labels, alph_ids, alphabets = None, val_split=0.1):
    train_idx, val_idx = [], []    
    img = np.asarray(images)
    lbl = np.asarray(labels)
    aids = np.asarray(alph_ids)
    rng = np.random.default_rng()
    if alphabets is None:
        alphabets = np.unique(alph_ids)
    t_img = []
    t_lbl = []
    for a in alphabets:
        t_img.extend(img[aids == a])
        t_lbl.extend(lbl[aids == a])
    img = np.asarray(t_img)
    uniq = np.unique(t_lbl)
    remap = {u: i for i, u in enumerate(uniq)}
    lbl = np.vectorize(remap.get)(t_lbl)
    for c in np.unique(lbl):
        idx = np.where(lbl == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_split))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return img[train_idx], lbl[train_idx], img[val_idx], lbl[val_idx]


def show_examples(images, labels, the_label):
    img = np.asarray(images)[np.asarray(labels) == the_label]
    Nplots = img.shape[0]
    glb_min = np.min(img.flatten())
    glb_max = np.max(img.flatten())
    print(f"Global min filter value: {glb_min}, GLobal max filter value: {glb_max}")
    if Nplots > 1:
        ploth = int(np.sqrt(Nplots))
        plotw = (Nplots-1)//ploth + 1
        fig,ax = plt.subplots(ploth,plotw,sharex=True,sharey=True)
        twoD = ploth > 1
        oneD = True
    else:
        ploth = 1
        plotw = 1
        fig = plt.figure()
        ax = plt.gca()
        twoD = False
        oneD = False

    for i in range(ploth):
        for j in range(plotw):
            id = i*plotw+j
            if twoD:
                the_ax = ax[i,j]
            elif oneD:
                the_ax = ax[j]
            else:
                the_ax = ax
            if id < Nplots:
                the_ax.imshow(img[id])
            else:
                the_ax.set_visible(False)
