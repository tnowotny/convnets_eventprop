import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

def spike_stats(layer, cb_data, key):
    ids = cb_data[key][1]
    shape = layer.population().shape
    n_neuron = np.prod(shape)
    n_trial = shape[0]
    spkn = []
    for b in range(len(ids)):
        spkn.append(np.histogram(ids[b],bins= n_neuron, range=(0.0,n_neuron))[0])
    spkn = np.asarray(spkn)
    meanspike = np.mean(np.mean(spkn))
    sigspike = np.mean(np.std(np.mean(spkn,axis=1)))
    Nall = np.mean(np.sum(spkn,axis=1))
    Nallsig = np.std(np.sum(spkn,axis=1))
    return meanspike, sigspike, Nall, Nallsig

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
        img = np.asarray(x["image"])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = 255 - img
        images.append(img)
        labels.append(x["label"])
        alph_ids.append(alph_id)
        char_ids.append(int(x["alphabet_char_id"]))

    print(f"[Omniglot] Loaded {len(images)} samples from {split}")
    print(f"           Classes={len(np.unique(labels))}")
    return np.asarray(images), labels, alph_ids, char_ids


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

def augment(images, aug):
    new_img = np.squeeze(images.copy())
    shape = new_img[0].shape
    zoom = aug.get("zoom")
    if zoom is not None:
        ht = new_img[0].shape[0]
        wd = new_img[0].shape[1]
        base = np.zeros((ht,wd))
        for i, img in enumerate(new_img):
            fac = np.random.uniform(zoom[0], zoom[1])
            new_ht = int(ht * fac)
            new_wd = int(wd * fac)
            x = int(abs(new_wd-wd)/2)
            y = int(abs(new_ht-ht)/2)
            img_l = cv2.resize(img, (new_wd,new_ht), interpolation=cv2.INTER_LINEAR)
            if fac > 1:
                new_img[i] = img_l[y:y+ht,x:x+wd]
            else:
                new_img[i] = base
                new_img[i][y:y+new_ht,x:x+new_wd]= img_l
    rot = aug.get("rotate")
    if rot is not None:
        image_center = tuple(np.array(new_img[0].shape[1::-1]) / 2)
        for i, img in enumerate(new_img):
            angle = np.random.uniform(rot[0], rot[1])
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            new_img[i] = cv2.warpAffine(img, rot_mat, shape[1::-1], flags=cv2.INTER_LINEAR)
    shift = aug.get("shift")
    if shift is not None:
        for i,img in enumerate(new_img):
            the_shift = np.random.uniform(shift[0],shift[1],2)
            shift_mat = np.asarray([[ 1, 0, the_shift[0]], [ 0, 1, the_shift[1]]])
            new_img[i] = cv2.warpAffine(img, shift_mat, shape[1::-1])
    return np.expand_dims(new_img, axis=-1)

def rescale(images):
    new_img = []
    for img in images:
        new_img.append(cv2.resize(img, (28,28), interpolation=cv2.INTER_LINEAR))
    return np.expand_dims(np.asarray(new_img),axis=-1)

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


def predictions_from_vAvg(vAvg):
    vAvg = np.asarray(vAvg)
    preds = np.argmax(vAvg[:,-1:,:],axis=-1)
    return np.squeeze(preds)


def extract_embeddings(compiled_net, input, output, test_img, layer, batch_size, var):
    """Load final checkpoint, run inference, return membrane voltage embeddings."""
    print(f"Extracting embeddings")
    with compiled_net:
        n_batches = int(np.ceil(len(test_img) / batch_size))
        all_embeddings = []
        pops = compiled_net.genn_model.neuron_populations
        the_pop = pops[layer]
        for batch_idx, start in enumerate(range(0, len(test_img), batch_size)):
            batch_img = test_img[start : start + batch_size]
            n = len(batch_img)
            compiled_net.evaluate({input: batch_img},{output: np.zeros(batch_img.shape[0])},callbacks=[])
            the_pop.vars[var].pull_from_device()
            all_embeddings.append(the_pop.vars[var].values)
        embeddings = np.concatenate(all_embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Done: shape {embeddings.shape}, "
              f"norm mean={norms.mean():.3f} std={norms.std():.3f} "
              f"min={norms.min():.3f} max={norms.max():.3f}")
    return embeddings


def l2_normalise(embeddings):
    """L2-normalise each row to unit length."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def sample_episode(embeddings, labels, n_way, k_shot):
    """Sample N_WAY classes, K_SHOT support and the rest query images each."""
    classes = np.random.choice(np.unique(labels), n_way, replace=False)
    sup_emb, sup_lb = [], []
    qry_emb, qry_lb = [], []
    for new_lbl, cls in enumerate(classes):
        idx = np.where(labels == cls)[0]
        sup_emb.extend(embeddings[idx[:k_shot]])
        sup_lb += [new_lbl] * k_shot
        qry_emb.extend(embeddings[idx[k_shot:]])
        qry_lb += [new_lbl] * len(idx[k_shot:])

    return np.asarray(sup_emb), np.array(sup_lb), np.asarray(qry_emb), np.array(qry_lb)


def run_episode(embeddings, labels, n_way, k_shot):
    """One episode: Euclidean nearest-centroid classifier."""
    sup_emb, sup_lb, qry_emb, qry_lb = sample_episode(embeddings, labels,n_way, k_shot)
    prototypes = np.array([sup_emb[sup_lb == c].mean(0) for c in range(n_way)])
    dists = ((qry_emb[:, None] - prototypes[None]) ** 2).sum(-1)
    return (np.argmin(dists, axis=1) == qry_lb).mean()


def run_episode_linear(embeddings, labels, n_way, k_shot):
    """One episode: logistic regression fitted on support embeddings."""
    sup_emb, sup_lb, qry_emb, qry_lb = sample_episode(embeddings, labels, n_way, k_shot)
    # StandardScaler is important: high-dim embeddings make L-BFGS slow otherwise
    scaler = StandardScaler()
    sup_emb = scaler.fit_transform(sup_emb)
    qry_emb = scaler.transform(qry_emb)

    clf = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
    clf.fit(sup_emb, sup_lb)
    return (clf.predict(qry_emb) == qry_lb).mean()


def run_episode_cosine(embeddings, labels, n_way, k_shot):
    """One episode: cosine nearest-centroid (per-episode L2 normalisation)."""
    sup_emb, sup_lb, qry_emb, qry_lb = sample_episode(embeddings, labels, n_way, k_shot)

    sup_emb = sup_emb / np.maximum(np.linalg.norm(sup_emb, axis=1, keepdims=True), 1e-8)
    qry_emb = qry_emb / np.maximum(np.linalg.norm(qry_emb, axis=1, keepdims=True), 1e-8)

    prototypes = np.array([sup_emb[sup_lb == c].mean(0) for c in range(n_way)])
    dists = ((qry_emb[:, None] - prototypes[None]) ** 2).sum(-1)
    return (np.argmin(dists, axis=1) == qry_lb).mean()
