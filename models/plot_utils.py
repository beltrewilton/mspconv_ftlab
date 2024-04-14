import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from vad.vad_lab import VAD


def plot_freq(loader:DataLoader, split: str):
    train_counter = Counter()
    for inputs, labels in loader:
        for l in labels:
            train_counter.update(l.tolist())

    vad = VAD(minmax=[-100, 100], mapping="OCC")
    items = sorted(train_counter.items())
    data = []
    for idx, freq in items:
        if idx == 0: continue
        data.append([vad.terms[idx - 1], freq]) #TODO deprecated !

    df_emo_freq = pd.DataFrame(columns=['Emotion', 'Frequency'], data=data)
    df_emo_freq = df_emo_freq.sort_values(by=['Frequency'], ascending=False)

    fig = px.histogram(df_emo_freq, x="Emotion", y="Frequency",
                       title=f'Histograma frecuencia de emociones (MSP) en {split}', text_auto=True)

    return df_emo_freq, fig


def conf_matrix(y_hat, true_labels, terms, current_epoch, ov_acc):
    cf_matrix = confusion_matrix(y_hat, true_labels)
    idx = sorted(np.unique(np.hstack([y_hat, true_labels])))
    df_cm = pd.DataFrame(cf_matrix, index=[f"{terms[i]}:{i}" for i in idx], columns=[f"{terms[i]}:{i}" for i in idx])
    vmin = np.min(cf_matrix)
    vmax = np.max(cf_matrix)
    off_diag_mask = np.eye(*cf_matrix.shape, dtype=bool)
    fig = plt.figure(figsize=(8, 6))
    # gs0 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[20, 2], hspace=0.05)
    # gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], hspace=0)
    # ax = fig.add_subplot(gs0[0])
    # cax1 = fig.add_subplot(gs00[0])
    # cax2 = fig.add_subplot(gs00[1])
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.title(f"epoch: {current_epoch-1} acc: {ov_acc}")
    cm = sn.heatmap(df_cm, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax, cbar=False, fmt='g')
    cm = sn.heatmap(df_cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]), cbar=False, fmt='g')
    plt.tight_layout()

    return cm
