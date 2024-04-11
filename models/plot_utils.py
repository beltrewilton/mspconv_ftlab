import pandas as pd
from torch.utils.data import DataLoader
from collections import Counter
import plotly.express as px
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
