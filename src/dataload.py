import os
import pandas as pd
import numpy as np
import librosa
import itertools
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from operator import itemgetter
import plotly.express as px
import plotly.graph_objects as go
import pickle
from plotly.subplots import make_subplots
from tqdm import tqdm


##### constant !
WEIGHT_AVG_LABEL = 'WAvg'
MSP_PKL_FILE = 'mspconvs.pkl'


def load_audio_data(df_annotations: pd.DataFrame, pc_num: int, part_num: int) -> (np.ndarray, int):
    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -pc_num: PodCast Number
            -part_num: Part 

        Output:
            1- A numpy array with the audio time series
            2- Integer sampling rate
    """
    
    df_part_data = df_annotations[(df_annotations['PC_Num'] == pc_num) & (df_annotations['Part_Num'] == part_num)].reset_index()

    audio_name = df_part_data['Audio_Name'][0]
    start_time = df_part_data['start_time'][0]
    end_time = df_part_data['end_time'][0]

    audio_path = "data/MSPCORPUS/Audio/" + audio_name

    data, sr = librosa.load(audio_path, offset = start_time, duration = end_time - start_time, sr = None)

    time = np.arange(0, len(data)) * (1.0 / sr)

    return data, time, sr


def audio_select_mean_vote(df_annotations: pd.DataFrame, pc_num: int, part_num: int) -> pd.DataFrame:
    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -pc_num: PodCast Number
            -part_num: Part 

        Output:
            Dataframe consisting of 3 columns: Time, Arousal, Dominance, Valence.
            The Arousal, Dominance and Valence columns represent the mean vote calculated for corresponding Time in the audio.
    """

    votation_means = pd.DataFrame(columns = ['Time','Vote','Emotion'])
    emotions = ['Valence','Arousal','Dominance']

    for emotion in emotions:
        
        time = pd.DataFrame(columns = ['Time','Annotation','Annotator'])    
        df_copy = df_annotations[(df_annotations['PC_Num'] == pc_num) & (df_annotations['Part_Num'] == part_num) & (df_annotations.Emotion == emotion)]
        
        for name, annotator, emot in zip(df_copy['Name'], df_copy['Annotator'], df_copy['Emotion']):
        
            temp_df = pd.read_csv(f'data/MSPCORPUS/Annotations/{emot}/{name}', skiprows=9, header=None, names=['Time', 'Annotation'])
            temp_df['Annotator'] = annotator
            time = pd.concat([time, temp_df], ignore_index = True)
            
        df_pivot = pd.DataFrame(time.pivot_table(columns = 'Annotator', index = 'Time', values = 'Annotation').to_records()).set_index('Time')
        df_pivot = df_pivot.fillna(method='ffill')
        df_pivot['Vote'] = df_pivot.mean(axis = 1)
        df_pivot['Emotion'] = emotion
        # df_pivot['Dummy'] = 100

        # votation_means = pd.concat([votation_means, df_pivot.reset_index()[['Time','Vote','Emotion']]])
        votation_means = pd.concat([votation_means, df_pivot.reset_index()])

    df_emotions_vote = pd.DataFrame(votation_means.pivot_table(columns = 'Emotion', index = 'Time', values = 'Vote').to_records()).set_index('Time')
    df_emotions_vote = df_emotions_vote.fillna(method='ffill')
    
    # return df_emotions_vote.reset_index()[['Time','Valence','Arousal','Dominance']]
    return df_emotions_vote.reset_index()


def get_annotated_data() -> (pd.DataFrame, pd.DataFrame):
    """
    :return: Una tupla con dos dataframes; 1ro. nivel de detalle ampliado con emotions,
                                           2do. indicadores del podcast number y el numero de parte del audio
    """
    df_annotations = pd.read_excel(f'../data/annotations_2.xlsx')
    df_reduced = df_annotations.drop(columns=['Name', 'Emotion', 'Annotator'])
    df_reduced = df_reduced.drop_duplicates()
    return df_annotations, df_reduced


def get_pivot(df_annotations: pd.DataFrame, pc_num: int, part_num: int, emotion: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Construye un dataset tipo 'pivot' con los raters/anotadores participantes en la anotación
    dado el numero de podcast, la parte del audio y una emocion.
    Cada secuencia de anotaciones por anotador tiene una longitud arbitraria, por lo que EMA logró
    implementar una solución para "estandarizar" la longitud y que esto no sea un problema.
    :param df_annotations: dataframe con nivel de detalle ampliado + emotions
    :param pc_num: indica el numero del podcast
    :param part_num: indica la parte del audio
    :param emotion: dim; Valence, Arousal, Dominance
    :return: un lindo dataset de longitud unificada
    """

    annotations = []

    time = pd.DataFrame(columns=['Time', 'Annotation', 'Annotator'])

    df_copy = df_annotations[
        (df_annotations.PC_Num == pc_num) & (df_annotations.Part_Num == part_num) & (df_annotations.Emotion == emotion)]

    # Leo cada audio, para cada anotador para le emoción especificada{}
    for file_name, annotator, emotion_path in zip(df_copy['Name'], df_copy['Annotator'], df_copy['Emotion']):
        temp_df = pd.read_csv(f'{os.path.abspath("..")}/data/MSPCORPUS/Annotations/{emotion_path}/{file_name}', skiprows=9,
                              header=None, names=['Time', 'Annotation'])
        annotations.append(temp_df)
        temp_df['Annotator'] = annotator
        time = pd.concat([time, temp_df], ignore_index=True)

    del temp_df
    del df_copy

    df_pivot = pd.DataFrame(
        time.pivot_table(columns='Annotator', index='Time', values='Annotation').to_records()).set_index('Time')
    df_pivot = df_pivot.fillna(method='ffill')

    return df_pivot, annotations


def __cohen_kappa_score(raters) -> np.ndarray:
    """
     Calculate cohen_kappa_score for every combination of raters
     Combinations are only calculated j -> k, but not k -> j, which are equal
     So not all places in the matrix are filled.
    :param raters:
    :return: data
    """
    data = np.zeros((len(raters), len(raters)))
    ##### robado de stackaverflow ####
    for j, k in list(itertools.combinations(range(len(raters)), r=2)):
        data[j, k] = cohen_kappa_score(raters[j], raters[k], weights='linear')
    ######### fin del robo #########
    return data


def score_scaler(annolist, data) -> (dict, np.ndarray):
    """suma del cohen_kappa_score de cada anotador como peso para sacar el "weighted average"
    :param annolist:
    :param data:
    :return: reps_score, reps_scaled
    """
    reps = []
    # print("*** Anotador c/u cohen_kappa_score + la suma horizontal ")
    for idx, a in enumerate(annolist):
        rep = data[:, idx] + data[idx, :]
        reps.append(np.sum(rep))

    reps = np.array(reps)
    scaler = MinMaxScaler()
    annolist = np.array(annolist).reshape(-1, 1)
    #################
    # Objetivo: suma del cohen_kappa_score de cada anotador como peso para sacar el "weighted average"
    # Razon por la que se usa MinMaxScaler [0, 1]
    #################
    reps_scaled = scaler.fit_transform(reps.reshape(-1, 1))
    reps_score = np.hstack((annolist, reps_scaled))
    reps_score = {u[0]: u[1] for u in reps_score}
    # print(
    #     "*** Anotador con la suma de sus scores transformada a [0,1] MinMaxScaler \n*** Objetivo: usar np.average con estos pesos.")

    return reps_score, reps_scaled


def calc_weighted_avg(df, reps_scaled) -> pd.DataFrame:
    """
    agraga un lindo dataframe al "pivot" con el weighted average,
    es decir, queremos una traza con mayor concordancia de los anotadores
    :param df:
    :param reps_scaled:
    :return:
    """
    df_w = df.copy()
    df_w['weighted_avg'] = np.average(df.iloc[:], weights=reps_scaled[:, 0], axis=1)
    df_w = df_w.reset_index()
    df_w = df_w.rename(columns={'weighted_avg': 'Annotation'})
    df_w['Annotator'] = WEIGHT_AVG_LABEL
    df_w = df_w[['Time', 'Annotator', 'Annotation']]
    return df_w


def build_mspconvs(df_annotations: pd.DataFrame, df_reduced: pd.DataFrame, pc_num: int = -1, part_num: int = -1) -> dict:
    """
    Proceso pesado para sacar TODOS los raters/anotadores y demás listas
    :return: un diccionario de mspconvs
    """
    import warnings
    warnings.filterwarnings("ignore")

    if pc_num != -1 and part_num != -1:
        df_reduced = df_reduced[(df_reduced.PC_Num == pc_num) & (df_reduced.Part_Num == part_num)]
    mspconvs = {}
    for idx, row in tqdm(df_reduced.iterrows(), desc=f"Procesando 🚧, tome un café ☕️ y espere ⏳...", total=len(df_reduced), ncols=120):
        emotions = ['Valence', 'Arousal', 'Dominance']
        for emotion in emotions:
            df_pivot, annotations = get_pivot(df_annotations, row['PC_Num'], row['Part_Num'], emotion)
            key = f"{row['PC_Num']}_{row['Part_Num']}_{emotion}"
            raters = df_pivot.T.to_numpy() * 100
            raters = raters.astype(np.int16)
            ck_score = __cohen_kappa_score(raters)
            annolist = df_pivot.columns.values
            reps_score, reps_scaled = score_scaler(annolist, ck_score)
            df_wavg = calc_weighted_avg(df_pivot, reps_scaled)
            annotations.append(df_wavg)

            mspconvs[key] = {
                'raters': raters,                   # ndarray: (6 anotadores, X annotations points de longitud estandarizada)
                'annolist': annolist,               # ndarray: (6 label anotadores)
                'time': df_pivot.index.to_numpy(),  # ndarray: (representa el tiempo de cada annotations points estandarizada)
                'ck_score': ck_score,               # ndarray: (6, 6) cohen_kappa_score heatmap/relación o "distacia" entre cada anotador
                'reps_score': reps_score,           # dict: (6) anotador y el peso que representa
                'reps_scaled': reps_scaled,         # ndarray: (6) peso de cada anotador
                'annotations': annotations,         # list(Dataframes) 6 dataframes de anotadores + 1 con el  weighted average
                                                    # (NO longitud estandarizada): tal como está en los archivos csv's
            }

    return mspconvs


def sample_heatmap(mspconv: dict, key: str):
    """
    plot un heatmap de la relaciónvo "distancia" de cada anotador
    :param mspconv:
    :param key:
    :return: None
    """
    raters = mspconv['raters']
    ck_score = mspconv['ck_score']
    annolist = mspconv['annolist']

    ax = plt.axes()
    ax.set_title(key)
    sns.heatmap(
        ck_score,
        mask=np.tri(len(raters)),
        annot=True, linewidths=5,
        vmin=0, vmax=1,
        xticklabels=[f" {a}" for a in annolist],
        yticklabels=[f" {a}" for a in annolist],
    )
    plt.show()


def sample_scatter(mspconv: dict, key: str):
    def color(opacity: int = 1):
        return (
            f'rgba({np.random.randint(127, high=256)}, {np.random.randint(127, high=256)}, {np.random.randint(127, high=256)}, {opacity})')

    def scolor(opacity: int = 1):
        static_colors = [f'rgba(250, 165, 105, {opacity})', f'rgba(179, 168, 20 ,{opacity})',
                         f'rgba(85, 161, 14 ,{opacity})', f'rgba(10, 199, 67 ,{opacity})',
                         f'rgba(9, 128, 184 ,{opacity})', f'rgba(108, 9, 184 ,{opacity})', ]
        return static_colors

    fig = go.Figure()
    annotations = mspconv['annotations']
    reps_score = mspconv['reps_score']
    for i, annotation in enumerate(annotations):
        label = str(annotation.Annotator.iloc[0])
        opacity = 0.8
        if label == WEIGHT_AVG_LABEL:
            fig.add_scatter(x=annotation.Time, y=annotation.Annotation, line=dict(color='rgba(217, 9, 92, 1)'))
        else:
            label = f"{label} \t{np.round(reps_score[label], 4)}"
            opacity = 0.3
            fig.add_scatter(x=annotation.Time, y=annotation.Annotation, line=dict(color=scolor(opacity)[i]))
        fig.data[i].name = label

    fig['data'][0]['showlegend']=True
    fig.update_yaxes(range=[-100, 100], autorange=False)
    # fig.update_layout(hovermode="x")
    # fig.update_traces(hovertemplate=None)
    # wa = "*Weighted Avg*" if with_weghts is not None else ""
    wa = ""
    fig.update_layout(
        xaxis_title="Time (seconds)", yaxis_title="Ratings",
        margin=dict(l=0, r=0, t=50, b=0),
        autosize=False, width=1600, height=500,
        title_text=f'Annotations on MSP-Conversation_{key}', title_x=0.5,
    )

    fig.show()


def save_mspconvs(mspconvs: dict):
    with open(f'../data/{MSP_PKL_FILE}', "wb") as pkl:
        pickle.dump(mspconvs, pkl)

def play():
    # key ="197_1_Valence"
    # mspconv = mspconvs[key]
    # sample_scatter(mspconv, key)
    #
    # key = "197_1_Arousal"
    # mspconv = mspconvs[key]
    # sample_scatter(mspconv, key)
    #
    # key = "197_1_Dominance"
    # mspconv = mspconvs[key]
    # sample_scatter(mspconv, key)
    #
    # key = "197_2_Valence"
    # mspconv = mspconvs[key]
    # sample_scatter(mspconv, key)
    #
    # key = "197_2_Arousal"
    # mspconv = mspconvs[key]
    # sample_scatter(mspconv, key)
    #
    # key = "197_2_Dominance"
    # mspconv = mspconvs[key]
    # sample_scatter(mspconv, key)
    #
    # key = "197_3_Valence"
    # mspconv = mspconvs[key]
    # sample_scatter(mspconv, key)
    #
    # key = "197_3_Arousal"
    # mspconv = mspconvs[key]
    # sample_scatter(mspconv, key)

    mspconv_from_disk = {}
    with open(f"../data/{MSP_PKL_FILE}", "rb") as pkl:
        mspconv_from_disk = pickle.load(pkl)

    key = "197_3_Dominance"
    mspconv_from_disk = mspconv_from_disk[key]
    sample_scatter(mspconv_from_disk, key)
    sample_heatmap(mspconv_from_disk, key)


if __name__ == "__main__":
    df_annotations, df_reduced = get_annotated_data()
    ####### mspconvs = build_mspconvs(df_annotations, df_reduced, pc_num=1701, part_num=1)
    ####### df_reduced = df_reduced.iloc[101:119]
    # df_reduced = df_reduced.iloc[101:111]
    mspconvs = build_mspconvs(df_annotations, df_reduced)

    #TODO: rethink names of variables and method
    #TODO: EDA with this data/rates/etc
    #TODO: BIG NEXT is according with conclusion from EDA try to reduce the size by quality choose
    #TODO: BIG NEXT is also split audio by 30 seconds

    save_mspconvs(mspconvs)

    print('Fin de proceso ⛱️.')
