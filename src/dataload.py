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
from pydub import AudioSegment
import base64
from vad.vad_lab import VAD


##### constant !
WEIGHT_AVG_LABEL = 'WAvg'
MSP_PKL_FILE = 'mspconvs.pkl'

def load_audio_data(df_annotations: pd.DataFrame, part_num: int, pc_num: int = None, audio_name: str = None) -> (np.ndarray, int):
    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -part_num: Audio Part
            -pc_num (optional): PodCast Number
            -audio_name (optional): Audio name, including the .wav extension 

        Output:
            1- A numpy array with the audio time series
            2- Integer sampling rate
    """
    
    if audio_name is not None: df_part_data = df_annotations[(df_annotations['Audio_Name'] == audio_name) & (df_annotations['Part_Num'] == part_num)].reset_index()
    else: df_part_data = df_annotations[(df_annotations['PC_Num'] == pc_num) & (df_annotations['Part_Num'] == part_num)].reset_index()

    audio_name = df_part_data['Audio_Name'][0]
    start_time = df_part_data['start_time'][0]
    end_time = df_part_data['end_time'][0]

    audio_path = "data/MSPCORPUS/Audio/" + audio_name

    data, sr = librosa.load(audio_path, offset = start_time, duration = end_time - start_time, sr = None)

    time = np.arange(0, len(data)) * (1.0 / sr)

    return data, time, sr


def audio_select_mean_vote(df_annotations: pd.DataFrame, part_num: int, pc_num: int = None, audio_name: str = None) -> pd.DataFrame:
    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -pc_num: PodCast Number
            -part_num: Part 
            -audio_name (optional): Audio name, including the .wav extension 

        Output:
            Dataframe consisting of 3 columns: Time, Arousal, Dominance, Valence.
            The Arousal, Dominance and Valence columns represent the mean vote calculated for corresponding Time in the audio.
    """

    votation_means = pd.DataFrame(columns = ['Time','Vote','Emotion'])
    emotions = ['Valence','Arousal','Dominance']

    for emotion in emotions:
        
        time = pd.DataFrame(columns = ['Time','Annotation','Annotator'])

        if audio_name is not None: df_copy = df_annotations[(df_annotations['Audio_Name'] == audio_name) & (df_annotations['Part_Num'] == part_num) & (df_annotations.Emotion == emotion)]
        else: df_copy = df_annotations[(df_annotations['PC_Num'] == pc_num) & (df_annotations['Part_Num'] == part_num) & (df_annotations.Emotion == emotion)]
        
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
    es decir, queremos una traza con la mejor concordancia entre los anotadores
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
    print('-----------------------------')
    print(f"MSP Conversation parts: {len(df_reduced)}")
    print('-----------------------------\n')
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
                'key': key,                         # str: la referencia en si misma
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

        static_colors = [f'rgba(0, 0, 255, {opacity})', f'rgba(0, 255, 0, {opacity})',
                         f'rgba(255, 165, 0,{opacity})', f'rgba(128, 0, 128 ,{opacity})',
                         f'rgba(0, 0, 0 ,{opacity})', f'rgba(0, 128, 128 ,{opacity})', ]

        return static_colors

    fig = go.Figure()
    annotations = mspconv['annotations']
    reps_score = mspconv['reps_score']
    for i, annotation in enumerate(annotations):
        label = str(annotation.Annotator.iloc[0])
        opacity = 0.9
        if label == WEIGHT_AVG_LABEL:
            fig.add_scatter(x=annotation.Time, y=annotation.Annotation, line=dict(color='rgba(217, 9, 92, 1)'))
        else:
            label = f"{label} \t{np.round(reps_score[label], 4)}"
            opacity = 0.4
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
        autosize=False, width=1300, height=500,
        title_text=f'Annotations on MSP-Conversation_{key}', title_x=0.5,
    )

    fig.show()


def save_mspconvs(mspconvs: dict):
    with open(f'../data/{MSP_PKL_FILE}', "wb") as pkl:
        pickle.dump(mspconvs, pkl)


def get_wadf(key: str = ""):
    """
    Para sacar un dict con el WA
    :param key: si lo mandas un key entonces retorna el dict completo
    :return:
    """
    wadf = {}
    try:
        with open(f"./data/wadf.pkl", "rb") as pkl:
            wadf = pickle.load(pkl)
    except Exception as ex:
        mspconv_from_disk = {}
        with open(f"./data/{MSP_PKL_FILE}", "rb") as pkl:
            mspconv_from_disk = pickle.load(pkl)
        for k in mspconv_from_disk.keys():
            _mspconv = mspconv_from_disk[k]
            for purge in ["raters", "ck_score", "reps_score", "reps_scaled"]:
                _mspconv.pop(purge, None)
            df = _mspconv['annotations'][6] # WA dataframe
            wadf[k] = df
        del mspconv_from_disk
        with open(f'./data/wadf.pkl', "wb") as pkl:
            pickle.dump(wadf, pkl)

    return wadf if key == "" else wadf[key]


def get_wa(pc_num: int, part_num: int, mapping: str = "Russell_Mehrabian") -> (dict, dict):
    """
    Esta es una version NO DEFINITIVA para sacar el WA dado un pc_num y una part_num
    TODO: Se debe encontrar una forma efectiva ya que en esta func ajusta el tamano de los datapoints del WA al de menor numero de datapoints.
    :param mapping: Russell_Mehrabian or Ekman
    :param pc_num:
    :param part_num:
    :return:
    """
    emotions = ['Valence', 'Arousal', 'Dominance']
    wadf = get_wadf()
    vad_mapping = VAD(minmax=[-100, 100], mapping=mapping)
    wa = {}
    for emotion in emotions:
        key = f"{pc_num}_{part_num}_{emotion}"
        wa[emotion] = [wadf[key]['Annotation'].fillna(0).to_numpy(), wadf[key]['Time'].fillna(0).to_numpy()]
    mi = np.argmin([wa['Valence'][0].shape[0], wa['Arousal'][0].shape[0], wa['Dominance'][0].shape[0], ])
    threshold = wa[emotions[mi]][0].shape[0]
    for emotion in emotions:
        wa[emotion][0] = wa[emotion][0][:threshold]
        wa[emotion][1] = wa[emotion][1][:threshold]

    c = 0
    wa['categorical'] = []
    for v, a, d in zip (wa['Valence'][0], wa['Arousal'][0], wa['Dominance'][0]):
        r = vad_mapping.vad2categorical(v, a, d, k=3)
        # wa['categorical'].append({'term': r[0][0]['term'], 'closest': r[0][0]['closest']})
        wa['categorical'].append(r[0])

    # TODO: se debe mejorar esto !
    timed_terms = {} # fuerza bruta, si hay: 0.001 term: 'Love', 0.05 term: 'Hate' lo que se queda es 0.1 term 'Hate'
    for k, t in enumerate(wa['Arousal'][1]):
        timed_terms[np.around(t, 1)] = wa['categorical'][k]

    wa['pc_num'] = pc_num
    wa['part_num'] = part_num
    wa['mapping'] = mapping

    return wa, timed_terms


def read_wav_segment_to_base64(file_path, start_time_ms, end_time_ms):
    # Load the WAV file
    audio = AudioSegment.from_file(file_path, format="wav")

    # Extract the specified segment
    segment = audio[start_time_ms:end_time_ms]

    # Convert the segment to base64
    segment_bytes = segment.export(format="wav").read()
    base64_encoded = base64.b64encode(segment_bytes).decode("utf-8")

    return base64_encoded

###############################    AUDIO VIZ COMPONENT    ###############################
############################### Setting up plotly.js URL  #################################
plotly_lib_url = "https://cdn.plot.ly/plotly-2.25.2.min.js"
wavesurfer_lib_url = "https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/2.0.4/wavesurfer.min.js"

audio_viz_js = """
function audio_viz(div_id, data){    
  var valence = {
    x: data.Time,
    y: data.Valence,
    type: 'scatter',
    name: 'Valence',
  };

  var arousal = {
    x: data.Time,
    y: data.Arousal,
    type: 'scatter',
    name: 'Arousal',
  };

  var dominance = {
    x: data.Time,
    y: data.Dominance,
    type: 'scatter',
    name: 'Dominance',
  };

  var traces = [valence, arousal, dominance];

  var layout = {
    shapes: [{
      type: 'line',
      x0: 100,
      y0: 0,
      x1: 100,
      yref: 'paper',
      y1: 1,
      line: {
        color: 'grey',
        width: 1.5,
        dash: 'dot'
      },
      label: {
          text: ' ',
          textangle: 0,
          textposition: 'end',
          xanchor: 'left',
          font: {
              color: '#5e62b5',
              size: 14,
          },
      },
    }],
    title: { 
        text: `Audio VAD to Categ Viz [${data.mapping}] @ MSP-Conversation_${data.PC_Num}_${data.Part_Num}`,
    },
    showlegend: true,
  };

  nid = document.querySelector(div_id);
  Plotly.newPlot(nid, traces, layout);

  const container = document.createElement("div");
  container.style.width = "1090px";
  container.style.marginLeft = "80px";
  container.id = "waveform";
  nid.appendChild(container);

  var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: "#9195f2",
    progressColor: "#c7c7c7",
    cursorColor: "#ea8b3e",
    cursorWidth: 2,
    barWidth: 3,
    barRadius: 30,
  });

  function round_f(num) {
      return Math.round((num + Number.EPSILON) * 10) / 10;
  }

  function refresh(position) {
    var num = round_f(position)
    var term = data.timed_terms[num.toString()]
    // console.log(term[0].term)
    nid.layout.shapes[0].x0 = position;
    nid.layout.shapes[0].x1 = position;
    if (term) {
        var r = `<br />${term[0].term} ${round_f(term[0].closest)}<br />${term[1].term} ${round_f(term[1].closest)}<br />${term[2].term} ${round_f(term[2].closest)}`
        nid.layout.shapes[0].label.text = r
        // nid.layout.shapes[0].label.text = `🤔 ${term.term} ${round_f(term.closest)}`
    }
    Plotly.redraw(nid);
  }

  const button = document.createElement("input");
  button.type = "button";
  button.value = "> play";
  button.style.paddingLeft = "80px";
  button.style.marginLeft = "80px";
  nid.appendChild(button)

  const indicator = document.createElement("div");
  indicator.class = "waveform-time-indicator";
  const time = document.createElement("span");
  time.innerHTML = "00:00:00";
  time.style.paddingLeft = "80px";
  time.class = "time";
  indicator.appendChild(time);
  nid.appendChild(indicator);  

  function base64ToArrayBuffer(base64) {
    var binaryString = atob(base64);
    var bytes = new Uint8Array(binaryString.length);
    for (var i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  const arrBuffAudio = base64ToArrayBuffer(data.audio);
  wavesurfer.loadArrayBuffer(arrBuffAudio);

  function playPause() {
    wavesurfer.playPause();
    if (wavesurfer.isPlaying()) {
        button.value = "!! pause";
    } else {
        button.value = "> play";
    }
  }

  button.addEventListener("click", playPause);

  function secondsToTimestamp(seconds) {
    seconds = Math.floor(seconds);
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds - (h * 3600)) / 60);
    var s = seconds - (h * 3600) - (m * 60);

    h = h < 10 ? '0' + h : h;
    m = m < 10 ? '0' + m : m;
    s = s < 10 ? '0' + s : s;

    return h + ':' + m + ':' + s;
  }

  function updateTimer() {
    var formattedTime = secondsToTimestamp(wavesurfer.getCurrentTime());
    time.innerHTML = formattedTime;
    const currentpos = wavesurfer.getCurrentTime();
    refresh(currentpos);
  }

  wavesurfer.on('ready', updateTimer)
  wavesurfer.on('audioprocess', updateTimer)
  wavesurfer.on('seek', updateTimer)

}
"""


def audio_viz(wa: dict, timed_terms: dict):
    # Code execution using notebookjs
    from notebookjs import execute_js
    import json
    import base64

    emotion = 'Valence' #use for TIME reference

    audio = f"/Users/beltre.wilton/Downloads/SER-Datasets/MSP-Conversation-1.1/Audio/MSP-Conversation_{str(wa['pc_num']).zfill(4)}.wav"
    # Solo impl para la primera parte del audio, por ahora.
    start_time = 0.0 * 1000
    end_time = int(wa[emotion][1][-1]) * 1000

    # Read the WAV segment and convert to base64
    encode_string = read_wav_segment_to_base64(audio, start_time, end_time)

    execute_js([plotly_lib_url, wavesurfer_lib_url, audio_viz_js], "audio_viz", {
        "Valence": wa['Valence'][0].tolist(),
        "Arousal": wa['Arousal'][0].tolist(),
        "Dominance": wa['Dominance'][0].tolist(),
        "Time": wa[emotion][1].tolist(),
        "audio": encode_string,
        "PC_Num": str(wa['pc_num']).zfill(4),
        "Part_Num": wa['part_num'],
        "mapping": wa['mapping'],
        "timed_terms": timed_terms,
    })



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
    # with open(f"../data/{MSP_PKL_FILE}", "rb") as pkl:
    #     mspconv_from_disk = pickle.load(pkl)

    wadict = get_wa(197, 1)

    # print(len(mspconv_from_disk.keys()))

    # wadf = get_wadf('197_1_Arousal')
    audio_viz(wadict, pc_num=197, part_num=1, emotion='Arousal')
    print()

    # for key, _msp_conv in mspconv_from_disk.items():
    #     for annotation in _msp_conv['annotations']:
    #         if annotation.Time.max() > 500:
    #             idx = annotation.Time.argmax()
    #             print(f"Annotator: {annotation.Annotator.iloc[idx]}\tpoint: {annotation.Annotation.iloc[idx]}\tkey: {key}\tTime: {annotation.Time.max()}")

    reps_score = {}
    for key, _msp_conv in mspconv_from_disk.items():
        for k, s in _msp_conv['reps_score'].items():
            a = reps_score.get(k, np.array([]))
            reps_score[k] = np.append(a, s)

    for annotator, arr in reps_score.items():
        print(f"Anotador: {annotator}, Mean: {arr.mean()}, Std: {arr.std()}, Sum: {arr.sum()}, Participación: {arr.shape[0]}")



    # key = "197_1_Dominance"
    # _mspconv = mspconv_from_disk[key]
    # sample_scatter(_mspconv, key)
    #
    # key = "197_2_Dominance"
    # _mspconv = mspconv_from_disk[key]
    # sample_scatter(_mspconv, key)
    #
    key = "197_1_Arousal"
    _mspconv = mspconv_from_disk[key]
    sample_scatter(_mspconv, key)

    # sample_heatmap(_mspconv, key)


if __name__ == "__main__":
    # df_annotations, df_reduced = get_annotated_data()
    # mspconvs = build_mspconvs(df_annotations, df_reduced)
    #
    #TODO: rethink names of variables and method
    #TODO: solucionar que el audio play solo anda con la parte # 1 del audio
    #TODO: EDA with this data/rates/etc
    #TODO: BIG NEXT is according with conclusion from EDA try to reduce the size by quality choose
    #TODO: BIG NEXT is also split audio by 30 seconds

    # save_mspconvs(mspconvs)

    play()


    print('Fin de proceso ⛱️.')
