import sys
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import math
import pickle
from tqdm import tqdm
from vad.vad_lab import VAD
MSP_PATH = '/Users/beltre.wilton/Downloads/SER-Datasets/MSP-Conversation-1.1'
ROOT = Path(__file__).parent.parent.__str__()
sys.path.append(ROOT)
AUDIO_SEGMENTS = f'{ROOT}/audiosegments'
AUDIO_PARTS = f'{ROOT}/audioparts'
OUTPUT_DIR = f'{ROOT}/checkpoint'
WADF_PATH = f'{ROOT}/data/wadf.pkl'
ANNO_FILE_2 = f'{ROOT}/data/annotations_2.xlsx'

os.makedirs(AUDIO_SEGMENTS, exist_ok=True)
os.makedirs(AUDIO_PARTS, exist_ok=True)

vad = VAD(minmax=[-100, 100], mapping="OCC")

# USE_ONNX = False
# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                               model='silero_vad',
#                               force_reload=False,
#                               onnx=USE_ONNX)
model, utils = None, None


def get_annotated_rdata() -> (pd.DataFrame):
    df_annotations = pd.read_excel(ANNO_FILE_2)
    df_reference = df_annotations.drop(columns=['Name', 'Emotion', 'Annotator'])
    df_reference = df_reference.drop_duplicates()
    return df_reference


def get_wadf() -> dict:
    wadf = {}
    try:
        with open(WADF_PATH, "rb") as pkl:
            wadf = pickle.load(pkl)
    except Exception as ex:
        print(ex)
    return wadf

medium = {"aa": 1, "ae": 2, "ah": 3, "aw": 4, "ay": 5, "b": 6, "ch": 7, "d": 8, "dh": 9, "dx": 10, "eh": 11, "er": 12, "ey": 13, "f": 14, "g": 15, "h#": 16, "hh": 17, "ih": 18, "iy": 19, "jh": 20, "k": 21, "l": 22, "m": 23, "ng": 24, "|": 0, "[UNK]": 25, "[PAD]": 26}
get_medium = lambda x: list(medium.keys())[list(medium.values()).index(x)]


class MSPDataProcessor:
    def __init__(self, msp_path: str, chunk_size: int, overlap: float, split: str, verbose: bool = False,):
        """
        :param msp_path: ruta absoluta a los audios wav del MSP-Conversation
        :param chunk_size: en segundos, tamano considerado del corte de audio, ej.
        :param split: filtro Train, Test, Development
        :param verbose: para imprimir eventos
        """
        self.msp_path = msp_path
        self.wadf = None
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.df_reference = None
        self.split = split
        self.input_features_path = f"class_input_features_{self.split.lower()}_balanced.pkl"
        self.verbose = verbose
        self.SAMPLE_RATE = 16_000
        self.TEMPERATURE_DATAPOINT = .5

        del self.wadf
        del self.df_reference

    def load_input_features(self, limit: int = -1):
        input_features = {}
        path_file = f'{ROOT}/data/{self.input_features_path}'
        if Path(path_file).is_file(): # soft load (pre-processed)
            print(f"\n####################################################################################################\n"
                  f"Inputs pre-loaded: {path_file}\n######################################################################"
                  f"##############################\n")
            try:
                with open(path_file, "rb") as pkl:
                    input_features = pickle.load(pkl)
                    print(f"inputs:{len(input_features['inputs'])}, labels:{len(input_features['labels'])}")
            except Exception as ex:
                print(ex)
            return input_features
        else: # heavy loads...
            print(
                f"\n####################################################################################################\n"
                f"Processing {self.split} Inputs...\n######################################################################"
                f"##############################\n")
            self.wadf = get_wadf()
            df_reference = get_annotated_rdata()
            limit = limit if limit != -1 else df_reference[df_reference.Type == self.split].shape[0]
            # self.df_reference = df_reference[df_reference.Type == self.split][:limit]
            self.df_reference = df_reference[df_reference.Type == self.split].sample(limit)
            # self.df_reference = df_reference[df_reference.Audio_Name == "MSP-Conversation_1184.wav"]
            return self.__prepare_input_features()

    def __save_input_features(self, dataset_object, name):
        with open(f'{ROOT}/data/{name}', "wb") as pkl:
            pickle.dump(dataset_object, pkl)

    def __get_wave_segment(self, waveform, key: str):
        """
        corta un segmento o parte de audio de acuerdo al tiempo de inicio-fin
        :param waveform: tensor del audio wav
        :param key: parte del audio, ej. 498_1_Arousal
        :return:
        """
        key = key.split('_')
        pc_num = int(key[0])
        part_num = int(key[1])
        ref = self.df_reference[(self.df_reference['PC_Num'] == pc_num) & (self.df_reference['Part_Num'] == part_num)]
        start_time = math.ceil(ref.start_time.iloc[0] * self.SAMPLE_RATE)  #TODO ceil workaround
        end_time = math.ceil(ref.end_time.iloc[0] * self.SAMPLE_RATE)  #TODO ceil workaround
        from_shape = waveform.shape
        waveform = waveform[start_time:end_time]
        return waveform, ref.start_time.iloc[0], ref.end_time.iloc[0]

    def __check_for_resample(self, waveform, sample_rate: int, aud: str):
        """
        :param waveform: tensor del audio wav
        :param sample_rate: debe ser 16,0000 (Hz) o frames por 1 segundo
        :param aud: ruta del audio en disco
        :return:
        """
        # Si el sample rate no es 16000 entonces resample
        if sample_rate != self.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.SAMPLE_RATE)
            if self.verbose:
                print(f'\t{aud} Resampleando!')
        # Si es estereo (2 canales) pues convierte a mono-canal
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)
            if self.verbose:
                print(f'\t{aud} Bajando a mono!')
        return waveform

    def __prepare_audio_inputs(self, chunked_data_points: dict, dump_to_disk: bool = False):
        """
        corta parte del audio (ej. 498_2_Arousal) en segmentos de chunk_size (ej. 15 segundos)
        :param chunked_data_points: contiene las partes de audio que se van a procesar
        :param dump_to_disk: guardar a disco cada segmento
        :return: waves que es dict con el mapping a todas las partes procesadas
        """
        ## reset file dir ##
        audiosegments = Path(AUDIO_SEGMENTS)
        for a in audiosegments.rglob(f"*{self.split}.wav"):
            a.unlink(missing_ok=True)

        waves = {}
        ran = chunked_data_points.keys()
        iter_tqdm = tqdm(ran, desc=f"[{self.split}] input segments {self.chunk_size}s...", total=len(chunked_data_points.keys()), ncols=130)
        for idx, key in enumerate(iter_tqdm):
            aud = f"MSP-Conversation_{key.split('_')[0].rjust(4, '0')}_{key.split('_')[1]}.wav"
            aud_path = f"{AUDIO_PARTS}/{aud}"
            # torchaudio.info(aud_path)
            # waveform, sample_rate = torchaudio.load(str(aud_path), normalize=True, bits_per_sample=16, encoding='PCM_S')
            waveform, sample_rate = torchaudio.load(str(aud_path), normalize=True)
            waveform = self.__check_for_resample(waveform, sample_rate, aud)
            waveform = waveform.squeeze()  # [c, s] => [s]
            # waveform, start_time, end_time = self.__get_wave_segment(waveform, key)
            # upto = math.ceil((waveform.size(0) / self.SAMPLE_RATE))
            chunked_parts = []
            for j, i in enumerate(chunked_data_points[key]['cuts']):
                start_time = i['start']
                rst = start_time if start_time < 1 else 0 if start_time == 0 else np.round(start_time % int(start_time), 4)
                start_time = (int(start_time) * self.SAMPLE_RATE) + int(rst * self.SAMPLE_RATE)

                end_time = i['end']
                ret = np.round(end_time % int(end_time), 4)
                end_time = (int(end_time) * self.SAMPLE_RATE) + int(ret * self.SAMPLE_RATE)

                wave_chunked = waveform[start_time:end_time]
                p = f"{key}_{j}_{i['start']}_{i['end']}_{self.split}.wav"
                # dump here to disk
                torchaudio.save(f"{AUDIO_SEGMENTS}/{p}", wave_chunked.unsqueeze(dim=0), sample_rate=sample_rate,
                                bits_per_sample=16, encoding='PCM_S')
                chunked_parts.append(p)
            waves[key] = chunked_parts

        return waves

    def __chunk_datapoints(self, pc_num: int, part_num: int ):
        """
        corta en segmentos de chunk_size (ej. 15 segundos) guiado por el tiempo del DataFrame
        la longitud NO es necesatiamente la misma entre cada chunk
        :param key: parte del audio, ej. 498_1_Arousal
        :return:
        """
        ref = self.df_reference[(self.df_reference['PC_Num'] == pc_num) & (self.df_reference['Part_Num'] == part_num)]

        # 1ro hay que cortar el audio para darle a silero
        start_time = ref.start_time.iloc[0]
        rst = 0 if start_time == 0 else np.round(start_time % int(start_time), 4)
        start_time = (int(start_time) * self.SAMPLE_RATE) + int(rst * self.SAMPLE_RATE)

        end_time = ref.end_time.iloc[0]
        ret = np.round(end_time % int(end_time), 4)
        end_time = (int(end_time) * self.SAMPLE_RATE) + int(ret * self.SAMPLE_RATE)

        # cargar archivo
        audio_name = ref.Audio_Name.values[0]
        audio_file = f'{self.msp_path}/Audio/{audio_name}'
        waveform, sample_rate = torchaudio.load(str(audio_file), normalize=True)
        waveform = self.__check_for_resample(waveform, sample_rate, audio_name)
        waveform = waveform.squeeze() # [c, s] => [s]
        wave_chunked = waveform[start_time:end_time]

        # guardar para silero
        audio_name = audio_name.replace(".wav", f"_{part_num}.wav")
        try:
            torchaudio.save(f"{AUDIO_PARTS}/{audio_name}", wave_chunked.unsqueeze(dim=0), sample_rate=self.SAMPLE_RATE,
                            bits_per_sample=16, encoding='PCM_S')
        except Exception as ex:
            print(ex)


        # llama a silero y preguntale por los cuts de esta parte de audio
        cuts = self.silero_cut(f"{AUDIO_PARTS}/{audio_name}")


        # iterar por cuts .....

        data_points = {'Valence': [], 'Arousal': [], 'Dominance': []}
        for emo in data_points.keys():
            key = f"{pc_num}_{part_num}_{emo}"
            wak = self.wadf[key]
            l = self.wadf[key].iloc[-1]['Time']

            # for i in range(0, int(l), self.chunk_size):
            for i in cuts:
                a = []
                wa = wak[(wak['Time'] > (i['start'] - self.overlap)) & (wak['Time'] < (i['end'] + self.overlap))]
                if len(wa) == 0:
                    continue
                # inner time
                try:
                    v = wa.iloc[0]['Time']
                    m = wa.iloc[-1]['Time']
                except Exception as ex:
                    print(ex)
                # M = 0.25 # cada 0.25 segundos saco un label 4 x 1seg
                # M = 0.50 # cada 0.5 segundos saco un label 2 x 1seg
                # M = 1.0 # cada 1 seg con un M_overlap
                # M = 0.1 # timit rules!
                # M_overlap = 0.0025
                # for j in np.arange(v, math.ceil(m), M):
                #     mwa = wa[(wa['Time'] >= (j - M_overlap)) & (wa['Time'] < (j + M + M_overlap))]
                #     if len(mwa) == 0:
                #         continue
                #     p = np.nan_to_num(mwa['Annotation'].to_numpy()).mean()
                #     a.append(np.round(p, 4))
                # mwa = wa[(wa['Time'] > j + M)] # algo se escapo?
                # if len(mwa) > 1:
                #     p = np.nan_to_num(mwa['Annotation'].to_numpy()).mean()
                #     a.append(np.round(p, 4))
                a = np.nan_to_num(wa['Annotation'].to_numpy()).mean()
                try:
                    data_points[emo].append([a])
                except Exception as ex:
                    print(ex)

        data_points['cuts'] = cuts
        return data_points

    def __prepare_datapoints(self):
        """
        Ejercicio de cortar las partes-de-audio cada `chunk_size` segundos
        prepara los datapoints/anotaciones del weighted average con las siguientes condiciones:
        :return: chunked_data_points
        """
        audioparts = Path(AUDIO_PARTS)
        for a in audioparts.rglob(f"*.wav"):
            a.unlink(missing_ok=True)

        chunked_data_points = {}
        ran = zip(self.df_reference['PC_Num'], self.df_reference['Part_Num'])
        iter_tqdm = tqdm(ran, desc=f"[{self.split}] labels datapoints... ", total=len(self.df_reference['PC_Num']), ncols=130)
        for pc_num, part_num in iter_tqdm:
            key = f"{pc_num}_{part_num}"
            data_points = self.__chunk_datapoints(pc_num, part_num)
            chunked_data_points[key] = data_points

        return chunked_data_points

    def silero_cut(self, msp_wave: str, CUT_NEAR: int = 3, SAMPLING_RATE: int = 16_000, silence_threshold: float = 0.100, length: int = 6) -> dict:
        torch.set_num_threads(1)

        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils

        wav = read_audio(msp_wave, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(wav, model,
                                                  sampling_rate=SAMPLING_RATE,
                                                  min_silence_duration_ms=100
                                                  )
        start, end, z = 0, 0, 0
        jumps = {}
        for i, t in enumerate(speech_timestamps):
            start = t['start'] / SAMPLING_RATE
            silence = start - end
            end = t['end'] / SAMPLING_RATE
            jumps[i] = {'silence': silence, 'start': start, 'end': end, 'cut': False}
            tolerance = end - z
            if tolerance >= CUT_NEAR and silence >= silence_threshold:
                # (silence >= silence_threshold
                #   or get the next end_of_sentence whit stable_ts
                #   in a given range of time, ex. 15 seconds)
                z = end
                jumps[i]['cut'] = True

        cuts = {}
        cuts[0] = {'start': jumps[0]['start'], 'end': jumps[0]['end'], 'length': -1, 'silence': jumps[0]['silence'],
                   'cut': jumps[0]['cut']}
        idx = 0
        for i in range(1, len(jumps)):
            if jumps[0]['cut'] and i == 1:
                cuts[idx]['length'] = cuts[idx]['end'] - cuts[idx]['start']
                idx += 1

            if jumps[i]['cut'] and jumps[i - 1]['cut']:
                # q[idx - 1]['end'] = a[i - 1]['end']
                cuts[idx] = {'start': jumps[i]['start'], 'end': -1, 'length': -1, 'silence': jumps[i]['silence']}
                # idx += 1

            if jumps[i]['cut']:
                cuts[idx]['end'] = jumps[i]['end']
                cuts[idx]['length'] = cuts[idx]['end'] - cuts[idx]['start']
                idx += 1  # define un grupo 1, 2, 3, 4, .... N

            if jumps[i - 1]['cut'] and not jumps[i]['cut']:
                cuts[idx] = {'start': jumps[i]['start'], 'end': -1, 'length': -1, 'silence': jumps[i]['silence']}
                idx = len(cuts) - 1

        if cuts[len(cuts) - 1]['end'] == -1:
            cuts[len(cuts) - 1]['end'] = jumps[i]['end']
            cuts[len(cuts) - 1]['length'] = cuts[len(cuts) - 1]['end'] - cuts[len(cuts) - 1]['start']

        return [c[1] for c in cuts.items() if c[1]['length'] <= length]


    def __prepare_input_features(self, ):
        """
        une los metodos de la classe, hace una validacion en tamano, y transforma la data en un dict de inputs y labels
        :return: input_features
        """
        # ran = zip(self.df_reference['PC_Num'], self.df_reference['Part_Num'])
        # iter_tqdm = tqdm(ran, desc=f"[{self.split}] silero... ", total=len(self.df_reference['PC_Num']),
        #                  ncols=130)
        # for pc_num, part_num in iter_tqdm:
        #     emos = {'Valence': [], 'Arousal': [], 'Dominance': []}
        #     for emo in emos.keys():
        #         key = f"{pc_num}_{part_num}_{emo}"
        #         print()
        #         # cuts = self.silero_cut()


        if len(self.df_reference) == 0:
            raise Exception(f'Parece que el split: {self.split} es incorrecto, intenta con Train, Test o Development')

        chunked_data_points = self.__prepare_datapoints()
        waves = self.__prepare_audio_inputs(chunked_data_points, dump_to_disk=True)

        labels_dp = {}
        for key in chunked_data_points.keys():
            cdp = []
            ii = min(len(chunked_data_points[key]['Valence']), len(chunked_data_points[key]['Arousal']),
                     len(chunked_data_points[key]['Dominance']))
            for i in range(ii):
                t = []
                jj = min(len(chunked_data_points[key]['Valence'][i]), len(chunked_data_points[key]['Arousal'][i]),
                    len(chunked_data_points[key]['Dominance'][i]))
                for j in range(jj):
                    t.append((chunked_data_points[key]['Valence'][i][j], chunked_data_points[key]['Arousal'][i][j], chunked_data_points[key]['Dominance'][i][j]))
                cdp.append(t)
            labels_dp[key] = cdp

        if self.verbose:
            ok = True
            for w in waves.keys():
                if len(waves[w]) != len(labels_dp[w]):
                    print(f'\n*** FAIL [{w}]:  Inputs y Labels waves:{len(waves[w])} datapoints:{len(labels_dp[w])}')
                    waves[w] = waves[w][:len(labels_dp[w])]
                    print(f'  -> FIXED [{w}]:  Inputs y Labels waves:{len(waves[w])} datapoints:{len(labels_dp[w])}')
                    ok = False
            if ok:
                print('\nInputs y Labels se corresponden en catidad de segmentos.\n')

        inputs = []
        labels = []
        for key in labels_dp.keys():
            for wv, dp in zip(waves[key], labels_dp[key]):
                inputs.append(wv)
                # cat = [get_medium(vad.vad2categorical(*vals, k=1, use_plot=False)[0][0]['index']) for vals in dp]
                cat = [vad.vad2categorical(*vals, k=1, use_plot=False)[0][0]['index'] for vals in dp]
                # labels.append(" ".join(cat))
                labels.append(cat)
        input_features = {'inputs': inputs, 'labels': labels}
        if self.verbose:
            print(
                f"input_features listos, {len(input_features['inputs'])} inputs-tensores y {len(input_features['labels'])} labels-np.arrays\n")

        self.__save_input_features(input_features, self.input_features_path)
        return input_features


if __name__ == "__main__":
    chunk_size = 3
    overlap = 0.10
    dataprocessor = MSPDataProcessor(msp_path=MSP_PATH, chunk_size=chunk_size, overlap=overlap, split="Train", verbose=True)
    dataprocessor.load_input_features()

    dataprocessor = MSPDataProcessor(msp_path=MSP_PATH, chunk_size=chunk_size, overlap=overlap, split="Development", verbose=True)
    dataprocessor.load_input_features()

    dataprocessor = MSPDataProcessor(msp_path=MSP_PATH, chunk_size=chunk_size, overlap=overlap, split="Test", verbose=True)
    dataprocessor.load_input_features()
