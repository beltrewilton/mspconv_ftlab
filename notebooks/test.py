import sys
import os
import pickle
import blosc

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

ROOT = '/Users/beltre.wilton/apps/mspconv_ftlab'
MSP_PATH = '/Users/beltre.wilton/Downloads/SER-Datasets/MSP-Conversation-1.1'

sys.path.append(ROOT)
from src.dataload import get_annotated_data

os.environ['HF_HOME'] = f'{ROOT}/cache'
os.environ['HF_DATASETS_CACHE'] = f'{ROOT}/cache'

AUDIO_SEGMENTS = f'{ROOT}/audiosegments'
OUTPUT_DIR = f'{ROOT}/checkpoint'
WADF_PATH = f'{ROOT}/data/wadf.pkl'

df_annotations, df_reduced = get_annotated_data()

os.makedirs(AUDIO_SEGMENTS, exist_ok=True)

from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
import math
import torch
import torchaudio
from torch.utils.data import Dataset
from IPython.display import display, Audio
from vad.vad_lab import VAD

vad = VAD(minmax=[-100, 100], mapping="OCC")

wadf = {}
try:
    with open(WADF_PATH, "rb") as pkl:
        wadf = pickle.load(pkl)
except Exception as ex:
    print(ex)


def save_data_compressed(dataset_object, name):
    pickled_data = pickle.dumps(dataset_object)
    compressed_pickle = blosc.compress(pickled_data)
    with open(f'{ROOT}/data/{name}', "wb") as f:
        f.write(compressed_pickle)


def save_data(dataset_object, name):
    with open(f'{ROOT}/data/{name}', "wb") as pkl:
        pickle.dump(dataset_object, pkl)


class MSPDataset(Dataset):
    def __init__(self, msp_path: str, wadf: dict, chunk_size: int, df_reference: pd.DataFrame, split: str,
                 input_features_path: str = None, filter_emotion_by: str = 'Arousal', verbose: bool = False,):
        """
        :param msp_path: ruta absoluta a los audios wav del MSP-Conversation
        :param wadf: weighted average previamente calculado, es un dict de dataframes de pandas
        :param chunk_size: en segundos, tamano considerado del corte de audio, ej.
        :param df_reference: usar get_annotated_data(), dataframe pandas con cada parte de audio y el tiempo de inicio-fin
        :param split: filtro Train, Test, Development
        :param filter_emotion_by: por ahora filtrar por una DIM
        :param verbose: para imprimir eventos
        """
        self.msp_path = msp_path
        self.wadf = wadf
        self.chunk_size = chunk_size
        self.df_reference = df_reference[df_reference.Type == split] #[:500]
        self.split = split
        self.filter_emotion_by = filter_emotion_by
        self.device = device
        self.verbose = verbose
        self.SAMPLE_RATE = 16_000
        self.TEMPERATURE_DATAPOINT = .5
        # self.__filter_df()
        self.input_features = self.__prepare_inputs() if input_features_path is None else self.__load_input_features(input_features_path)
        del self.wadf
        del self.df_reference


    def __filter_df(self):
        """
        Reduce el tamano del `self.wadf` acorde al split
        :return:
        """
        keys = [f"{pc_num}_{part_num}_{self.filter_emotion_by}" for pc_num, part_num in
                zip(self.df_reference['PC_Num'], self.df_reference['Part_Num'])]
        self.wadf = {key: self.wadf[key] for key in keys}


    def __load_input_features(self, input_features_path: str):
        input_features = {}
        try:
            with open(f'{ROOT}/data/{input_features_path}', "rb") as pkl:
                input_features = pickle.load(pkl)
        except Exception as ex:
            print(ex)
        return input_features


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

        audio_path = Path(f'{self.msp_path}/Audio')
        waves = {}
        if self.verbose:
            print(f'Inicio de lectura de audios wav y segmentacion cada {self.chunk_size} segundos...')
        for idx, key in enumerate(chunked_data_points.keys()):
            aud = f"MSP-Conversation_{key.split('_')[0].rjust(4, '0')}.wav"
            aud_path = audio_path / aud
            # torchaudio.info(aud_path)
            # waveform, sample_rate = torchaudio.load(str(aud_path), normalize=True, bits_per_sample=16, encoding='PCM_S')
            waveform, sample_rate = torchaudio.load(str(aud_path), normalize=True)
            waveform = self.__check_for_resample(waveform, sample_rate, aud)
            waveform = waveform.squeeze()  # [c, s] => [s]
            waveform, start_time, end_time = self.__get_wave_segment(waveform, key)
            upto = math.ceil((waveform.size(0) / self.SAMPLE_RATE))
            chunked_parts = []
            for j, i in enumerate(range(0, upto, self.chunk_size)):
                start = i * self.SAMPLE_RATE
                end = (i + self.chunk_size) * self.SAMPLE_RATE
                wave_chunked = waveform[start:end]
                if dump_to_disk:
                    p = f"{key}_{j}_{self.split}.wav"
                    # dump here to disk
                    torchaudio.save(f"{AUDIO_SEGMENTS}/{p}", wave_chunked.unsqueeze(dim=0), sample_rate=sample_rate, bits_per_sample=16, encoding='PCM_S')
                    chunked_parts.append(p)
                else:
                    chunked_parts.append(wave_chunked)
            waves[key] = chunked_parts

            try:
                if self.verbose:
                    print(
                        f"  ({idx + 1}) {aud}; {key}[{round(start_time, 2)}:{round(end_time, 2)}] {len(chunked_parts)} segmentos")
            except Exception as ex:
                print(ex)

        return waves

    def __chunk_datapoints(self, pc_num: int, part_num: int ):
        """
        corta en segmentos de chunk_size (ej. 15 segundos) guiado por el tiempo del DataFrame
        la longitud NO es necesatiamente la misma entre cada chunk
        :param key: parte del audio, ej. 498_1_Arousal
        :return:
        """
        data_points = {'Valence': [], 'Arousal': [], 'Dominance': []}
        for emo in data_points.keys():
            key = f"{pc_num}_{part_num}_{emo}"
            wak = self.wadf[key]
            # data_points = []
            l = self.wadf[key].iloc[-1]['Time']
            for i in range(0, int(l), self.chunk_size):
                a = []
                wa = wak[(wak['Time'] > i) & (wak['Time'] < i + self.chunk_size)]
                if len(wa) == 0:
                    continue
                #micro time
                try:
                    v = wa.iloc[0]['Time']
                    m = wa.iloc[-1]['Time']
                except Exception as ex:
                    print(ex)
                M = 0.25 # cada 0.25 segundos saco un label 4 x 1seg
                for j in np.arange(v, math.ceil(m), M):
                    mwa = wa[(wa['Time'] >= j) & (wa['Time'] < j + M)]
                    if len(mwa) == 0:
                        continue
                    p = np.nan_to_num(mwa['Annotation'].to_numpy()).mean()
                    a.append(np.round(p, 2))
                mwa = wa[(wa['Time'] > j + M)] # algo se escapo?
                if len(mwa) > 1:
                    p = np.nan_to_num(mwa['Annotation'].to_numpy()).mean()
                    a.append(np.round(p, 2))
                try:
                    data_points[emo].append(a)
                except Exception as ex:
                    print(ex)
                # a = np.append(a, len(wa))
                # data_points.append(np.nan_to_num(wa['Annotation'].to_numpy()))
            # wa = wak[(wak['Time'] > i + self.chunk_size)]
            # if len(wa) > 1:
            #     a = np.append(a, len(wa))
            #     data_points.append(np.nan_to_num(wa['Annotation'].to_numpy()))
        # return [key, a.min(), a.mean(), a.max()], data_points
        return data_points

    ## Ejercicio de cortar las partes-de-audio cada `chunk_size` segundos
    def __prepare_datapoints(self):
        """
        prepara los datapoints/anotaciones del weighted average con las siguientes condiciones:
        (1) `mu`: es la media de la cantidad de datapoints/anotaciones por `chunk` * un `TEMPERATURE_DATAPOINT`
        (2) filtro#1: el MIN de los datapoints/anotaciones DEBE ser mayor-igual que `mu`
        (3) filtro#2: reduce por DIM emotion (filter_emotion_by)
        :return: chunked_data_points
        """
        # data = []
        chunked_data_points = {}
        # for key in self.wadf.keys():
        for pc_num, part_num in zip(self.df_reference['PC_Num'], self.df_reference['Part_Num']):
            key = f"{pc_num}_{part_num}"
            # a, data_points = self.__chunk_datapoints(pc_num, part_num)
            data_points = self.__chunk_datapoints(pc_num, part_num)
            # data.append(a)
            chunked_data_points[key] = data_points
        # wa_data_points = pd.DataFrame(data, columns=['Key', 'Min', 'Mean', 'Max'])

        # mu = wa_data_points.Mean.mean() * self.TEMPERATURE_DATAPOINT
        # wa_data_points = wa_data_points[(wa_data_points.Min >= int(mu))]
        # wa_data_points = wa_data_points[wa_data_points['Key'].str.contains(self.filter_emotion_by)]
        if self.verbose:
            print(f'Cantidad de partes antes de filtro: {len(chunked_data_points)}')
        # chunked_data_points = {key: chunked_data_points[key] for key in wa_data_points['Key']}
        if self.verbose:
            print(f'Cantidad de partes despues de filtro: {len(chunked_data_points)}\n')

        # del data
        # del wa_data_points

        if self.verbose:
            print(f'Segmentos de Datapoints completado.\n')

        return chunked_data_points

    def __len__(self):
        return len(self.input_features['inputs'])

    def __getitem__(self, idx):
        """
        lee del disco la parte segmentada de parte de audio
        :param idx:
        :return: tensor que representa el wave, tensor label
        """
        print(f"idx:{idx} {self.input_features['inputs'][idx]}")
        part = f"{AUDIO_SEGMENTS}/{self.input_features['inputs'][idx]}"
        waveform, _ = torchaudio.load(str(part), normalize=True)
        waveform = waveform.squeeze()
        label = self.input_features['labels'][idx]
        return waveform, torch.tensor(label, dtype=torch.int)  # x is also float32 by default, cause of normalize=True

    def __prepare_inputs(self, ):
        """
        une los metodos de la classe, hace una validacion en tamano, y transforma la data en un dict de inputs y labels
        :return: input_features
        """
        if len(self.df_reference) == 0:
            raise Exception(f'Parece que el split: {self.split} es incorrecto, intenta con Train, Test o Development')

        chunked_data_points = self.__prepare_datapoints()
        waves = self.__prepare_audio_inputs(chunked_data_points, dump_to_disk=True)
        #TODO: aqui unir VAD a un numero representativo
        labels_dp = {}
        for key in chunked_data_points.keys():
            a = []
            ii = min(len(chunked_data_points[key]['Valence']), len(chunked_data_points[key]['Arousal']),
                     len(chunked_data_points[key]['Dominance']))
            for i in range(ii):
                t = []
                jj = min(len(chunked_data_points[key]['Valence'][i]), len(chunked_data_points[key]['Arousal'][i]),
                    len(chunked_data_points[key]['Dominance'][i]))
                for j in range(jj):
                    t.append((chunked_data_points[key]['Valence'][i][j], chunked_data_points[key]['Arousal'][i][j], chunked_data_points[key]['Dominance'][i][j]))
                a.append(t)
            labels_dp[key] = a

        if self.verbose:
            ok = True
            for w in waves.keys():
                if len(waves[w]) != len(labels_dp[w]): #TODO: comparar con Arousal, Dominance
                    print(f'\n*** FAIL: Inputs y Labels waves:{len(waves[w])} datapoints:{len(chunked_data_points[w])}')
                    ok = False
            if ok:
                print('\nInputs y Labels se corresponden en catidad de segmentos.\n')

        # exploratoria
        import collections

        Output = collections.defaultdict(int)
        for vals in labels_dp.values():
            for l in vals:
                for tup in l:
                    Output[tup] += 1


        inputs = []
        labels = []
        for key in labels_dp.keys():
            for wv, dp in zip(waves[key], labels_dp[key]):
                inputs.append(wv)
                cat = [vad.vad2categorical(*vals, k=1, use_plot=False)[0][0]['index'] for vals in dp]
                labels.append(cat)
        input_features = {'inputs': inputs, 'labels': labels}
        if self.verbose:
            print(
                f"input_features listos, {len(input_features['inputs'])} inputs-tensores y {len(input_features['labels'])} labels-np.arrays\n")

        self.__save_input_features(input_features, f"input_features-{self.split}.pkl")
        return input_features


############### MODEL ##################
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


class RNNHead(nn.Module):
    def __init__(self, n_classes: int = 25): # n_class = 25 zise of occ-vocab + 1 (blank)
        super().__init__()
        self.feature_dim = 768 # base 768,  large 1024
        """
        In this paper, we explore methods for fine-tuning wav2vec
        2.0 on SER. We show that by adding a simple neural network 
        self.linear_head on top of wav2vec 2.0, vanilla fine-tuning (V-FT) 
        outperforms state-of-the-art (SOTA)
        """
        # self.lstm_head = nn.LSTM(self.feature_dim, 256, 1, bidirectional=True)  #TODO: never used !!!!
        self.gru_head = nn.GRU(input_size=self.feature_dim, hidden_size=self.feature_dim, num_layers=3, dropout=0.2, bidirectional=False)
        # self.linear_head = nn.Sequential(nn.ReLU(),  nn.Linear(768, 20) )
        # - -- - --
        # r_out, (h_n, h_c) = self.lstm_head(logits)
        # r_out.shape
        # self.linear_head(r_out)
        #TODO: taste with GRU ""modern version"  --->  self.rnn_head = nn.GRU(feature_dim, 256, 1, bidirectional=True)
        # self.rnn_head = nn.LSTM(feature_dim, 512, 1, bidirectional=True)
        self.linear_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, n_classes)
        )

    def redefine_linear_head(self, n_classes, device):
        self.linear_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, n_classes, device=torch.device(device))
        )

    def trainable_params(self):
        return list(self.gru_head.parameters()) + list(self.linear_head.parameters()) + list(self.wav2vec2.trainable_params())

    def forward(self, x, last_feat_pos):
        hidden_reps = x  ## [Wilton]
        logits = x.permute(1, 0, 2) #TODO: -> Length, Batch, Classes
        # print(f"logits.device {logits.device}")
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0), -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        # Whats special here? from [166, 1, 768] ==to==>  [1, 768]
        # mask: hay una relacion length para conocer y multiplicar los logits por zero cuando el audio no es el mas grande.
        # TODO: no aplicar, quede asi torch.Size([749, 1, 58]) por eso quitar siguiente linea.
        # logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        # xlogits  = ((logits * masks.T.unsqueeze(-1)).permute(1, 0, 2).sum(2) / last_feat_pos.unsqueeze(1))
        # xlogits = (fe.permute(2, 0, 1) * masks.T.unsqueeze(-1)).permute(1, 0, 2).mean(2)

        logits = self.linear_head(logits)
        return logits


class Wav2vec2Wrapper(nn.Module):
    def __init__(self, wav2vec2, chunk_size, pretrain=True,):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.rnn_head = RNNHead()
        #TODO: Disable gradient checkpointing for ddp
        self.wav2vec2.encoder.config.gradient_checkpointing = False
        self.pretrain = pretrain
        if pretrain:
            self.mask_time_length = chunk_size # [Wilton] was 15
            self.mask_time_prob = 0.06 #Probability of each time step is masked!
            self.observe_time_prob = 0.0 #Percentage of tokens that are perserved
            self.mask_feature_prob = 0.05
            self.mask_feature_length = 64
        else:
            #SpecAug
            self.mask_time_length = chunk_size # [Wilton] was 15
            self.mask_time_prob = 0.08
            self.observe_time_prob = 0.0
            self.mask_feature_prob = 0.05
            self.mask_feature_length = 64

    def prepare_mask(self, length, shape, dtype, device):
        # Modified from huggingface
        mask = torch.zeros(
            shape, dtype=dtype, device=device
        )
        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        mask[
            (torch.arange(mask.shape[0], device=device), length - 1)
        ] = 1
        mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return mask

    def trainable_params(self):
        ret = list(self.wav2vec2.encoder.parameters())
        return ret

    def __forward_wrapper(self, x, length=None):
        # [Wilton] it adapted from:
        # https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1510
        self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(chk_name, output_hidden_states=True).wav2vec2
        with torch.no_grad(): ## [Wilton] partial Fine-tuning
            x = self.wav2vec2.feature_extractor(x)
            fe = x
            x = x.transpose(1, 2) #New version of huggingface
            x, a = self.wav2vec2.feature_projection(x) #New version of huggingface
            mask = None
            if length is not None:
                length = self.get_feat_extract_output_lengths(length)
                mask = self.prepare_mask(length, x.shape[:2], x.dtype, x.device)
            if self.pretrain or self.training:
                batch_size, sequence_length, hidden_size = x.size()

                # [Wilton] from paper:
                # Wav2vec 2.0 differs from its NLP
                # counterparts [7] in that there is no utterance-level pretraining
                # task to naturally form a sentence representation. As a consequence, aggregation across time steps is required to fine-tune
                # on utterance level classification tasks.
                #
                # In addition, a modified version of SpecAugment [22] proposed in
                # wav2vec 2.0 is applied during training for better generalization
                #
                # apply SpecAugment along time axis VS. original: along feature axis [Wilton]
                if self.mask_time_prob > 0:
                    mask_time_indices = _compute_mask_indices(
                        (batch_size, sequence_length),
                        self.mask_time_prob,
                        self.mask_time_length,
                        min_masks=2,
                        # device=x.device #TODO: porque no??
                    )

                    mask_time_indices = torch.from_numpy(mask_time_indices).to(x.device) # [Wilton] fix to new torch and numpy versions.
                    masked_indicies = mask_time_indices & mask
                    flip_mask = torch.rand((batch_size, sequence_length), device=x.device) > self.observe_time_prob
                    x[masked_indicies & flip_mask] = self.wav2vec2.masked_spec_embed.to(x.dtype)

                # apply SpecAugment along feature axis
                if self.mask_feature_prob > 0:
                    mask_feature_indices = _compute_mask_indices(
                        (batch_size, hidden_size),
                        self.mask_feature_prob,
                        self.mask_feature_length,
                        # device=x.device, #TODO: porque no??
                        min_masks=1
                    )
                    mask_feature_indices = torch.from_numpy(mask_feature_indices).to(x.device)
                    x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        x = self.wav2vec2.encoder(x, attention_mask=mask)[0]
        reps = F.relu(x)
        # if self.pretrain:
        #     return reps, masked_indicies
        return reps, x

    #From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

    def forward(self, input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,):
        # print(f"input_values.device {input_values.device}")
        length = torch.tensor([len(i) for i in input_values],) # set device
        maxl = np.max([l.size(0) for l in labels])

        x, a = self.__forward_wrapper(input_values, length=length)
        last_feat_pos = (self.get_feat_extract_output_lengths(length) - 1).to(input_values.device)

        # self.rnn_head.redefine_linear_head(n_classes=maxl, device=input_values.device)
        # print(f"x.device {x.device}")
        # print(f"last_feat_pos.device {last_feat_pos.device}")
        logits = self.rnn_head(x, last_feat_pos)

        # return logits, labels
        return logits

###### play sound utility ####
def play(audio, sr=16_000):
    import sounddevice as sd
    sd.play(audio.numpy(), samplerate=sr)
    sd.wait()



################## DATACOLLATOR ####################
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
  processor: Wav2Vec2Processor

  def __call__(
      self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
  ) -> Dict[str, torch.Tensor]:
    input_ids = [{"input_values": feature[0]} for feature in features]
    labels = [{"input_ids": feature[1]} for feature in features] # TODO: pad labels approach
    # labels = [feature[1] for feature in features]
    batch = processor.pad(input_features=input_ids, labels=labels, return_tensors="pt",) # TODO: pad labels approach
    # TODO: este approach no me deja pasarlo a tensor ya que cada vector no son del mismo tamano ,
    # batch = processor.pad(input_features=input_ids, return_tensors="pt", return_attention_mask=True)
    # batch["labels"] = labels

    # batch = self.processor.pad(
    #     input_features,
    #     padding=self.padding,
    #     max_length=self.max_length,
    #     pad_to_multiple_of=self.pad_to_multiple_of,
    #     return_tensors="pt",
    # )
    # batch['input_values'][18] tensor([0.0248, 0.0270, 0.0582,  ..., 0.0006, 0.0014, 0.0017])
    # batch['labels'][18]       tensor([16.9640, 16.9640, 17.4840,  ...,  0.0000,  0.0000,  0.0000])
    return batch

############# CUSTOM TRAINER ################
from transformers import Seq2SeqTrainer
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, mean_squared_error
from transformers import EvalPrediction


class CustomTrainer(Seq2SeqTrainer):
    shuffle_batch: bool
    shuffle_items: bool
    cross_entropy_loss: nn.CrossEntropyLoss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = self.cross_entropy_loss(outputs , inputs['labels'])
            # if return_outputs: # TODO: esto es para evaluacion
            #     outputs = outputs.squeeze()

        return (loss, outputs) if return_outputs else loss


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        self.args
        model.train()
        inputs = self._prepare_inputs(inputs)

        # ['input_values', 'labels']
        # print(f"input_values.device {inputs['input_values'].device}")
        # print(f"labels.device {inputs['labels'].device}")
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        # loss = loss.mean()
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.shuffle_batch:
                dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, shuffle=self.shuffle_items, **dataloader_params))


def compute_metrics(p: EvalPrediction):
    """This use dev_dataset"""
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return {"mse": mean_squared_error(y_pred=preds, y_true=p.label_ids)}


############## TRAINING ARGS ############
from transformers import Seq2SeqTrainingArguments


def get_device():
    device = "cuda" \
        if torch.cuda.is_available() \
        else "mps" if torch.backends.mps.is_available() \
        else "cpu"
    return torch.device(device)


device = get_device()
device = torch.device("cpu")
chunk_size=15
batch_size=1
epochs=1
use_cpu=True

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    # gradient_accumulation_steps=8,
    learning_rate=1e-5,
    # warmup_steps=500,
    # max_steps=4000,
    # gradient_checkpointing=True,
    # fp16=True,
    evaluation_strategy="epoch", # epoch
    save_strategy="epoch",
    per_device_eval_batch_size=1, #4
    save_steps=1000, # 1000
    eval_steps=1000, #1000
    logging_steps=1, # 10
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    # greater_is_better=False,
    label_names=["labels"],
    push_to_hub=False,
    use_cpu=use_cpu,
)


chk_name = "facebook/wav2vec2-base-960h"
wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(chk_name, output_hidden_states=True).wav2vec2.to(device=device)
processor = Wav2Vec2Processor.from_pretrained(chk_name)
model = Wav2vec2Wrapper(wav2vec2=wav2vec2, chunk_size=chunk_size).to(device=device)
data_collator = DataCollatorCTCWithPadding(processor=processor)

#
train_dataset = MSPDataset(msp_path=MSP_PATH, wadf=wadf, df_reference=df_reduced, split="Train",  chunk_size=chunk_size, verbose=True,)
dev_dataset = MSPDataset(msp_path=MSP_PATH, wadf=wadf, df_reference=df_reduced, split="Development",  chunk_size=chunk_size, verbose=True, )

print('bye!')
exit(-1)

trainer = CustomTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    tokenizer=processor,
    # compute_metrics=compute_metrics
)
setattr(trainer, 'shuffle_batch', False)
setattr(trainer, 'shuffle_items', False)
setattr(trainer, 'cross_entropy_loss', nn.BCEWithLogitsLoss())

trainer.train()
trainer.save_model()
processor.save_pretrained(training_args.output_dir)


################ TEST ##################
from safetensors.torch import load_model, save_model, safe_open
from sklearn.metrics import classification_report, mean_squared_error

test_dataset = MSPDataset(msp_path=MSP_PATH, wadf=wadf, df_reference=df_reduced, split="Test", chunk_size=chunk_size, verbose=True,)

chk_name = "facebook/wav2vec2-base-960h"
wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(chk_name, output_hidden_states=True).wav2vec2.to(device=device)
processor = Wav2Vec2Processor.from_pretrained(chk_name)
model = Wav2vec2Wrapper(wav2vec2=wav2vec2, chunk_size=chunk_size).to(device=device)
data_collator = DataCollatorCTCWithPadding(processor=processor)
my_checkpoint = '/Users/beltre.wilton/apps/mspconv_ftlab/checkpoint/checkpoint-204/model.safetensors'

obj = safe_open(my_checkpoint, framework="pt", device="cpu")
n_classes = obj.get_slice('rnn_head.linear_head.1.weight').get_shape()[0]
model.linear_layer.redefine_linear_head(n_classes=n_classes, device=device)
del obj

load_model(model, my_checkpoint)


#TODO iterate it para get los wavesform correctos.
#TODO: falta data_collator.
#TODO: mer metricas con function de metricas mse
test_dataset = MSPDataset(msp_path=MSP_PATH, wadf=wadf, df_reference=df_reduced, split="Test", chunk_size=chunk_size, verbose=True,)
inputs = test_dataset.input_features['inputs']

with torch.no_grad():
    model.eval()
    acc_mse = []
    for (input, label), n in zip(test_dataset, inputs):
        label = label.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
        logits = model(input_values=input, labels=label)
        mse = mean_squared_error(logits.cpu().numpy(), label.cpu().numpy())
        acc_mse.append(mse)
    print(f"Accuracy MSE: {np.mean(acc_mse)}" )




#
# print()



