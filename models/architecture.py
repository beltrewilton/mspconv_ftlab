import inspect
from typing import Any
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.models.decoder import ctc_decoder
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torchmetrics
from torchmetrics.regression import MeanSquaredError
from torchmetrics.text import CharErrorRate
from transformers import Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from models.data_processor import ROOT

TOKENS = f"{ROOT}/models/tokens.txt"
LEXICON = f"{ROOT}/models/lexicon.txt"

LM_WEIGHT = 3.23
WORD_SCORE = -0.26

beam_search_decoder = ctc_decoder(
    lexicon=LEXICON,
    tokens=TOKENS,
    lm=None,
    nbest=1,
    beam_size=1500, # heavy
    # beam_size=100,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
)

# utileria para medor tiempo.
class Timem:
    def __init__(self, verbose: bool = True, threshold: int = 10):
        self.start_time = time.time()
        self.threshold = threshold
        self.verbose = verbose

    def start(self, msg: str):
        self.start_time = time.time()

    def end(self, msg: str):
        end_time = time.time()
        execution_time = end_time - self.start_time
        if self.verbose and execution_time > self.threshold:
            print(f"Time ended :{msg}, execution time:", execution_time/60, "minutes")
        return execution_time

timem = Timem(verbose=True)

class LinearLayer(nn.Module):
    def __init__(self, feature_dim: int = 768,  n_classes: int = 26, final_dropout: float = 0.01): # n_class = 26 zise of occ-vocab + 1 (blank "-" token) + 1 (silence "|" token)
        super().__init__()
        self.linear_head = nn.Sequential(
            nn.ReLU(), # TODO: alguna diferencia?
            nn.Dropout(final_dropout), # copiado de Wav2Vec2ForCTC
            nn.ReLU(), #TODO en impl. no hay ReLU, lo dejo y veo.
            nn.Linear(feature_dim, n_classes),
        )

    def trainable_params(self): #TODO: ojo con esto
        return list(self.linear_head.parameters())

    def forward(self, x, last_feat_pos): #TODO --> last_feat_pos
        logits = x.permute(1, 0, 2) #TODO: -> Length, Batch, Classes | CTC equivalente: T = input length, N  = batch_size, C = Number of classes with 0 (blank)
        logits = self.linear_head(logits)
        return logits


class Wav2vec2ModelWrapper(nn.Module):
    def __init__(self, checkpoint_name: str, chunk_size: int, final_dropout: float, train_mode: bool = False):
        super().__init__()
        timem.start("Wav2Vec2ForPreTraining.from_pretrained")
        self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(checkpoint_name, output_hidden_states=True).wav2vec2
        timem.end("Wav2Vec2ForPreTraining.from_pretrained")
        self.wav2vec2.encoder.config.gradient_checkpointing = False
        self.linear_layer = LinearLayer(final_dropout=final_dropout)
        self.train_mode = train_mode
        self.wav2vec2.training = train_mode

        if train_mode: #SpecAug compara con self.wav2vec2.config.*  setear este mask_feature_prob aparte
            self.mask_time_length = chunk_size # [Wilton] was 15
            self.wav2vec2.config.mask_feature_prob = 0.05

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

    def trainable_params(self): #TODO: ojo con esto
        return list(self.linear_layer.trainable_params()) + list(self.wav2vec2.encoder.parameters())

    def __forward_wrapper(self, input_values, length=None):
        with torch.no_grad(): ## [Wilton] for partial Fine-tuning
            self.wav2vec2.training = self.train_mode

            output_attentions = False
            output_hidden_states = True
            mask_time_indices = None
            attention_mask = None

            extract_features = self.wav2vec2.feature_extractor(input_values)
            extract_features = extract_features.transpose(1, 2)

            hidden_states, extract_features = self.wav2vec2.feature_projection(extract_features)
            hidden_states = self.wav2vec2._mask_hidden_states(
                hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
            )

        encoder_outputs = self.wav2vec2.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = encoder_outputs[0]

        return hidden_states

    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            # return (input_length - kernel_size) // stride + 1
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1 #TODO: si da problemas, regresaar v. anterior

        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

    def forward(self, input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                labels=None, ):
        # print(f"input_values.device {input_values.device}")
        length = torch.tensor([len(i) for i in input_values], )  # set device
        maxl = np.max([l.size(0) for l in labels])

        hidden_states = self.__forward_wrapper(input_values, length=length)

        last_feat_pos = (self.get_feat_extract_output_lengths(length) - 1).to(input_values.device)

        # print(f"x.device {x.device}")
        # print(f"last_feat_pos.device {last_feat_pos.device}")
        timem.start("self.linear_layer")
        logits = self.linear_layer(hidden_states, last_feat_pos)
        logits = F.log_softmax(logits, dim=2)  # TODO: necesario para inferencia?, dim=2?????
        timem.end("self.linear_layer")

        return logits


class MSPImplementation(L.LightningModule):
    def __init__(self, model: Wav2vec2ModelWrapper, lr: float, train_mode: bool = False):
        super().__init__()
        self.lr = lr
        self.model = model
        self.save_hyperparameters(ignore=["model"]) # skip model parameters to save to log :|
        self.train_mode = train_mode
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self.train_err_rate = CharErrorRate()
        self.val_err_rate = CharErrorRate()
        self.test_err_rate = CharErrorRate()

    def forward(self, x, labels):
        return self.model(x, labels=labels)

    def _greedy_decoder(self, logits, true_labels, blank_label=0,):
        indices = torch.argmax(logits.transpose(0, 1), dim=-1)
        hypothesis = []
        for idx in indices:
            unique = torch.unique_consecutive(idx, dim=-1)
            hypothesis.append([i.item() for i in unique if i != blank_label])

        labels = []
        for tl in true_labels:
            labels.append([i.item() for i in tl if i != blank_label])

        return labels, hypothesis

    def _beam_search(self, logits, true_labels, blank_label=0):
        labels = []
        for tl in true_labels:
            labels.append([i.item() for i in tl if i != blank_label])
        true_labels_chr = [" ".join([chr(ord('`') + t) for t in bloc]) for bloc in labels]
        emmisions = logits.to(torch.device("cpu")) if logits.device.type == "mps" else logits
        beam_search_result = beam_search_decoder(emmisions.transpose(0, 1).contiguous()) #TODO: 14 seg dura esto :S
        # beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
        predicted_labels_chr = [" ".join(b[0].words) for b in beam_search_result]
        predicted_tokens = [r[0].tokens for r in beam_search_result]
        print("predicted_tokens:", predicted_tokens)
        # beam_search_wer = torchaudio.functional.edit_distance(true_labels_chr, predicted_labels_chr) / len(true_labels_chr)

        # print(f"Transcript: {beam_search_transcript}")
        # print(f"WER: {beam_search_wer}")

        return true_labels_chr, predicted_labels_chr, predicted_tokens

    def _iter_step(self, batch):
        inputs, true_labels = batch

        logits = self(inputs, true_labels)

        input_len, batch_size, n_class = logits.shape
        input_lengths = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.long, device=inputs.device)
        max_ = np.max([l.size(0) for l in true_labels])
        target_lengths = torch.tensor([max_], dtype=torch.long, device=inputs.device).repeat(batch_size)

        emmisions = logits.to(torch.device("cpu")) if inputs.device.type == "mps" else inputs
        # self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction='none') ver que pasa.
        # input_lengths.cpu() >= target_lengths.cpu()
        # TODO: torch.count_nonzero for target_lengths ademas quita el padding del label ....
        target_lengths = torch.count_nonzero(true_labels, axis=1) # length of the labels # ctc_loss doesn't support fp16
        loss = self.ctc_loss(emmisions, true_labels.cpu(), input_lengths.cpu(), target_lengths.cpu()) #TODO: warning .cpu() en Ligthing Studio
        print()

        # w_size = target_lengths[0].item()
        # m = input_len // w_size
        # divisible_length = m * w_size
        # divisible_logits = logits[:divisible_length, :, :]
        # divisible_logits = divisible_logits.view(w_size, batch_size, n_class, m).mean(dim=-1)
        # residual = logits[divisible_length:, :, :]
        # residual = residual.mean(dim=0) # residual no parece necesario ya que dividi perfectamente 60 pasos.
        # predicted_labels = torch.argmax(divisible_logits, dim=-1).transpose(0, 1) + 1 # to avoid 0's and match with labels

        # decoded_preds, decoded_targets = self.GreedyDecoder(logits.transpose(0, 1), true_labels, target_lengths,)

        # :ctc_decoder
        # CTC Decoder suit requires flashlight-text package and optionally KenLM. Please install them.

        # simple prototipo basado en GreedyCTCDecoder (pytorch)
        # labels, hypothesis = self._greedy_decoder(logits, true_labels)
        timem.start("self._beam_search")
        true_labels_chr, predicted_labels_chr, predicted_tokens = self._beam_search(logits, true_labels)
        timem.end("self._beam_search")


        # torch.unique_consecutive(indices[0, :], dim=-1).shape
        # torch.unique_consecutive(indices[1, :], dim=-1).shape TODO fail: al objeto completo 'indices'
        # quita los zeros y luego
        # f = torch.unique_consecutive(indices[1, :], dim=-1)
        # t = [i.item() for i in true_labels[1, :] if i != 0]
        # t = torch.tensor(t, device=inputs.device)
        # h = [i.item() for i in f if i != 0][:t.size(1)]
        # h = torch.tensor(h, device=inputs.device)
        # from torchmetrics import Accuracy
        # acc = Accuracy(task="MULTICLASS", num_classes=25)
        # acc(t, h)

        # TODO esta promete :)
        # from torchmetrics.text import CharErrorRate
        # cer = CharErrorRate()
        # f = torch.unique_consecutive(indices[1, :], dim=-1)  #TODO hacerlo iterendo el batch 0, 1, ... 19 si bath=20
        # t = " ".join([str(i.item()) for i in true_labels[1, :] if i != 0])  #TODO hacerlo iterendo el batch 0, 1, ... 19 si bath=20
        # h = " ".join([str(i.item()) for i in f if i != 0])
        # cer(t, h)
        #
        ## TODO : hcer inferencia L.load_from_check_point(check_point_path="path", model=pure_pytorch_model)

        #from torchmetrics import Accuracy
        # pred = indices[:, :true_labels.size(1)] # sera que en algun momento al sacar las consecutivas baja a 60????
        # [self.test_acc(p, t) for p, t in zip(true_labels, pred)]
        #
        # acc = Accuracy(task="MULTICLASS", num_classes=25)
        # [acc(p, t) for p, t in zip(true_labels, pred.transpose(0, 1))]
        # TODO: hacer inferencia con el checkpoint de los 10 epocs y ver resultado [dataset visto: de un sample de train]
        # probar con el prototipo greddy
        # probar con ctc_loss y ver resultados (requiere descargas...)
        # wav2vec2 extrae cada 0.020027 parte de 1 segundo
        # TODO: CREO que hay que sacar los labels de nuevo intercalando indices con blank-token para isolarlo que se colapsen repetidos


        # algoritmo#2
        # t = []
        # for k in range(0, divisible_length, m):
        #     t.append(logits[k:k+m, :, :].mean(dim=0))
        # last = logits[k + m:, :, :].mean(dim=0)
        # if len(last) > 0:
        #     t.append(last)
        # logits_mean = torch.stack(t)
        # predicted_labels = torch.argmax(logits_mean, dim=-1).transpose(0, 1) + 1

        return loss, true_labels_chr, predicted_labels_chr, predicted_tokens

    def _accuracy(self, acc_func, hypothesis, true_labels):
        # tl = [" ".join([chr(ord('`')+t) for t in bloc]) for bloc in true_labels]
        # hyp = [" ".join([chr(ord('`') + h) for h in bloc]) for bloc in hypothesis]
        tl = [" ".join(t) for t in true_labels]
        hyp = [" ".join(t) for t in hypothesis]
        return acc_func(tl, hyp)

    def training_step(self, batch, batch_idx):
        loss, true_labels, hypothesis, predicted_tokens = self._iter_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        #Accuracy
        timem.start("self.train_err_rate")
        err_rate = self.train_err_rate(hypothesis, true_labels)
        timem.end("self.train_err_rate")
        self.log(
            "train_err_rate", err_rate, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, hypothesis, predicted_tokens = self._iter_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        # Accuracy
        timem.start("self.val_err_rate")
        err_rate = self.val_err_rate(hypothesis, true_labels)
        timem.end("self.val_err_rate")
        self.log(
            "val_err_rate", err_rate, on_step=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, hypothesis, predicted_tokens = self._iter_step(batch)
        timem.start("self.test_err_rate")
        err_rate = self.test_err_rate(hypothesis, true_labels)
        timem.end("self.test_err_rate")
        self.log("test_err_rate", err_rate)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        timem.start("torch.optim.Adamax")
        optimizer = torch.optim.AdamW(self.model.trainable_params(), lr=self.lr) #TODO: ver otros optimizadores.
        timem.end("torch.optim.Adamax")
        return optimizer






# TODO; otra consideraciones:
# Wav2Vec2ForCTC.freeze_base_model:  disable the gradient computation for the base model
# self.config.vocab_size setear en CTC
# el forwadr llama al modelo base y luego hace mas cosas.
# TODO: EN CTCLOSS with torch.backends.cudnn.flags(enabled=False): para cuuidar uso de MPS ????
# TODO: comida para CTCLoss:
# log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
# dim=-1 dim=2 para mi inputs es lo mismo







