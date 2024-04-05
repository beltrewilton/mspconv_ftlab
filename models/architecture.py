import inspect
from typing import Any
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.regression import MeanSquaredError
from torchmetrics.text import CharErrorRate
from torchmetrics import Accuracy
from transformers import Wav2Vec2ForPreTraining
from models.data_processor import ROOT

from vad.vad_lab import VAD

vad = VAD(minmax=[-100, 100], mapping="OCC")

# TOKENS = f"{ROOT}/models/tokens.txt"
TOKENS = f"{ROOT}/models/timit_vocab.txt"
LEXICON = f"{ROOT}/models/timit_lexicon.txt"

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


# utileria para medir tiempo.
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
    def __init__(self, feature_dim: int = 1024,  n_classes: int = 41, final_dropout: float = 0.1, chunk_size_abs: float = 6): # n_class = 26 zise of occ-vocab + 1 (blank "-" token) + 1 (silence "|" token)
        super().__init__()
        self.w2v_tw = 0.02002 # aproximacion del time frame window de wav2vec2
        batchnorm_in_dim = int(chunk_size_abs // self.w2v_tw)
        self.linear_head = nn.Sequential(
            # nn.ReLU(), # TODO: alguna diferencia?
            nn.Dropout(final_dropout), # copiado de Wav2Vec2ForCTC
            # nn.ReLU(), #TODO en impl. no hay ReLU, lo dejo y veo.
            nn.Linear(feature_dim, n_classes),
            # nn.BatchNorm1d(batchnorm_in_dim),
        )

    def trainable_params(self):
        return list(self.linear_head.parameters())

    def forward(self, x): #TODO --> last_feat_pos
        logits = self.linear_head(x)
        return logits


class LinearLayerForClassification(nn.Module):
    def __init__(self, feature_dim: int = 1024,  n_classes: int = 24):
        super().__init__()
        self.linear_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_dim, n_classes),
            nn.Softmax(),
            # nn.BatchNorm1d(batchnorm_in_dim),
        )

    def trainable_params(self):
        return list(self.linear_head.parameters())

    def forward(self, x):
        logits = self.linear_head(x)
        return logits


class Wav2vec2ModelWrapper(nn.Module):
    def __init__(self, checkpoint_name: str, chunk_size: int, overlap: float, final_dropout: float, train_mode: bool = False):
        super().__init__()
        timem.start("Wav2Vec2ForPreTraining.from_pretrained")
        self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(checkpoint_name, output_hidden_states=True).wav2vec2
        timem.end("Wav2Vec2ForPreTraining.from_pretrained")
        self.wav2vec2.encoder.config.gradient_checkpointing = False
        self.linear_layer = LinearLayer(final_dropout=final_dropout, chunk_size_abs=(chunk_size + (2*overlap)))
        self.train_mode = train_mode
        self.wav2vec2.training = train_mode
        self.wav2vec2.init_weights()

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
        # return self.linear_layer.trainable_params()

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
        length = torch.tensor([len(i) for i in input_values], )  # set device
        maxl = np.max([l.size(0) for l in labels])

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

        last_feat_pos = (self.get_feat_extract_output_lengths(length) - 1).to(input_values.device)

        timem.start("self.linear_layer")
        logits = self.linear_layer(hidden_states)
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        timem.end("self.linear_layer")

        return log_probs, logits


class Wav2vec2ModelWrapperForClassification(nn.Module):
    def __init__(self, checkpoint_name: str, n_classes: int = 24, train_mode: bool = False):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(checkpoint_name, output_hidden_states=True).wav2vec2
        self.wav2vec2.encoder.config.gradient_checkpointing = False
        self.n_classes = n_classes
        self.linear_layer = LinearLayerForClassification(n_classes=n_classes)
        self.train_mode = train_mode
        self.wav2vec2.training = train_mode
        self.wav2vec2.init_weights()

        if train_mode: #SpecAug compara con self.wav2vec2.config.*  setear este mask_feature_prob aparte
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
        # return self.linear_layer.trainable_params()

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
        length = torch.tensor([len(i) for i in input_values], )  # set device

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
        logits = F.relu(hidden_states)
        logits = logits.permute(1, 0, 2)  # L, B, C

        last_feat_pos = (self.get_feat_extract_output_lengths(length) - 1).to(input_values.device)
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0),-1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        # xlogits = ((logits * masks.T.unsqueeze(-1)).permute(1, 0, 2).sum(2) / last_feat_pos.unsqueeze(1))

        logits = self.linear_layer(logits)

        return logits, hidden_states


class MSPImplementationForClassification(L.LightningModule):
    def __init__(self, model: Wav2vec2ModelWrapperForClassification, lr: float, train_mode: bool = False):
        super().__init__()
        self.lr = lr
        self.model = model
        self.save_hyperparameters(ignore=["model"]) # skip model parameters to save to log :|
        self.train_mode = train_mode
        self.entropy_loss = torch.nn.CrossEntropyLoss() #TODO : <--- init weigths
        self.train_acc = Accuracy(task="multiclass", num_classes=self.model.n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.model.n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.model.n_classes)

    def forward(self, x, labels):
        return self.model(x, labels=labels)

    def _iter_step(self, batch):
        inputs, true_labels = batch
        logits, hidden_states = self(inputs, true_labels)
        loss = self.entropy_loss(logits, true_labels)
        return loss, true_labels, logits

    def training_step(self, batch, batch_idx):
        loss, true_labels, logits = self._iter_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        #Accuracy
        if batch_idx % 10 == 0:
            y_hat = torch.argmax(logits, axis=1)
            pred = [vad.terms[y] for y in y_hat] # para mirar
            print("y_hat      :", y_hat)
            print("true_labels:", true_labels)
            print("\n")
            acc = self.train_acc(y_hat, true_labels)
            self.log(
                "train_acc", acc.item(), on_step=True, on_epoch=True, prog_bar=True
            )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, logits = self._iter_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        #Accuracy
        if batch_idx % 20 == 0:
            y_hat = torch.argmax(logits, axis=1)
            pred = [vad.terms[y] for y in y_hat]  # para mirar
            acc = self.val_acc(y_hat, true_labels)
            self.log(
                "val_acc", acc.item(), on_step=True, on_epoch=True, prog_bar=True
            )

    def test_step(self, batch, batch_idx):
        loss, true_labels, logits = self._iter_step(batch)
        self.log("test_loss", loss)
        #Accuracy
        y_hat = torch.argmax(logits, axis=1)
        pred = [vad.terms[y] for y in y_hat]  # para mirar
        acc = self.test_acc(y_hat, true_labels)
        self.log("test_acc", acc.item())

    def configure_optimizers(self):
        # timem.start("torch.optim.AdamW")
        # optimizer = torch.optim.AdamW(self.model.trainable_params(), lr=self.lr) #TODO: ver otros optimizadores.
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="min", verbose=True)
        # timem.end("torch.optim.AdamW")
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "train_loss",
        #         "interval": "epoch",
        #         "frequency": 1,
        #     }
        # }

        optimizer = torch.optim.AdamW(self.model.trainable_params(), lr=self.lr)
        return optimizer


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

    def _beam_search(self, emmisions, true_labels, space_label=25):
        labels = []
        for tl in true_labels:
            labels.append([i.item() for i in tl if i != space_label])
        # true_labels_chr = [" ".join([chr(ord('`') + t) for t in bloc if t != 0]) for bloc in labels]
        vocab = {'-': 0, 'aa': 1, 'ae': 2, 'ah': 3, 'aw': 4, 'ay': 5, 'b': 6, 'ch': 7, 'd': 8, 'dh': 9, 'dx': 10,
                 'eh': 11,
                 'er': 12, 'ey': 13, 'f': 14, 'g': 15, 'h#': 16, 'hh': 17, 'ih': 18, 'iy': 19, 'jh': 20, 'k': 21,
                 'l': 22,
                 'm': 23, 'n': 24, 'ng': 25, 'ow': 26, 'oy': 27, 'p': 28, 'r': 29, 's': 30, 'sh': 31, 't': 32, 'th': 33,
                 'uh': 34, 'uw': 35, 'v': 36, 'w': 37, 'y': 38, 'z': 39, '?': 40, '|': 41}
        get_medium = lambda x: list(vocab.keys())[list(vocab.values()).index(x)]
        true_labels_chr = [" ".join([get_medium(t) for t in bloc if t != 0 and t != 41]) for bloc in labels]
        emmisions = emmisions.to(torch.device("cpu")) if (emmisions.device.type == "cuda" or emmisions.device.type == "mps") else emmisions
        beam_search_result = beam_search_decoder(emmisions.contiguous())
        predicted_labels_chr = [" ".join(b[0].words) for b in beam_search_result]
        predicted_tokens = [r[0].tokens for r in beam_search_result]

        return true_labels_chr, predicted_labels_chr, predicted_tokens

    def _iter_step(self, batch):
        inputs, true_labels = batch

        log_probs, logits = self(inputs, true_labels)

        input_len, batch_size, n_class = log_probs.shape
        input_lengths = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.long, device=inputs.device)
        max_ = np.max([l.size(0) for l in true_labels])

        #TODO: cual usar?
        target_lengths = torch.tensor([max_], dtype=torch.long, device=inputs.device).repeat(batch_size)
        target_lengths = torch.count_nonzero(true_labels, axis=1) # length of the labels # ctc_loss doesn't support fp16

        emmisions = log_probs.to(torch.device("cpu")) if (log_probs.device.type == "cuda" or log_probs.device.type == "mps") else log_probs
        if log_probs.device.type == "mps":
            loss = self.ctc_loss(emmisions, true_labels.cpu(), input_lengths.cpu(), target_lengths.cpu())
        else:
            loss = self.ctc_loss(emmisions, true_labels, input_lengths, target_lengths)

        return loss, log_probs, true_labels, logits


    def training_step(self, batch, batch_idx):
        loss, log_probs, true_labels, logits = self._iter_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        #Accuracy
        if batch_idx % 100 == 0:
            true_labels_chr, predicted_labels_chr, predicted_tokens = self._beam_search(logits, true_labels)
            print("-------------------------------true_labels_chr----------------------------------")
            print(true_labels_chr)
            print("------------------------------predicted_labels_chr-----------------------------------")
            print(predicted_labels_chr)
            print("-------------------------------predicted_tokens----------------------------------")
            print(predicted_tokens)
            print("-----------------------------------------------------------------")

            err_rate = self.train_err_rate(predicted_labels_chr, true_labels_chr)
            self.log(
                "train_err_rate", err_rate, on_step=True, on_epoch=True, prog_bar=True
            )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_probs, true_labels, logits = self._iter_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        #Accuracy
        if batch_idx % 20 == 0:
            true_labels_chr, predicted_labels_chr, predicted_tokens = self._beam_search(logits, true_labels)
            print("-------------------------------true_labels_chr----------------------------------")
            print(true_labels_chr)
            print("------------------------------predicted_labels_chr-----------------------------------")
            print(predicted_labels_chr)
            print("-------------------------------predicted_tokens----------------------------------")
            print(predicted_tokens)
            print("-----------------------------------------------------------------")
            err_rate = self.val_err_rate(predicted_labels_chr, true_labels_chr)
            self.log(
                "val_err_rate", err_rate, on_step=True, on_epoch=True, prog_bar=True
            )

    def test_step(self, batch, batch_idx):
        loss, log_probs, true_labels, logits = self._iter_step(batch)
        self.log("test_loss", loss)
        #Accuracy
        true_labels_chr, predicted_labels_chr, predicted_tokens = self._beam_search(logits, true_labels)
        err_rate = self.test_err_rate(predicted_labels_chr, true_labels_chr)
        self.log("test_err_rate", err_rate)

    def configure_optimizers(self):
        timem.start("torch.optim.AdamW")
        optimizer = torch.optim.AdamW(self.model.trainable_params(), lr=self.lr) #TODO: ver otros optimizadores.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="min", verbose=True)
        timem.end("torch.optim.AdamW")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }

# TODO; otra consideraciones:
# Wav2Vec2ForCTC.freeze_base_model:  disable the gradient computation for the base model
# self.config.vocab_size setear en CTC
# el forwadr llama al modelo base y luego hace mas cosas.
# TODO: EN CTCLOSS with torch.backends.cudnn.flags(enabled=False): para cuuidar uso de MPS ????
# TODO: comida para CTCLoss:
# log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
# dim=-1 dim=2 para mi inputs es lo mismo