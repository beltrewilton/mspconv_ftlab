import sys
import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
sys.path.append("../")
from models.data_processor import MSPDataProcessor, MSP_PATH, ROOT, AUDIO_SEGMENTS
from models.dataset_utils import MSPDataset
from models.architecture import Wav2vec2ModelWrapper, Wav2vec2ModelWrapperForClassification, MSPImplementation, MSPImplementationForClassification, Timem

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

os.environ['HF_HOME'] = f'{ROOT}/cache'
os.environ['HF_DATASETS_CACHE'] = f'{ROOT}/cache'
LOG_DIR = f"{ROOT}/logs"

batch_size = 8
chunk_size = 5
overlap = 0.5
num_workers = 7
# checkpoint_name = "facebook/wav2vec2-base-960h"
checkpoint_name = "facebook/wav2vec2-xls-r-300m"
train_mode = True
lr = 1e-5
epochs = 30 #10


def get_loaders(batch_size: int, chunk_size: int, overlap: float, num_workers: int = 0 ):
    datapros_train = MSPDataProcessor(msp_path=MSP_PATH, chunk_size=chunk_size, overlap=overlap, split="Train", verbose=True)
    datapros_test = MSPDataProcessor(msp_path=MSP_PATH, chunk_size=chunk_size, overlap=overlap, split="Test", verbose=True)
    datapros_dev = MSPDataProcessor(msp_path=MSP_PATH, chunk_size=chunk_size, overlap=overlap, split="Development", verbose=True)

    dataset_train = MSPDataset(input_features=datapros_train.load_input_features())
    dataset_test = MSPDataset(input_features=datapros_test.load_input_features(),)
    dataset_dev = MSPDataset(input_features=datapros_dev.load_input_features(),)

    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
        collate_fn=dataset_train.seqCollate,
    )

    loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_test.seqCollate,
    )

    loader_dev = DataLoader(
        dataset=dataset_dev,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_dev.seqCollate,
    )

    return loader_train, loader_dev, loader_test


class MeasureCallback(L.Callback):
    def __init__(self):
        self.time_batch = Timem()
        self.time_back = Timem()
        self.time_epoch = Timem()
        self.time_train = Timem()

    def on_train_start(self, trainer, pl_module):
        self.time_train.start("***** on_train_start *****")

    def on_train_end(self, trainer, pl_module):
        self.time_train.end("***** on_train_end *****")

    def on_train_batch_start(
        self, trainer, pl_module,  batch, batch_idx
    ):
        self.time_batch.start("")
        # torch.cuda.empty_cache()
        # import gc
        # gc.collect()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        execution_time = self.time_batch.end(f"on_train_batch_end batch_idx:{batch_idx}")
        pl_module.log("train_mins_per_batch", execution_time / 60)

    def on_before_backward(self, trainer, pl_module, loss):
        self.time_back.start("")

    def on_after_backward(self, trainer, pl_module):
        self.time_back.end("on_after_backward")

    def on_train_epoch_start(self, trainer, pl_module):
        self.time_epoch.start("")

    def on_train_epoch_end(self, trainer, pl_module):
        execution_time = self.time_epoch.end("on_train_epoch_end")
        pl_module.log("train_mins_per_epoch", execution_time / 60)

    # def on_before_optimizer_step(
    #     self, trainer, pl_module, optimizer: torch.optim.AdamW
    # ) :
    #     # Now let's inspect the optimizer's state
    #     for group in optimizer.param_groups:
    #         for param in group['params']:
    #             print(f"param.shape:{param.shape} param.grad:{param.grad}")


if __name__ == "__main__":
    loader_train, loader_dev, loader_test = get_loaders(batch_size, chunk_size, overlap, num_workers)

    is_classification = True
    if is_classification:
        model = Wav2vec2ModelWrapperForClassification(checkpoint_name=checkpoint_name, train_mode=train_mode)
        msp_impl_model = MSPImplementationForClassification(model=model, lr=lr, train_mode=train_mode)
    else:
        pass

    checkpoint_callback = ModelCheckpoint(save_top_k=1, mode="min", monitor="train_loss", save_last=True)
    time_measure_callback = MeasureCallback()

    trainer = L.Trainer(
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        accelerator="auto",
        devices="auto",
        default_root_dir=LOG_DIR,
        callbacks=[checkpoint_callback, time_measure_callback],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model=msp_impl_model,
        train_dataloaders=loader_train,
        val_dataloaders=loader_test,
    )

    train_acc = trainer.test(model=msp_impl_model, dataloaders=loader_train, ckpt_path="best")[0]["test_acc"]
    val_acc = trainer.test(model=msp_impl_model, dataloaders=loader_dev, ckpt_path="best")[0]["test_acc"]
    test_acc = trainer.test(model=msp_impl_model, dataloaders=loader_test, ckpt_path="best")[0]["test_acc"]

    print(
        f"Train Acc {train_acc*100:.2f}%"
        f"Val Acc {val_acc*100:.2f}%"
        f"Test Acc {test_acc*100:.2f}%"
    )


# Time in MPS
# batch_duration
#  C = lambda samples, epochs: (samples/20 * batch_duration) * epochs 
# C(100,1) # 100 samples in 1 epochs = n minutes

