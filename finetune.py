import pytorch_lightning as pl
import random

import argparse
import os
import numpy as np
import pandas as pd
import re

import torch, gc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EncoderDecoderModel

from torch.optim import AdamW

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)

def remove_all_punkt(text):
    return re.sub(r'[^\w\s]', '', text)

class Seq2Seq(Dataset):
    def __init__(self, df, tokenizer, target_tokenizer, max_len, no_punkt:bool = False):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_len = max_len
        self.no_punkt = no_punkt

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        return dict(self.df.iloc[idx])

    def collate(self, batch):
        batch_df = pd.DataFrame(list(batch))
        x, y = batch_df.source, batch_df.target
        if self.no_punkt:
            x = list(i if random.random()>0.5 else remove_all_punkt(i) for i in x)
        else:
            x = list(x)
        x_batch = self.tokenizer(
            x,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        y_batch = self.target_tokenizer(
            list(y),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        x_batch['decoder_input_ids'] = y_batch['input_ids']
        x_batch['labels'] = y_batch['input_ids'].clone()
        x_batch['labels'][x_batch['labels'] == self.tokenizer.pad_token_id] = -100
        return x_batch

    def dataloader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate)

    def split_train_valid(self, valid_size=0.1):
        split_index = int(len(self) * (1 - valid_size))
        cls = type(self)
        shuffled = self.df.sample(frac=1).reset_index(drop=True)
        train_set = cls(
            shuffled.iloc[:split_index],
            tokenizer=self.tokenizer,
            target_tokenizer=self.target_tokenizer,
            max_len=self.max_len,
            no_punkt=self.no_punkt,
        )
        valid_set = cls(
            shuffled.iloc[split_index:],
            tokenizer=self.tokenizer,
            target_tokenizer=self.target_tokenizer,
            max_len=self.max_len,
            no_punkt=self.no_punkt,
        )
        return train_set, valid_set

class Seq2SeqData(pl.LightningDataModule):
    def __init__(self, df, tokenizer, target_tokenizer, params, no_punkt:bool=False):
        super().__init__()
        self.df = df
        self.ds = Seq2Seq(df, tokenizer, target_tokenizer, max_len=params.max_len, no_punkt=no_punkt)
        self.tokenizer = tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_len = params.max_len
        self.batch_size = params.batch_size

    def setup(self, stage=None):
        self.train_set, self.valid_set = self.ds.split_train_valid()

    def train_dataloader(self):
        return self.train_set.dataloader(batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return self.valid_set.dataloader(batch_size=self.batch_size*2, shuffle=False)

class FineTuneModel(pl.LightningModule):
    def __init__(self, params=None):
        super(FineTuneModel, self).__init__()
        if params is None:
            params = self.hparams
        else:
            self.save_hyperparameters(params)
        
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path = self.hparams.encoder,
            decoder_pretrained_model_name_or_path = self.hparams.decoder,
        )
    
    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=params.lr, weight_decay=params.weight_decay)

def inference(text, tokenizer, model):
    kwargs = dict(
      truncation=True,
      max_length=128,
      padding="max_length",
      return_tensors='pt')
   
    inputs = tokenizer([text,],**kwargs)
    with torch.no_grad():
        return tokenizer.batch_decode(
            model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=4,
            max_length=256,
            bos_token_id=101,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ), skip_special_tokens=True)

def main(params):
    TO_CLASSICAL = False
    df = pd.read_csv(params.df)
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer)
    data_module = Seq2SeqData(df, tokenizer, tokenizer, params,
                              no_punkt=False if TO_CLASSICAL else True)

    model = FineTuneModel(params)
    current_directory = os.path.dirname(os.path.realpath(__file__))
    save = pl.callbacks.ModelCheckpoint(
        dirpath=current_directory,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    trainer = pl.Trainer(
        accelerator = 'gpu',
        callbacks=[save],
        precision="bf16-mixed",
        accumulate_grad_batches=params.acc_grad,
        max_epochs=params.epochs,
    )

    trainer.fit(model, data_module)

    current_path = os.getcwd()
    if not os.path.exists('model'):
        os.makedirs('model')

    model_path = os.path.join(current_path, "model")

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # print one translation for test
    # model.load_state_dict(torch.load(save.best, map_location='cpu')['state_dict'])
    # module = model.model
    # module.cpu()
    # module.eval()

    # print(inference(df.source[0], tokenizer, module))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    # the default values here is not the best parameter so far!
    parser.add_argument("--df", type=str, default="data.csv")
    parser.add_argument("--encoder", type=str, default="bert-base-chinese")
    parser.add_argument("--decoder", type=str, default="bert-base-chinese")
    parser.add_argument("--tokenizer", type=str, default="bert-base-chinese")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--acc_grad", type=int, default=4)

    params, unknown = parser.parse_known_args()
    main(params)