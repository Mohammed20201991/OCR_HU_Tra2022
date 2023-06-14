import sys
sys.path.append('/home/ngyongyossy/mohammad/asdf/TrOCR-finetune/')
import unit_test
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '6' # which_gpu_wanna_work_with    
import pandas as pd
import torch
import argparse
from PIL import Image
from datasets import load_metric
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments,
                          TrOCRProcessor ,
                          AutoTokenizer,
                          VisionEncoderDecoderModel ,
                          AutoModel,
                          default_data_collator)


torch.cuda.is_available()
# parser = argparse.ArgumentParser(description="Example script for finetuning TrOCR model",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("text_path", help="Location of transcriptions (single text file)")
# parser.add_argument("images_path", help="Location of image files (folder)")

# parser.add_argument("encoder_type", type = str ,help="Which type of featuers Extraction)")
# parser.add_argument("decoder_type", help="Which type of Text Generation (huggingface model)")

# args = parser.parse_args()
# config = vars(args)    

# cer_metric = load_metric("cer")
# wer_metric = load_metric('wer')
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# train_text = config['text_path']
# train_img = config['images_path']

# Encoder = config['encoder_type']
# Decoder = config['decoder_type']

# Decoder = "bert-base-uncased"
# Encoder = "google/vit-base-patch16-384"
# train_text = "/home/ngyongyossy/mohammad/asdf/TrOCR-finetune/training-data-ex/lines.txt"
# train_img  = "/home/ngyongyossy/mohammad/asdf/TrOCR-finetune/training-data-ex/img/"

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str) * 100
    wer = wer_metric.compute(predictions=pred_str, references=label_str) * 100
    return {"cer": cer, 'wer': wer}

class LaiaDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in
                  labels]
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}


def load_dataset():
    df = pd.read_fwf('dataset/gt_test.txt', header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    df['file_name'] = df['file_name'].apply(lambda x: x + '.jpg')

    return df


def load_laia() -> pd.DataFrame:
    # path = '/home/ngyongyossy/mohammad/asdf/TrOCR-finetune/training-data-ex/lines.txt'
    df = pd.read_csv(train_text,sep=' ', header=None)
    data = []
    with open(train_text) as infile:
        for line in infile:
            file_name, _, _, _, _, _, _, _, text = line.strip().split(' ')
            data.append((file_name, clean_text(text)))

    df = pd.DataFrame(data, columns=['file_name', 'text'])
    df.rename(columns={0: 'file_name', 8: 'text'}, inplace=True)
    df['file_name'] = df['file_name'].apply(lambda x: x + '.jpg')
    df = df[['file_name', 'text']]
    return df


def clean_text(input_text: str) -> str:
    text = input_text.replace('+', '-')
    text = text.replace('|', ' ')
    return text


def create_datasets(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.05, random_state=42069)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_dataset = LaiaDataset(
                                root_dir=train_img,
                                df=train_df,
                                processor=processor)
    eval_dataset = LaiaDataset(
                                root_dir=train_img,
                                df=test_df,
                                processor=processor)
    return train_dataset, eval_dataset


def trocr_model_config(model):
    # set decoder config to causal lm
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    assert model.config.decoder_start_token_id == processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    return model