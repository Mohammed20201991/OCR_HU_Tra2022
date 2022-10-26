from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '5' # which_gpu_wanna_work_with    

# import sys
import pandas as pd
import torch
# print(torch.__version__)
from PIL import Image
from datasets import load_metric
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrOCRProcessor ,AutoTokenizer
from transformers import VisionEncoderDecoderModel ,AutoModel
from transformers import default_data_collator

# import argparse

torch.cuda.is_available()
# parser = argparse.ArgumentParser(description="Example script for finetuning TrOCR model",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("text_path", help="Location of transcriptions (single text file)")
# parser.add_argument("images_path", help="Location of image files (folder)")
# args = parser.parse_args()
# config = vars(args)    

cer_metric = load_metric("cer")
wer_metric = load_metric('wer')
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# train_text = config['text_path']
# train_img = config['images_path']

train_text = "/home/ngyongyossy/mohammad/asdf/TrOCR-finetune/training-data-ex/lines.txt"
train_img  = "/home/ngyongyossy/mohammad/asdf/TrOCR-finetune/training-data-ex/img/"

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

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


def load_dataset():
    df = pd.read_fwf('dataset/gt_test.txt', header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    df['file_name'] = df['file_name'].apply(lambda x: x + '.jpg')

    return df


def load_laia() -> pd.DataFrame:

    df = pd.read_csv('/home/ngyongyossy/mohammad/asdf/TrOCR-finetune/training-data-ex/lines.txt',sep=' ', header=None)
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
