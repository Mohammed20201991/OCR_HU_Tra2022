import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  
import torch ,evaluate ,argparse 
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display, HTML
from pynvml import *
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
                            AutoTokenizer, VisionEncoderDecoderModel, RobertaTokenizer,
                            TrOCRProcessor, Seq2SeqTrainer, set_seed ,BertTokenizer,
                            Seq2SeqTrainingArguments,default_data_collator
                         ) 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers.trainer_utils import get_last_checkpoint
from evaluate import load 
from datetime import datetime

time_now = datetime.now()

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

parser = argparse.ArgumentParser(description="Example script for finetuning TrOCR model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("text_path",   help="Location of transcriptions (single text file)")
parser.add_argument("images_path", help="Location of image files (folder)")
args = parser.parse_args()
config = vars(args)

train_text = config['text_path']
imgs       = config['images_path']

skip_special_tokens = True
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids   = pred.predictions

    # print('pred_ids : \n',pred_ids)
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens= skip_special_tokens)
    # print('\n pred_str : ',pred_str)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=skip_special_tokens)
    # print('\n \n label_str : ', label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str) *100
    wer = wer_metric.compute(predictions=pred_str, references=label_str) *100

    return {"cer": cer, "wer": wer}

class OCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length= 96): #, tokenizer
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.tokenizer = processor.tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text      
        labels = self.processor.tokenizer( text, 
                                           stride=32,
                                           truncation=True,
                                           padding="max_length", 
                                           max_length=self.max_target_length).input_ids 
        # print('labels with input_ids', labels)       
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        # to do here skip start token 
        # print('\n labels: \n', labels)
        labels= labels[1:]
        # encoding 
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

# We split up the data into training + testing, using sklearn's train_test_split function.
def create_datasets(df: pd.DataFrame):
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42069)
    # We reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_dataset = OCRDataset( root_dir=imgs,
        			            df=train_df,
                                # tokenizer = tokenizer,
        			            processor = processor,
                                max_target_length = 96,
                               )
    eval_dataset =  OCRDataset( root_dir=imgs,
        			            df= val_df,
                                # tokenizer = tokenizer,
        			            processor = processor,
                                max_target_length= 96, 
                               )
    return train_dataset, eval_dataset ,train_df
    
# def parser_args(train_notebook=False):
    # parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description="Example script for finetuning TrOCR model",
    #                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--text_path", default= '/home/ngyongyossy/mohammad/Data/hu_lines/lines.txt', help="Location of transcriptions (single text file)")
    # parser.add_argument("--images_path", default= '/home/ngyongyossy/mohammad/Data/hu_lines/img/', help="Location of image files (folder)")
    # # args = parser.parse_args()

    # Default Setting
    # parser.add_argument("--epochs", type=int, default=12)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--train_batch_size", type=int, default=16)
    # parser.add_argument("--eval_batch_size", type=int, default=16)
    # parser.add_argument("--max_length", type=int, default=64)
    # parser.add_argument("--stride", type=int, default=32)
    # parser.add_argument("--warmup_steps", type=int, default=100)
    # parser.add_argument("--logging_steps", type=int, default=100)
    # parser.add_argument("--learning_rate", type=str, default=4e-5)
    # parser.add_argument("--disable_tqdm", type=bool, default=False)
    # parser.add_argument("--fp16", type=bool, default=True)
    # parser.add_argument("--debug", type=bool, default=False)      
    
    # SageMaker Container environment
    # parser.add_argument("--output_data_dir", type=str, default='models/_ft_hu') # os.environ["SM_OUTPUT_DATA_DIR"]
    # parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    # parser.add_argument("--train_dir", type=str, default='/home/ngyongyossy/mohammad/Data/hu_lines/img/') #os.environ["SM_CHANNEL_TRAIN"]
    # parser.add_argument('--chkpt_dir', type=str, default='/models/')     
def load_jsonl():   
    return pd.read_json(
                         path_or_buf = train_text, 
                         lines=True
                        )
def main():
    print(args)    
    n_gpus = torch.cuda.device_count()
    print('n_gpus', n_gpus)
    df = load_jsonl()
    print(df.head(2),df.tail(2)) 
    # From testing max seq length we can guess the length of the token 
    # 1- get max seq in df 2- do tokenize with the corresponding tokenizer and get its length by index  
    # 3 - set value to model config max_len
    print('Max sequence length is :',df.text.str.len().max())
    print('Index of Max seq length',df.text.str.len().idxmax())
    print('Text from max seq length : ', df['text'][df.text.str.len().idxmax()])
    print('Path to image with max Seq. ', imgs +df['file_name'][df.text.str.len().idxmax()])
    suggested_max_token_len = len(processor.tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
    print('Length of max tokenized row ',suggested_max_token_len,
          'Token ids for max tokens', (processor.tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
         )
 
    train_dataset, eval_dataset ,train_df = create_datasets(df) 

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))  
    img_idx = np.random.randint(len(train_df))
    encoding = train_dataset[img_idx]
    for k,v in encoding.items():
        print(k, v.shape)

    print(train_dataset.root_dir + train_df['file_name'][img_idx])    
    image = Image.open(train_dataset.root_dir + train_df['file_name'][img_idx]).convert("RGB")
    print(image)  

    labels = encoding['labels']
    labels[labels == -100] = processor.tokenizer.pad_token_id
    labels = labels[:96]
    label_str = processor.tokenizer.decode(labels, skip_special_tokens= skip_special_tokens)
    print('label_str',label_str)

    # set decoder config to causal lm
    # model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True
    
    # set special tokens used for creating the decoder_input_ids from the labels
    print('Before model.config.decoder_start_token_id : ', model.config.decoder_start_token_id)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id #2
    print( f'\n model.config.decoder_start_token    : {processor.tokenizer.cls_token}',
           f'\n model.config.decoder_start_token_id : {model.config.decoder_start_token_id}'
         )
    print(f'\n Before --> model.config.pad_token_id {model.config.pad_token_id}')
    model.config.pad_token_id = processor.tokenizer.pad_token_id # 0
    model.config.pad_token = '[PAD]'
    print(  f'\n model.config.pad_token{model.config.pad_token}',
            f'\n model.config.pad_token_id{model.config.pad_token_id}'
         )
    print(f'\n model.config.decoder.vocab_size : {model.config.decoder.vocab_size}')
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    print(f'\n Before model.config.eos_token_id{model.config.eos_token_id}') #3
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    print( f'\n model.config.eos_token{processor.tokenizer.sep_token}' ,
           f'\n model.config.eos_token_id{model.config.eos_token_id}'
          )
    # if suggested_max_token_len > 64 :
    #     model.config.max_length = suggested_max_token_len  
    # else :
    #     model.config.max_length = 64 # 94 #128 
    # model.config.max_length = 32
    # model.config.max_length = suggested_max_token_len if suggested_max_token_len > 94 else 94
    model.config.max_length = 96 # max(suggested_max_token_len, 32)
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # corrected name from 4 to new vertion v2-1
    working_dir = '/data/aramis/ngyongyossy/PULI_BERT_Large/PULI-Deit_lines_hu_v2_1' 
    training_args = Seq2SeqTrainingArguments(
                                              predict_with_generate=True,
                                              evaluation_strategy="steps",
                                              per_device_train_batch_size= 24, 
                                              per_device_eval_batch_size = 24, 
                                              num_train_epochs= 25,
                                              fp16= True ,
                                              learning_rate= 4e-5, 
                                              output_dir = working_dir, 
                                              logging_dir=f"{working_dir}/logs",
                                              logging_steps=100, 
                                              save_steps=5000, 
                                              eval_steps=5000, 
                                              save_total_limit = 1,
                                              report_to=["tensorboard"],
                                              load_best_model_at_end =True,       
                                            )
    # instantiate trainer
    trainer = Seq2SeqTrainer(
                              model=model,
                              tokenizer=processor.tokenizer,#feature_extractor
                              args=training_args,
                              compute_metrics=compute_metrics,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              data_collator=default_data_collator,
                            )
    checkpoint = '/data/aramis/ngyongyossy/PULI_BERT_Large/PULI-Deit_lines_hu_v2_1/checkpoint-505000'
    trainer.train(resume_from_checkpoint= checkpoint) 
    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works               
    processor.tokenizer.save_pretrained(f'{working_dir}/tokenizer') 
    processor.save_pretrained(f'{working_dir}/processor')        
    trainer.save_model(output_dir = f'{working_dir}/model')
    # Or model.save_pretrained(f'{working_dir}model')

    # After training, access the path of the best checkpoint 
    best_ckpt_path = trainer.state.best_model_checkpoint
    print('best_ckpt_path', best_ckpt_path)
    print('/-------- +++++++++ --------/')       

if __name__ == "__main__":
    FULL_TRAINING =  True  
    if FULL_TRAINING:
        vision_model = 'facebook/deit-base-distilled-patch16-384'
        nlp_model    = "NYTK/PULI-BERT-Large" 

        # Reference: https://github.com/huggingface/transformers/issues/15823
        # initialize the encoder from a pretrained ViT and the decoder from a pretrained BERT model. 
        # Note that the cross-attention layers will be randomly initialized, and need to be fine-tuned on a downstream dataset
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(vision_model, nlp_model)
        processor.tokenizer = AutoTokenizer.from_pretrained(nlp_model)# BertTokenizer
        print(f'Bfore pad_token{processor.tokenizer.pad_token}', f'pad_token_id   : {processor.tokenizer.pad_token_id}\n')
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'After pad_token': '[PAD]'})
        print(f'pad_token{processor.tokenizer.pad_token}', f'pad_token_id{processor.tokenizer.pad_token_id}\n')
        print(f'cls_token{processor.tokenizer.cls_token}', f'cls_token_id{processor.tokenizer.cls_token_id}\n')
        print(f'bos_token{processor.tokenizer.bos_token}', f'bos_token_id{processor.tokenizer.bos_token_id}\n')
        print(f'eos_token{processor.tokenizer.eos_token}', f'eos_token_id{processor.tokenizer.eos_token_id}\n')
        print(f'unk_token{processor.tokenizer.unk_token}', f'unk_token_id{processor.tokenizer.unk_token_id}\n')
        print(f'sep_token{processor.tokenizer.sep_token}', f'sep_token_id{processor.tokenizer.sep_token_id}\n')

        # print('\n tokenizer.config \n\n',tokenizer.config)
    else:
        trocr_model = 'microsoft/trocr-base-handwritten' 
        model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
        tokenizer = AutoTokenizer.from_pretrained(trocr_model)

    print('Current DateTime:', time_now)
    print('Type:', type(time_now))

    main() 

    time_after_train_finsh = datetime.now()
    print('Current DateTime time_after_train_finsh :', time_after_train_finsh)
    print('Type:', type(time_after_train_finsh))
# python3 train.py /home/ngyongyossy/mohammad/Data/lines_hu_v2_1/train.jsonl /home/ngyongyossy/mohammad/Data/lines_hu_v2_1/images/
#  Only train set 
# https://huggingface.co/blog/how-to-generate
# Asking to pad but the toakenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})