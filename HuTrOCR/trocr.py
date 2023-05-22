import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import evaluate, torch, argparse
from evaluate import load
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                         VisionEncoderDecoderModel,default_data_collator ,
                         TrOCRProcessor, RobertaTokenizer)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

parser = argparse.ArgumentParser(description="Example script for finetuning TrOCR model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("text_path", help="Location of transcriptions (single text file)")
parser.add_argument("images_path", help="Location of image files (folder)")
args = parser.parse_args()
config = vars(args)    

cer_metric = load("cer")
wer_metric = load("wer")
accuracy   = load('accuracy')

train_text = config['text_path']
imgs       = config['images_path']

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
def compute_metrics(pred):
    labels_ids = pred.label_ids
    # print('labels_ids',len(labels_ids), type(labels_ids),labels_ids)
    pred_ids = pred.predictions
    # print('pred_ids',len(pred_ids), type(pred_ids),pred_ids)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # print(pred_str)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    # print(label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str) * 100
    wer = wer_metric.compute(predictions=pred_str, references=label_str) * 100
    # accu = accuracy.compute(references=labels_ids, predictions=pred_ids) * 100
    # 'accu': accu
    return {"cer": cer, 'wer': wer, }

def load_jsonl():
    return pd.read_json(
                        path_or_buf = train_text,
                        lines=True) 

class OCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=96):
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
		# https://huggingface.co/docs/transformers/pad_truncation
        labels = self.processor.tokenizer(text, 
                                          padding="max_length",
					                      truncation=True, # this is now trying to solve not equal lengths during batch passing 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        # May be passing config file during load model could be othere possiblity 
        # skiping <s>  start of token comming from tokenizer because this will be seting by Trocr model 
        labels = labels[1:]
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
        			            processor=processor)
    eval_dataset =  OCRDataset( root_dir=imgs,
        			            df= val_df,
        			            processor=processor)
    return train_dataset, eval_dataset ,train_df

def main():
    df = load_jsonl()
    print(df.head(2),df.tail(2))
    # From testing max seq length we can guess the length of the token 
    # 1- get max seq in df 2- do tokenize with the corresponding tokenizer and get its length by index  
    # 3 - set value to model config max_len
    print('Max sequence length is :',df.text.str.len().max())
    print('Index of Max seq length',df.text.str.len().idxmax())
    print('Text from max seq length : ', df['text'][df.text.str.len().idxmax()])
    print('Path to image with max Seq. ', imgs +df['file_name'][df.text.str.len().idxmax()])
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    suggested_max_token_len = len(tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
    print('Length of max tokenized row ',suggested_max_token_len,
          'Token ids for max toks', (tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"]))
 
    train_dataset, eval_dataset ,train_df = create_datasets(df) # , may it need to pass processor 
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))
    # Let's verify an example from the training dataset
    encoding = train_dataset[0]
    for k,v in encoding.items():
       print(k, v.shape)
    # We can also check the original image and decode the labels:          
    image = Image.open(train_dataset.root_dir + train_df['file_name'][0]).convert("RGB")
    print(image)

    labels = encoding['labels']
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.decode(labels, skip_special_tokens=True) 
    print(label_str)

    FULL_TRAIN = True
    if FULL_TRAIN:
         #Train a model
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        # Set special tokens used for creating the decoder_input_ids from the labels
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        # Make sure vocab size is set correctly
        model.config.vocab_size = model.config.decoder.vocab_size
        # set beam search parameters
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        model.config.max_length = suggested_max_token_len -10 # 64 #128
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

        # set decoder config to causal lm
        model.config.decoder.is_decoder = True
        model.config.decoder.add_cross_attention = True
        # at this experiment with hu_lines for bigger data 14-4-23
        working_dir =  './trocr_large_hu_lines_v2_2/'
        training_args = Seq2SeqTrainingArguments(
        		                                 learning_rate=5e-5,
        		                                 num_train_epochs=25, 
                                                 predict_with_generate=True,
                                                 evaluation_strategy="steps",
                                                 per_device_train_batch_size = 64, # 100 4-5-23
                                                 per_device_eval_batch_size  = 64, # 100
                                                 fp16=True, 
                                                 output_dir=f"{working_dir}",
        		                                 report_to=["tensorboard"],
        		                                 logging_dir=f'{working_dir}logs', 
                                                 logging_steps=500,
                                                 save_steps=5000,
                                                 eval_steps=5000,
                                                 save_total_limit = 1,
                                                 load_best_model_at_end =True, 
                                                 # Gradient check-pointing is only needed if training leads to out-of-memory (OOM) errors 
                                                 gradient_checkpointing =True, 
        		                                )
        # instantiate trainer
        trainer = Seq2SeqTrainer(
                                  model=model,                
                                  args=training_args,
                                  train_dataset = train_dataset,
                                  eval_dataset = eval_dataset,
                                  compute_metrics=compute_metrics,
                                  tokenizer = processor.feature_extractor,
                                  data_collator = default_data_collator,
                                 )
    # checkpoint = '/home/ngyongyossy/mohammad/OCR_HU_Tra2022/1/trocr_large_hu_lines_v2_2/checkpoint-85000'
    trainer.train(resume_from_checkpoint= True)
    processor.tokenizer.save_pretrained(f'{working_dir}tokenizer')       
    trainer.save_model(output_dir = f'{working_dir}model')
    processor.save_pretrained(f'{working_dir}processor') 
    # After training, access the path of the best checkpoint like this
    best_ckpt_path = trainer.state.best_model_checkpoint
    print('best_ckpt_path', best_ckpt_path)
    print('/---------')    
    train_loss = []
    for elem in trainer.state.log_history:
        if 'loss' in elem.keys():
         train_loss.append(elem['loss']) 
    print('train_loss', train_loss)

    eval_loss = []
    for elem in trainer.state.log_history:
        if 'eval_loss' in elem.keys():
         eval_loss.append(elem['eval_loss'])    
    print('eval_loss', eval_loss)

    model.eval()
    with torch.no_grad():
        eval_result = trainer.evaluate(eval_dataset,
                                       max_length=suggested_max_token_len-10)
        print('\n eval_result : ', eval_result)  

if __name__ == '__main__':
    main()
# python3 train.py /home/ngyongyossy/mohammad/Data/lines_hu_v2_1/train.jsonl /home/ngyongyossy/mohammad/Data/lines_hu_v2_1/images/
#  Only train set s

