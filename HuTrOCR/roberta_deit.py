import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'  
import torch , json ,evaluate ,argparse , logging
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
                            TrOCRProcessor, Seq2SeqTrainer, set_seed ,
                            Seq2SeqTrainingArguments,default_data_collator
                         ) 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers.trainer_utils import get_last_checkpoint
from evaluate import load 
from datetime import datetime

time_now = datetime.now()
# logging.basicConfig(
#     level=logging.INFO, 
#     format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)
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

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids   = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # print('\n pred_str : ',pred_str)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # print('\n \n label_str : ', label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str) *100
    wer = wer_metric.compute(predictions=pred_str, references=label_str) *100

    return {"cer": cer, "wer": wer}

class OCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, tokenizer, max_target_length= 96):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

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
        labels = self.tokenizer(text, 
                                stride=32,
                                truncation=True,
                                padding="max_length", 
                                max_length=self.max_target_length).input_ids        
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]
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
                                tokenizer = tokenizer,
        			            processor = processor,
                                max_target_length = 96,
                               )
    eval_dataset =  OCRDataset( root_dir=imgs,
        			            df= val_df,
                                tokenizer = tokenizer,
        			            processor = processor,
                                max_target_length= 96, 
                               )
    return train_dataset, eval_dataset ,train_df

# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")


# def print_summary(result):
#     print(f"Time: {result.metrics['train_runtime']:.2f}")
#     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#     print_gpu_utilization()
    
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

    # if train_notebook:
    #     args = parser.parse_args([])
    # else:
    #     args = parser.parse_args()
    # return args     
def load_jsonl():   
    return pd.read_json(
                         path_or_buf = train_text, 
                         lines=True
                        )

def main():
    # is_sm_container = True    
    # if os.environ.get('SM_CURRENT_HOST') is None:
    #     is_sm_container = False        
    #     train_dir = 'train'
    #     model_dir = 'model'
    #     output_data_dir = 'data'
    #     src_dir = '/'.join(os.getcwd().split('/')[:-1])
    #     #src_dir = os.getcwd()
    #     os.environ['SM_MODEL_DIR'] = f'{src_dir}/{model_dir}'
    #     os.environ['SM_OUTPUT_DATA_DIR'] = f'{src_dir}/{output_data_dir}'
    #     os.environ['SM_NUM_GPUS'] = str(1)
    #     os.environ['SM_CHANNEL_TRAIN'] = f'{src_dir}/{train_dir}'

    # args = parser_args(train_notebook=True)    
    print(args)
    # config = vars(args)
    # train_text = config['text_path']
    # imgs  = config['images_path']
    # if os.environ.get('SM_CURRENT_HOST') is None:
    #     args.chkpt_dir = 'chkpt'
        
    n_gpus = torch.cuda.device_count()
    print('n_gpus', n_gpus)
    # if os.getenv("SM_NUM_GPUS")==None:
    #     print("Explicitly specifying the number of GPUs.")
    #     os.environ["GPU_NUM_DEVICES"] = n_gpus
    # else:
    #     os.environ["GPU_NUM_DEVICES"] = os.environ["SM_NUM_GPUS"]
    
    # logger.info("***** Arguments *****")    
    # logger.info(''.join(f'{k}={v}\n' for k, v in vars(args).items()))
    
    # os.makedirs(args.chkpt_dir, exist_ok=True) 
    # os.makedirs(args.model_dir, exist_ok=True)
    # os.makedirs(args.output_data_dir, exist_ok=True)    
    
    # df['text'] = df['text'].str.strip()

    df = load_jsonl()
    print(df.head(2),df.tail(2)) 
    # From testing max seq length we can guess the length of the token 
    # 1- get max seq in df 2- do tokenize with the corresponding tokenizer and get its length by index  
    # 3 - set value to model config max_len
    print('Max sequence length is :',df.text.str.len().max())
    print('Index of Max seq length',df.text.str.len().idxmax())
    print('Text from max seq length : ', df['text'][df.text.str.len().idxmax()])
    print('Path to image with max Seq. ', imgs +df['file_name'][df.text.str.len().idxmax()])
    suggested_max_token_len = len(tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
    print('Length of max tokenized row ',suggested_max_token_len,
          'Token ids for max tokens', (tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
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
    labels[labels == -100] = tokenizer.pad_token_id
    labels = labels[:96]
    label_str = tokenizer.decode(labels, skip_special_tokens=True)
    print(label_str)

    # set decoder config to causal lm
    # model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True
    
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    # if suggested_max_token_len > 64 :
    #     model.config.max_length = suggested_max_token_len  
    # else :
    #     model.config.max_length = 64 # 94 #128 
    # model.config.max_length = 32
    # model.config.max_length = suggested_max_token_len if suggested_max_token_len > 94 else 94
    model.config.max_length = max(suggested_max_token_len, 96)
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    working_dir = '/data/aramis/ngyongyossy/Roberta-base_Deit/Roberta-base-Deit_hu_lines_v2'
    training_args = Seq2SeqTrainingArguments(
                                              predict_with_generate=True,
                                              evaluation_strategy="steps",
                                              per_device_train_batch_size= 32, 
                                              per_device_eval_batch_size= 32, 
                                              num_train_epochs= 25,#
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
                              tokenizer=tokenizer,
                              args=training_args,
                              compute_metrics=compute_metrics,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              data_collator=default_data_collator,
                            )

    checkpoint_dir = ''
    trainer.train() 
    
    # import glob
    # sample_img_paths = glob.glob(imgs)  # 'sample_imgs/*.jpg'
    # img_idx = np.random.randint(len(sample_img_paths))
    # image = Image.open(sample_img_paths[img_idx])
    # #img_idx = np.random.randint(len(eval_dataset))
    # #image = Image.open(eval_dataset.dataset_dir + train_df['file_name'][img_idx])
    # print(image)

    # device = torch.device('cuda')

    # clean env 
    # !rm -rf model
    # Upload to Hugging Face Hub
    # !pip install huggingface_hub
    # from huggingface_hub import notebook_login
    # notebook_login()

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works               
    tokenizer.save_pretrained(f'{working_dir}/tokenizer') 
    processor.save_pretrained(f'{working_dir}/processor')        
    trainer.save_model(output_dir = f'{working_dir}/model')
    # Or model.save_pretrained(f'{working_dir}model')

    # After training, access the path of the best checkpoint 
    best_ckpt_path = trainer.state.best_model_checkpoint
    print('best_ckpt_path', best_ckpt_path)
    print('/-------- +++++++++ --------/')       
    
    # view last train results 
    # train_result = trainer.state.log_history[-1]
    # print(train_result)
    # Evaluation 
    # model.eval()
    # with torch.no_grad():
    #     eval_result = trainer.evaluate(eval_dataset, max_length=94) 
    # print('\n\neval_result\n',eval_result)


    # for i in trainer.state.log_history[:2]:
    #     print(type(i), i['loss'])

    # for i in trainer.state.log_history:
    #     print(type(i), i['step']) 

    # from pandas.core.dtypes.base import object_hash
    # if FULL_TRAINING:
    # steps = []
    # losses = []
    # for obj in trainer.state.log_history:
    #     # print(obj['step'])
    #     # print(obj['train_loss'])
    #     steps.append(obj['step'])
    #     # losses.append(obj['loss'])
    #     losses.append(obj['epoch'])
    #     # print(obj)

    # f = plt.figure(figsize=(12,6))
    # plt.plot(steps, losses)
    # plt.xlabel('step')
    # plt.ylabel('training loss')
    # plt.show()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    FULL_TRAINING =  True  
    if FULL_TRAINING:
        vision_model = 'facebook/deit-base-distilled-patch16-384'
        nlp_model    = "roberta-base" 

        # Reference: https://github.com/huggingface/transformers/issues/15823
        # initialize the encoder from a pretrained ViT and the decoder from a pretrained BERT model. 
        # Note that the cross-attention layers will be randomly initialized, and need to be fine-tuned on a downstream dataset
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(vision_model, nlp_model)
        tokenizer = AutoTokenizer.from_pretrained(nlp_model)
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

# python3 train4_Roberta.py /home/ngyongyossy/mohammad/Data/lines_hu_v2_1/train.jsonl /home/ngyongyossy/mohammad/Data/lines_hu_v2_1/images/ 
#  Only train set 
