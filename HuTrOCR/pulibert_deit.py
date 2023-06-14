import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5' 
import numpy as np
import pandas as pd 
import torch ,evaluate ,argparse, logging 
from tqdm import tqdm
from pynvml import *
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
                            AutoTokenizer, VisionEncoderDecoderModel,
                            TrOCRProcessor, RobertaTokenizer, Seq2SeqTrainer, 
                            Seq2SeqTrainingArguments,default_data_collator,
                         ) 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset 
from evaluate import load 
from PIL import Image
from datetime import datetime

time_now = datetime.now()
logging.basicConfig(
    level=logging.INFO, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

train_notebook = True     
parser = argparse.ArgumentParser( description="Example script for lavearging Robeta with Deit model",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                )

# Featuers + Labels
parser.add_argument("--text_path", default= '../Data/DH-Lab/train.jsonl', help="Location of transcriptions (single train file)")
parser.add_argument("--images_path", default= '../Data/DH-Lab/images/', help="Location of image files (folder)")

#  Default Setting
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--evaluation_strategy", type=str, default="steps")
parser.add_argument("--predict_with_generate", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--stride", type=int, default=32)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--logging_steps", type=int, default=10)  
parser.add_argument("--save_steps", type=int, default= 5000) 
parser.add_argument("--eval_steps", type=int, default= 5000) 
parser.add_argument("--save_total_limit", type=int, default= 1)
parser.add_argument("--learning_rate", type=str, default=5e-5)
parser.add_argument("--disable_tqdm", type=bool, default=False)
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--load_best_model_at_end", type=bool, default=True)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--report_to", type=list, default=["tensorboard"]) 
parser.add_argument("--resume_from_checkpoint", type=bool or str, default=False)
parser.add_argument("--gradient_checkpointing", type= bool, default= False)
parser.add_argument("--full_train", type= bool, default = True )
parser.add_argument("--nlp_model_dir", type=str, default='NYTK/PULI-BERT-Large')
parser.add_argument("--vision_model_dir", type=str, default='facebook/deit-base-distilled-patch16-384')   
parser.add_argument("--processor_dir", type=str, default='microsoft/trocr-base-handwritten')
parser.add_argument("--ft_model_id", type=str, default='microsoft/trocr-large-handwritten')

# Model Configuration 
parser.add_argument("--num_beams", type=int, default= 4 ) 
parser.add_argument("--max_length", type=int, default= 128)
parser.add_argument("--early_stopping", type= bool, default= True ) 
# OCR_HU_Tra2022 Container environment 
parser.add_argument("--working_dir", type=str, default='Models/PULI-BERT_Deit/')
parser.add_argument("--n_gpus", type=str, default = '8')
if train_notebook :
    args = parser.parse_args()

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids   = pred.predictions

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens= True)
    # print('\n pred_str : ',pred_str)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # print('\n \n label_str : ', label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str) *100
    wer = wer_metric.compute(predictions=pred_str, references=label_str) *100

    return {"cer": cer, "wer": wer}

class OCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length= args.max_length): 
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
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        '''
        add labels (input_ids) by encoding the text 
        https://huggingface.co/docs/transformers/pad_truncation
        '''
        labels = self.processor.tokenizer( text, 
                                           stride=args.stride, # during training trocr large this commented
                                           truncation=True, # this is will solve not equal lengths during batch passing
                                           padding="max_length", 
                                           max_length=self.max_target_length).input_ids       
        # Important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        '''
        Skiping <s>  start of token comming from tokenizer because this will be seting by Trocr model 
        the returned value here is encoding 
        '''
        labels= labels[1:]
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

def create_datasets(df: pd.DataFrame):
    '''
    We split up the data into training + validation, using sklearn's train_test_split function.
    And the test set alrady disjoint 
    '''
    train_df, val_df = train_test_split(df, test_size=0.1, random_state= 42069)
    # We reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_dataset = OCRDataset( root_dir= args.images_path,
        			            df = train_df,
        			            processor = processor,
                                max_target_length = args.max_length,
                               )
    eval_dataset =  OCRDataset( root_dir= args.images_path,
        			            df= val_df,
        			            processor = processor,
                                max_target_length= args.max_length, 
                               )
    return train_dataset, eval_dataset, train_df

def load_jsonl():   
    return pd.read_json(
                         path_or_buf = args.text_path, 
                         lines=True
                        )
def main():
    
    logger.info("***** Arguments *****")    
    logger.info(''.join(f'{k}={v}\n' for k, v in vars(args).items()))

    # os.makedirs(args.chkpt_dir, exist_ok=True) 
    # os.makedirs(args.model_dir, exist_ok=True)
    # os.makedirs(args.output_data_dir, exist_ok=True)    
    
    df = load_jsonl()
    print(df.head(2),df.tail(2)) 
    # From testing max seq length we can guess the length of the tokens by :  
    # 1- Getting the  max Seq. in the  df 
    # 2- Do tokenize with the corresponding tokenizer and get its length by index  
    # 3- Set value to model config (max_len)
    print('Max sequence length is : ',df.text.str.len().max())
    print('\n Index of Max seq length is : ',df.text.str.len().idxmax())
    print('\n Text for max Seq length  : ', df['text'][df.text.str.len().idxmax()])
    print('\n Path to image with max Seq. ', args.images_path +df['file_name'][df.text.str.len().idxmax()])
    if args.full_train == False:
        tokenizer = RobertaTokenizer.from_pretrained(args.nlp_model_dir)
        suggested_max_token_len = len(tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
        print('\n The Length of max tokenized row  : ',suggested_max_token_len,
              '\n Token ids for max tokens : ', (tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
             )
    else:
        suggested_max_token_len = len(processor.tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
        print('\n The Length of max tokenized row  : ',suggested_max_token_len,
              '\n Token ids for max tokens : ', (processor.tokenizer(df['text'][df.text.str.len().idxmax()])["input_ids"])
             )
 
    train_dataset, eval_dataset, train_df = create_datasets(df) 

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))  

    '''
    Set decoder config to causal lm
    model.config.decoder.is_decoder = True
    '''
    model.config.decoder.add_cross_attention = True
    '''
    Let's verify an example from the training dataset
    set special tokens used for creating the decoder_input_ids from the labels
    '''
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id # 2 In case PULI
    model.config.pad_token_id = processor.tokenizer.pad_token_id # 0 In case PULI
    '''
    Here if we are using Languge Model(LM) to be fine-tune 
    it needs some special setup. 
    '''
    if args.nlp_model_dir == 'NYTK/PULI-BERT-Large':
        model.config.pad_token = '[PAD]' 
    # Make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id =  processor.tokenizer.sep_token_id # 3
    model.config.max_length = max(suggested_max_token_len,  args.max_length)
    model.config.early_stopping = args.early_stopping
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = args.num_beams

    # Check by selecting sample randomly to see if the label matching the image
    img_idx = np.random.randint(len(train_df))
    encoding = train_dataset[img_idx]
    for k,v in encoding.items():
        print(k, v.shape)

    print(train_dataset.root_dir + train_df['file_name'][img_idx])    
    image = Image.open(train_dataset.root_dir + train_df['file_name'][img_idx]).convert("RGB")
    print(image)  

    labels = encoding['labels']
    labels[labels == -100] = processor.tokenizer.pad_token_id
    labels = labels[:model.config.max_length]
    # If we are using TrOCR large we need to change to processor.decode instead of processor.tokenizer.decode
    label_str = processor.tokenizer.decode(labels, skip_special_tokens= True)
    print('label_str : ',label_str)

    training_args = Seq2SeqTrainingArguments(
                                              predict_with_generate = args.predict_with_generate,
                                              evaluation_strategy = args.evaluation_strategy,
                                              per_device_train_batch_size = args.train_batch_size, 
                                              per_device_eval_batch_size  = args.eval_batch_size, 
                                              num_train_epochs = args.epochs,
                                              fp16 = args.fp16,
                                              learning_rate = float(args.learning_rate), 
                                              output_dir = args.working_dir,
                                              # In case TrOCR may need to uncomment line below to save logs
                                              # logging_dir=f'{args.working_dir}/logs',
                                              logging_steps=args.logging_steps,
                                              save_steps = args.save_steps,
                                              eval_steps = args.eval_steps,
                                              save_total_limit = args.save_total_limit,
                                              report_to= args.report_to,
                                              load_best_model_at_end = args.load_best_model_at_end,
                                              # Gradient check-pointing is only needed if training leads to out-of-memory (OOM) errors 
                                              gradient_checkpointing = args.gradient_checkpointing, 
                                            )
    # instantiate trainer
    trainer = Seq2SeqTrainer(
                              model = model,
                              tokenizer = processor.feature_extractor if args.full_train else processor.tokenizer,                              
                              args = training_args,
                              compute_metrics = compute_metrics,
                              train_dataset = train_dataset,
                              eval_dataset  = eval_dataset,
                              data_collator = default_data_collator,
                            )
    # checkpoint_dir = args.checkpoint_dir
    trainer.train() 
    
    # Saves the model, tokenizer and processor to make sure checkpointing works with correct config 
    processor.tokenizer.save_pretrained(f'{args.working_dir}tokenizer') 
    processor.save_pretrained(f'{args.working_dir}processor')        
    trainer.save_model(output_dir = f'{args.working_dir}model')
    '''
      Or model.save_pretrained(f'{working_dir}model')

      After training, access the path of the best checkpoint like this
    ''' 
    best_ckpt_path = trainer.state.best_model_checkpoint
    print('best_ckpt_path', best_ckpt_path)
    print('/-------- +++++++++ --------/')       
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
    # view last train results 
    train_result = trainer.state.log_history[-1]
    print(train_result)

    # Running the Evaluation 
    model.eval()
    with torch.no_grad():
        eval_result = trainer.evaluate(eval_dataset, max_length = args.max_length) 
    print('\n\neval_result\n',eval_result)


if __name__ == "__main__":

    print(args)    
    n_gpus = torch.cuda.device_count()
    print('n_gpus', n_gpus)
    
    processor = TrOCRProcessor.from_pretrained(args.processor_dir)
    if args.full_train:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(args.vision_model_dir, args.nlp_model_dir)
        processor.tokenizer = AutoTokenizer.from_pretrained(args.nlp_model_dir)   
    else:
        '''
        Fine-tunning
        or Re pre-training 
        '''
        model = VisionEncoderDecoderModel.from_pretrained(args.ft_model_id)
        processor.tokenizer = AutoTokenizer.from_pretrained(args.ft_model_id)

    print('Current DateTime:', time_now)
    print('Type:', type(time_now))
    main() 

    time_after_train_finsh = datetime.now()
    print('Current DateTime time_after_train_finsh :', time_after_train_finsh)
    print('Type:', type(time_after_train_finsh))