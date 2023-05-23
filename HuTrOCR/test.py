# for accessing private models type in your terminale 
# huggingface-cli login
# Tthen copy and past prvoided token ="token to be used will be secrit"
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import pandas as pd
import torch ,argparse ,evaluate
from torch.utils.data import Dataset , DataLoader
from PIL import Image
from transformers import TrOCRProcessor ,VisionEncoderDecoderModel
from tqdm.notebook import tqdm
from evaluate import load

# def parser_args(train_notebook=False):
train_notebook = False   
parser = argparse.ArgumentParser()

# Featuers + Labels
parser.add_argument("--text_path", default= '../Data/DH-Lab/test.jsonl', help="Location of transcriptions (single train file)")
parser.add_argument("--images_path", default= '../Data/DH-Lab/images/', help="Location of image files (folder)")

parser.add_argument("--test_batch_size", type=int, default = 16)
parser.add_argument("--get_model_id", type=bool, default = True)
parser.add_argument("--stride", type=int, default = 32)
parser.add_argument("--nlp_model_dir", type=str, default='AlhitawiMohammed22/trocr_large_hu_lines_v2_2')
parser.add_argument("--model_id", type=str, default='AlhitawiMohammed22/trocr_large_hu_lines_v2_2')
parser.add_argument("--processor_dir", type=str, default='AlhitawiMohammed22/trocr_large_hu_lines_v2_2')
parser.add_argument("--load_model_from_checkpoint_dir", type=str, default='./Models/PULI-BERT_Deit')
# Model Configuration 
parser.add_argument("--num_beams", type=int, default= 4 ) 
parser.add_argument("--max_length", type=int, default= 128)

args = parser.parse_args([]) if not train_notebook else parser.parse_args()
# return args
print('args : ', args)
SKIP_SPECIAL_TOKENS = True
cer = load("cer")
wer = load("wer")

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
        # some file names end with jp instead of jpg, the two lines below fix this
        if file_name.endswith('jp'):
          file_name = file_name + 'g'
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                        #   stride=args.stride, # roberta case and puli 
                                          padding="max_length",
                                          truncation=True, # this is new trying to solve not equal lenths 
                                          max_length=self.max_target_length).input_ids
        labels = labels[1:]
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        # print('labels', labels)
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

def load_jsonl():
    return pd.read_json(
                        path_or_buf = args.text_path,
                        lines=True) 
def main(): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.get_model_id: 
        processor = TrOCRProcessor.from_pretrained(args.processor_dir)
        model = VisionEncoderDecoderModel.from_pretrained(args.model_id)
    else:
        processor = TrOCRProcessor.from_pretrained(f'{args.load_model_from_checkpoint_dir}/processor') 
        model = VisionEncoderDecoderModel.from_pretrained(f'{args.load_model_from_checkpoint_dir}/checkpoint-81000')
    model.to(device) 

    df =   load_jsonl()
    print(df.head(2),df.tail(2)) 
    test_dataset = OCRDataset(root_dir= args.images_path,
                              df=df,
                              processor=processor)    
    test_dataloader = DataLoader(test_dataset, batch_size= args.test_batch_size)
    batch = next(iter(test_dataloader))
    print('batch shape : ',type(batch))
    for k,v in batch.items():
        print(k, v.shape)
    labels = batch["labels"]
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels, skip_special_tokens= SKIP_SPECIAL_TOKENS) # trunction ? to add is benficial during batch using 
    print('label_str',label_str)   

    print("Running evaluation...")
    for batch in tqdm(test_dataloader):
        # predict using generate
        pixel_values = batch["pixel_values"].to(device)
        outputs = model.generate(pixel_values , max_length = args.max_length) # repres max_seq -12 #,eos_token ,pad_token_id=processor.tokenizer.eos_token_id
        # decode
        pred_str = processor.batch_decode(outputs, skip_special_tokens = SKIP_SPECIAL_TOKENS)
        print('pred_str',pred_str)
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens = SKIP_SPECIAL_TOKENS)
        print('label_str', label_str)
        # add batch to metric
        cer.add_batch(predictions=pred_str, references=label_str)
        wer.add_batch(predictions=pred_str, references=label_str)
    # to do save test results in dict then jsonl format 
    final_score_CER = cer.compute()
    final_score_WER = wer.compute()
    print("Character error rate on test set:", final_score_CER *100)
    print("Word error rate on test set:", final_score_WER *100)

if __name__ == '__main__':
    main()