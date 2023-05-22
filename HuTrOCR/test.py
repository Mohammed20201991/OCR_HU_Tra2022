# from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
# import requests 
# from io import BytesIO
# from PIL import Image

# processor = TrOCRProcessor.from_pretrained("/home/ngyongyossy/mohammad/OCR_HU_Tra2022/1/base_lines_hu_v4/processor") 
# model = VisionEncoderDecoderModel.from_pretrained("/home/ngyongyossy/mohammad/OCR_HU_Tra2022/1/base_lines_hu_v4/checkpoint-50")

# img =  '/home/ngyongyossy/mohammad/OCR_HU_Tra2022/GPT-2_Parallel/process/MixedData/images/276-17.jpg'
# img = Image.open(img).convert("RGB")
# pixel_values = processor(images=img, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_ids ,generated_text)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import pandas as pd
import torch ,argparse ,evaluate
from torch.utils.data import Dataset , DataLoader
from PIL import Image
from transformers import TrOCRProcessor ,VisionEncoderDecoderModel
from tqdm.notebook import tqdm
from evaluate import load
parser = argparse.ArgumentParser(description="Example script for Testing TrOCR model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("test_path", help="Location of transcriptions (single text JSON lines file)")
parser.add_argument("imgs_path", help="Location of test image files (folder)")
args = parser.parse_args()
config = vars(args)

test_text = config['test_path']
test_imgs = config['imgs_path']
GET_MODEL_BY_ID = False 
SKIP_SPECIAL_TOKENS = True
cer = load("cer")
wer = load("wer")

class OCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length= 64): 
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
                                        #   stride=32,# roberta case 
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
                        path_or_buf = test_text,
                        lines=True) 
def main(): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODL_ID = 'AlhitawiMohammed22/trocr_large_hu_Arrany_3'
    load_model_from_checkpoint_dir = '/home/ngyongyossy/mohammad/OCR_HU_Tra2022/1/trocr_large_lines_44_ft_on_dh-lab_aug/'
    if GET_MODEL_BY_ID: 
        processor = TrOCRProcessor.from_pretrained(MODL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(MODL_ID)
    else:
        processor = TrOCRProcessor.from_pretrained(f'{load_model_from_checkpoint_dir}processor') 
        model = VisionEncoderDecoderModel.from_pretrained(f'{load_model_from_checkpoint_dir}checkpoint-81000')
    model.to(device) 

    df =   load_jsonl()
    print(df.head(2),df.tail(2)) 
    test_dataset = OCRDataset(root_dir=test_imgs,
                              df=df,
                              processor=processor)    
    test_dataloader = DataLoader(test_dataset, batch_size= 24)
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
        outputs = model.generate(pixel_values , max_length = 64) # repres max_seq -12 #,eos_token ,pad_token_id=processor.tokenizer.eos_token_id
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
# python3 test.py path/to/test.jsonl  path/to/images/