import pandas as pd
import torch ,argparse ,evaluate
from torch.utils.data import Dataset , DataLoader
from PIL import Image
from transformers import TrOCRProcessor ,VisionEncoderDecoderModel
from tqdm.notebook import tqdm
from evaluate import load

test_notebook = True   
parser = argparse.ArgumentParser()

# Featuers + Labels(for test set)
parser.add_argument("--text_path", default= '../Data/DH-Lab/test.jsonl', help="Location of transcriptions (single train file)")
parser.add_argument("--images_path", default= '../Data/DH-Lab/images/', help="Location of image files (folder)")

parser.add_argument("--test_batch_size", type=int, default = 16)
parser.add_argument("--get_by_model_id", type=bool, default =False)
parser.add_argument("--stride", type=int, default = 32)
parser.add_argument("--nlp_model_dir", type=str, default='AlhitawiMohammed22/trocr_large_hu_lines_v2_2')
parser.add_argument("--model_id", type=str, default='AlhitawiMohammed22/trocr_large_hu_lines_v2_2')
parser.add_argument("--processor_dir", type=str, default='AlhitawiMohammed22/trocr_large_hu_lines_v2_2')
parser.add_argument("--load_model_from_checkpoint_dir", type=str, default='./Models/TrOCR_large_handwritten')
# Model Configuration 
parser.add_argument("--num_beams", type=int, default= 4) 
parser.add_argument("--max_length", type=int, default= 128)

if test_notebook:
    args = parser.parse_args()

print('args : ', args)
SKIP_SPECIAL_TOKENS = True
cer = load("cer")
wer = load("wer")

class OCRDataset(Dataset):
    '''
    In this class OCRDataset return for each element of the json line row dataset:
    pixel_values(features), which serve as input to the model.
    target, which are the input_ids of the corresponding text in the image.
    The processor type is TrOCRProcessor to prepare the data for the model. \n
    TrOCRProcessor is actually just a wrapper around a ViTFeatureExtractor (which can be used to resize + normalize images).\n
    And a RobertaTokenizer (which can be used to encode and decode text into/from input_ids).
    '''
    def __init__(self, root_dir, df, processor, max_target_length= args.max_length): 
        '''
        Pass the necessary arguments 
        '''
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        '''
        Get the length for the dataframe
        '''
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Get one elemnt by index at each time step and process it.
        '''
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
                                          #stride=args.stride, # roberta case and puli 
                                          padding="max_length",
                                          truncation=True, # this is new trying to solve not equal lenths 
                                          max_length=self.max_target_length).input_ids
        '''
        Skiping <s>  start of token comming from tokenizer because this will be seting by Trocr model 
        the returned value here is encoding 
        '''
        labels = labels[1:]
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        # print('labels', labels)
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

def load_jsonl():
    '''
    This function recive the passed path by args path in jsonline format and return pandas dataframe.
    '''
    return pd.read_json(
                        path_or_buf = args.text_path,
                        lines=True) 
def main(): 
    '''
    The main function do mosrt of the things first of all dataset loading, And then choose appropriate sequence length \n
    based on selecting the index of maximum text length and dataframe labels and tokenize it and choose the length an approximately. 
    The second thing is to use class for loading to data trainer class and spilt it .\n
    Setup Model Configuration
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.get_by_model_id: 
        processor = TrOCRProcessor.from_pretrained(args.processor_dir)
        model = VisionEncoderDecoderModel.from_pretrained(args.model_id)
    else:
        processor = TrOCRProcessor.from_pretrained(f'{args.load_model_from_checkpoint_dir}/processor') 
        model = VisionEncoderDecoderModel.from_pretrained(f'{args.load_model_from_checkpoint_dir}/checkpoint-xxxx')
    model.to(device) 

    df = load_jsonl()
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
        outputs = model.generate(pixel_values , max_length = args.max_length)
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
    print("Character Error Rate on test set:", final_score_CER *100)
    print("Word Error Rate on test set:", final_score_WER *100)

if __name__ == '__main__':
    '''
    Statement when the program is run directly by the Python interpreter.\n
    And then call the main funtion.
    '''
    main()