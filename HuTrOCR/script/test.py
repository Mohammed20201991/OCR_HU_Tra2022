import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Example script for testing TrOCR model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("model", help="Model location")
parser.add_argument("input", help="Destination location. It can be a file or a folder. See dump option!")
parser.add_argument("-d", "---dump", action="store_true", help="Process the content of a folder provided as input")
parser.add_argument("-o", "--output", help="Write output to a text file. If dump option is False it will be ignored")
args = parser.parse_args()
config = vars(args)

def main():
    
    # prepare model
	print("Loading model...")
	processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
	model = VisionEncoderDecoderModel.from_pretrained(config['model']).cuda()

	def process_image(img_path):
		
		print('Processing image: ' + img_path)
		
		# prepare image
		img = cv2.imread(img_path)
		pixel_values = processor(img, return_tensors="pt").pixel_values

		# generate (no beam search)
		generated_ids = model.generate(inputs=pixel_values.cuda(), max_length=50,num_beams=5, early_stopping=True)

		# decode
		generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		
		# display result
		print("Output: " + generated_text)
		
		if (config['output'] != None and config['dump'] == True):
			f = open(config['output'], "a")
			f.write(img_path + " " + generated_text + "\n")
	
	if (config['dump'] == True):
		for filename in os.scandir(config['input']):
			if filename.is_file():
				process_image(filename.path)
	else:
		process_image(config['input'])
	
if __name__ == '__main__':
    main()
