# OCR_HU_Tra2022
############# TrOCR Finetuning #############

Description:

Example source code to finetune and test TrOCR models for handwritten text recognition

Content:

- train.py: training script for finetuning pre-trained models
- fun.py : models that has all processing , load data and configuratiions for TrOCR
- training-data-ex: example dataset (image segments and the corresponding text) for finetuning.
- models/: contians list of a finetuned models on the example dataset (including Arany200 )
- script/test: script for testing

##How to use it?

1. unzip the image files

$ cd training-data-ex
$ unzip imgs.zip

2. finetune pre-trained model
(pass 4 arguments text, coressponding image ,encoder and decoder )
& python3 train.py  training-data-ex/lines.txt training-data-ex/img/ google/vit-base-patch16-384 bert-base-uncased

3. test the model

The image files must be chunks of text pages presenting only one sentence

Processing a single image:

$ python3 script/test.py models/Arany200 test_examples/test.jpg

Dump processing & writing output to a text file:

$ python3 script/test.py -d models/Arany200 test_examples -o output.txt
 
