# Data augmentation in efficient ways
<!DOCTYPE html>
<html>
<head>

</head>
<body>

<ul dir="auto">
<li><a href="#Description">Description</a></li>
<li><a href="#How to use it?">How to use it?</a></li>
<li><a href="#illusturation">illusturation</a></li>
<li><a href="#Transformers Models Table">Transformers Models Table</a></li>
<li><a href="#DataSets">DataSets</a></li>
<li><a href="#Fine-tuning">Fine-tuning</a></li>
<li><a href="#Inference">Inference</a></li>
<li><a href="#Evaluation">Evaluation</a></li>
<li><a href="#Demo">Demo</a></li>
<li><a href="#IAM">IAM</a></li>
</ul>

**<p id="Description">Description:</p>**

Example source code to finetune and test TrOCR models for handwritten text recognition

**Content:**

- train.py: training script for finetuning pre-trained models
- fun.py : models that has all processing , load data and configuratiions for TrOCR
- training-data-ex: example dataset (image segments and the corresponding text) for finetuning.
- models/: contians list of a finetuned models on the example dataset (including Arany200 )
- script/test: script for testing

**<p id= "How to use it?"> How to use it?</p>**

<h4> Step 0: download  </h4>

~~~bash  
!git clone https://github.com/Mohammed20201991/OCR_HU_Tra2022.git
!cd OCR_HU_Tra2022/TrOCR-finetune/
~~~

<h4> Step 1: unzip the image files </h4>

~~~bash  
!cd training-data-ex
!unzip img.zip
~~~

<h4> Step 2: finetune pre-trained model</h4>

(pass 4 arguments text (segmented by lines), coressponding images ,encoder and decoder )
~~~bash
!cd ..
python3 train.py  training-data-ex/lines.txt training-data-ex/img/ google/vit-base-patch16-384 bert-base-uncased
~~~

<h4> Step 3: test the model</h4>

The image files must be chunks of text pages presenting only one sentence

Processing a single image:

~~~bash
python3 script/test.py models/Arany200 test_examples/test.jpg
~~~

Dump processing & writing output to a text file:

~~~bash 
python3 script/test.py -d models/Arany200 test_examples -o output.txt
~~~

<h3> List of Encoder Transformers used </h3>

<h3 id="illusturation"> illusturation on how TrOCR architecture look like  </h3>

![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/TrOCR-finetune/images/illustration%20for%20%20%20Hungarain%20Transformer.PNG)

</html>
</body>