# TrOCR_HU_2022
 **TrOCR Finetuning** 
<!DOCTYPE html>
<html>
<head>

</head>
<body>

official implementation for TrOCR <a href = "https://github.com/microsoft/unilm/tree/master/trocr">this repo</a>

<ul dir="auto">
<li><a href="#Description">Description</a></li>
<li><a href="#How to use it?">How to use it?</a></li>
<li><a href="#illusturation">illusturation</a></li>
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

<h3 id="illusturation"> illusturation on how TrOCR architecture look like  </h3>

![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/illustration%20for%20%20%20Hungarain%20Transformer.PNG)


<hr>
<h2 id ="DataSets"> DataSets </h2>
The baseline models are trained with a proprietary dataset. The dataset is private data that contains images (in jpg format) these images have been containing segmented by lines and annotated with corresponding text in the text file. The annotations contain the image name,
the status whether is it ok or not, and many other meta parameters(they are not important in our task), and the last will contains the text for this image (features) separating each word by(|)characters, the (+) sign used to concatenate next line with current sentence. As Figure below .

<p>sampling image:</p>
<img src="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/RALK987_1865_817_121_001-031.jpg" alt="alt" border-radius: 8px>
 <h4 style="font-size:50px;"> And the corresponding text:tott űlése határozata folytán</h4> 

## <p id ="Inference"> An Inference Example </p>
Please see detials in [Inference_with_TrOCR_+_Gradio_demo.ipynb](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/Inference_with_TrOCR_%2B_Gradio_demo.ipynb).

## <p id = "Fine-tuning" >Fine-tuning  </p>
Fine-tuning TrOCR on HuDataset
[Fine_tune_TrOCR_Handwriting_using_Seq2SeqTrainer](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/Fine_tune_TrOCR_Handwriting_using_Seq2SeqTrainer.ipynb)

## <p id ="Evaluation">Evaluation </p>
Evaluation can be found on this jupyternotebook [Evaluating_TrOCR_base_handwritten_on_the_huData_test_set](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/Evaluating_TrOCR_base_handwritten_on_the_huData_test_set.ipynb)

## <p id="Demo"> Demo :</p>


## <p id="IAM"> IAM :</p>
Next step is to initilize the Transformers models with [IAM] dataset(https://fki.tic.heia-fr.ch/) that hase forms and this forms already segmented by (Words,Lines,Sentences and forms) where it converted the handwritten and annotated with ASIIC code .
I process the raw-data to convert it to TrOCR format.
<li> <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/IAM/IAM_words_processing.ipynb">[script]</a> for processing words(Text,Image)</li>
<li> <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/IAM/IAM_Sentence_Processing.ipynb">[script]</a> for processing sentences (Text,Iamge)</li>
<br>
<table>
<tr>
    <th>Model Name</th>
    <th>CER</th>
    <th>WER</th>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td> </td>
    <td></td>
  </tr>
</table>
</html>
</body>
