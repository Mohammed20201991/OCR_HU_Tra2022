# TrOCR_HU_2022
 **TrOCR Finetuning** 
<!DOCTYPE html>
<html>
<head>

</head>
<body>

Important to note this baseline code provided by DHLab Researcher (<a href ="https://github.com/Mohammed20201991/OCR_HU_Tra2022/commit/8540285b77cde44169cedc8cb5e49fe807fde683">train.py</a> , <a href ="https://github.com/Mohammed20201991/OCR_HU_Tra2022/commit/c6f580c4268ddb846f2a48415044bb2e5ada755d">test.py</a>)which depend on official implementation for TrOCR <a href = "https://github.com/microsoft/unilm/tree/master/trocr">this repo</a>

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

 

<h1 id ="Transformers Models Table">Transformers Models Table</h1>

<table style="width:100%">
  <caption>List of Encoder Transformers </caption>
<tr>
    <th>Model Name</th>
    <th>huggingface link</th>
    <th>Reference</th>
  </tr>
  <tr>
    <td id ="vit">ViT</td>
    <td> <a href ="https://huggingface.co/google/vit-base-patch16-384">vit-base-patch16-384</a> </td>
    <td> <a href="https://arxiv.org/abs/2010.11929">https://arxiv.org/abs/2010.11929</a> </td>
  </tr>
  
  <tr>
    <td id="deit">DeiT</td>
    <td> <a href ="https://huggingface.co/facebook/deit-base-distilled-patch16-384">deit-base-distilled-patch16-384</a> </td>
    <td> <a href="http://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf">touvron21a.pdf</a> </td>
  </tr>
  
  <tr>
    <td id="beit">BeiT</td>
    <td> <a href ="https://huggingface.co/microsoft/beit-base-patch16-384">beit-base-patch16-384</a> </td>
    <td> <a href="https://arxiv.org/abs/2106.08254">https://arxiv.org/abs/2106.08254</a> </td>
  </tr>
  <tr>
    <td id="swin">Swin</td>
    <td> <a href ="https://huggingface.co/microsoft/swin-base-patch4-window12-384">swin-base-patch4-window12-384</a> </td>
    <td> <a href="https://arxiv.org/abs/2103.14030">https://arxiv.org/abs/2103.14030</a> </td>
  </tr>
  
</table>

<hr>
<table style="width:100%">
  <caption>List of Decoder Transformers </caption>
<tr>
    <th>Model Name</th>
    <th>huggingface link</th>
    <th>Reference</th>
  </tr>
  <tr>
    <td id="bert">Bert</td>
    <td> <a href ="https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.">bert-base-uncased</a> </td>
    <td> <a href="https://arxiv.org/abs/1810.04805">https://arxiv.org/abs/1810.04805</a> </td>
  </tr>
  
  <tr>
    <td id="hubert">huBERT</td>
    <td> <a href ="https://huggingface.co/SZTAKI-HLT/hubert-base-cc">hubert-base-cc</a> </td>
    <td> <a href="https://hlt.bme.hu/en/publ/nemeskey_2020">nemeskey_2020</a> </td>
  </tr>
  
  <tr>
    <td id="gpt-2">GPT-2</td>
    <td> <a href ="https://huggingface.co/gpt2?text=Once+upon+a+time%2C">gpt2</a> </td>
    <td> <a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">language_models_are_unsupervised_multitask_learners</a> </td>
  </tr>
  <tr>
    <td id ="roberta">Roberta</td>
    <td> <a href ="https://huggingface.co/xlm-roberta-base">xlm-roberta-base</a> </td>
    <td> <a href="https://arxiv.org/abs/1911.02116">https://arxiv.org/abs/1911.02116</a> </td>
  </tr>
  
  <tr>
    <td id="distilledbert">DistilledBert</td>
    <td> <a href ="https://huggingface.co/facebook/nllb-200-distilled-600M">nllb-200-distilled-600M</a> </td>
    <td> <a href="https://arxiv.org/abs/2207.04672">https://arxiv.org/abs/2207.04672</a> </td>
  </tr>
  
  <tr>
    <td id="mgpt">mGPT</td>
    <td> <a href ="https://huggingface.co/sberbank-ai/mGPT?text=Once+upon+a+time%2C">mGPT</a> </td>
    <td> <a href="https://arxiv.org/abs/2204.07580">https://arxiv.org/abs/2204.07580</a> </td>
  </tr>
  <tr>
    <td id="bart">Bart</td>
    <td> <a href ="https://huggingface.co/facebook/bart-large">bart-large</a> </td>
    <td> <a href= ""https://arxiv.org/abs/1910.13461>https://arxiv.org/abs/1910.13461</a> </td>
  </tr>
</table>


<table id="customers">
  <tr>
    <th>Rank</th>
    <th>Model name</th>
    <th>Encoder Transformer</th>
    <th>Decoder Transformers</th>
    <th>CER %</th>
    <th>WER %</th>
  </tr>
  <tr>
    <td>1</>
    <td>TrOCRHu𝐿𝑎𝑟𝑔𝑒</td>
    <td > <a href="#vit"> ViT </a></td>
    <td><a href="#roberta"> CLM </a> </td>
    <td>4.447</td>
    <td>19.806</td>

  </tr>
  <tr>
    <td>2</td>
    <td>TrOCRHu𝑝𝑟𝑖𝑛𝑡𝑒𝑑</td>
    <td><a href="#vit"> ViT</a></td>
    <td><a href="#roberta"> CLM </a></td>
    <td>6.0731</td>
    <td>24.603</td>
  </tr>
  <tr>
    <td>3</td>
    <td>HuMu𝐿𝑎𝑟𝑔𝑒𝑆𝑡𝑎𝑔𝑒1 </td>
    <td><a href="#vit">ViT </a></td>
    <td><a href="#roberta"> CLM </a></td>
    <td>7.115</td>
    <td>26.570</td>
  </tr>
  <tr>
    <td>4</td>
    <td>TrOCR-base-ver021-aicup-ft</td>
    <td> <a href="https://huggingface.co/ycchen/TrOCR-base-ver021-aicup-ft"> link-hugf</a></td>
    <td><a href="#roberta"> CLM </a></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>5</td>
    <td> BeiT_CLM </td>
    <td><a href="#beit"> BeiT </a></td>
    <td><a href="#roberta"> CLM </a></td>
    <td>4.684</td>
    <td>20.676</td>
  </tr>
  <tr>
    <td>6</td>
    <td> Swin_CLM </td>
    <td> <a href="#swin">Swin </a></td>
    <td> <a href="#roberta"> CLM </a></td>
    <td>4.876</td>
    <td>21.64</td>
  </tr>
  <tr>
    <td>7</td>
    <td>  DeiT_CLM  </td>
    <td> <a href="#deit"> DeiT </a></td>
    <td> <a href="#roberta"> CLM </a></td>
    <td>4.624</td>
    <td>20.869</td>
  </tr>
  <tr>
    <td>8</td>
    <td>SwinHuBert</td>
    <td><a href="#swin">Swin </a></td>
    <td><a href="#hubert"> HuBert </a></td>
    <td>77.364</td>
    <td>93.339</td>
  </tr>
  <tr>
    <td>9</td>
    <td>SwinBart</td>
    <td> <a href="#swin">Swin </a></td>
    <td> <a href="#bart"> Bart </a></td>
    <td>136.22</td>
    <td>100.48 </td>
  </tr>
  <tr>
    <td>10</td>
    <td>ViTHuBert</td>
    <td><a href="#vit">Vit </a></td>
    <td> <a href="#hubert"> HuBert </a></td>
    <td>77.977</td>
    <td>95.442</td>
  </tr>
 
 <tr>
    <td>11</td>
    <td>Vitdistilbert</td>
    <td><a href="#vit">ViT </a></td>
    <td> <a href="#distilledbert"> Distilbert </a> </td>
    <td>80.437</td>
    <td>95.877</td>
  </tr>
 <tr>
    <td>12</td>
    <td>hubert-m-wiki-s128</td>
    <td><a href="https://huggingface.co/SzegedAI/hubert-medium-wiki-seq128"> hubert-medium-wiki-seq128</a></td>
    <td></td>
    <td>80.691</td>
    <td>97.721</td>
  </tr>
 <tr>
    <td>13</td>
    <td>hubert-medium-wiki</td>
    <td><a href="https://huggingface.co/SzegedAI/hubert-medium-wiki-seq128">ViT </a></td>
    <td> <a href="#hubert"> Hubert</a> </td>
    <td>82.194</td>
    <td>99.561</td>
  </tr>
 <tr>
    <td>14</td>
    <td>BeitHuBert</td>
    <td><a href="#beit"> BeiT </a></td>
    <td><a href="#hubert"> HuBert </a></td>
    <td>98.146</td>
    <td>96.319</td>
  </tr>
 
 <tr>
    <td>15</td>
    <td>DeitHubert</td>
    <td> <a href="#deit"> DeiT </a></td>
    <td> <a href= "#hubert"> HuBert </a></td>
    <td>80.95</td>
    <td>104</td>
  </tr>
 <tr>
    <td>16</td>
    <td>ViTgpt2</td>
    <td> <a href="#vit">ViT </a></td>
    <td> <a href="#gpt-2"> gpt2</a></td>
    <td></td>
    <td></td>
  </tr>
 
 <tr>
    <td>17</td>
    <td>ViTgpt2smallhu poem</td>
    <td><a href="#vit">ViT </a></td>
    <td><a href="#roberta"> gpt2 on hu poem</a></td>
    <td>362</td>
    <td>281</td>
  </tr>
 
 <tr>
    <td>18</td>
    <td>ViTRoberta</td>
    <td><a href="#vit">ViT </a></td>
    <td><a href="#roberta"> XLRberta </a></td>
    <td>198.75</td>
    <td>405.02</td>
  </tr>
 
 <tr>
    <td>19</td>
    <td>ViTmGPT</td>
    <td> <a href="#vit">ViT </a></td>
    <td> <a href="#mgpt">mGPT </a></td>
    <td>163</td>
    <td>787</td>
  </tr>
 
</table>

</body>
</html>

<hr>
<h2 id ="DataSets"> DataSets </h2>
The baseline models are trained with a proprietary dataset. The dataset is private data that contains images (in jpg format) these images have been containing segmented by lines and annotated with corresponding text in the text file. The annotations contain the image name,
the status whether is it ok or not, and many other meta parameters(they are not important in our task), and the last will contains the text for this image (features) separating each word by(|)characters, the (+) sign used to concatenate next line with current sentence. As Figure below .

<p>sampling image:</p>
<img src="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/TrOCR-finetune/images/RALK987_1865_817_121_001-031.jpg" alt="alt" border-radius: 8px>
 <h4 style="font-size:50px;"> And the corresponding text:tott űlése határozata folytán</h4> 

## <p id ="Inference"> An Inference Example </p>
Please see detials in [Inference_with_TrOCR_+_Gradio_demo.ipynb](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/Inference_with_TrOCR_%2B_Gradio_demo.ipynb).

## <p id = "Fine-tuning" >Fine-tuning  </p>
Fine-tuning TrOCR on HuDataset
[Fine_tune_TrOCR_Handwriting_using_Seq2SeqTrainer](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/Fine_tune_TrOCR_Handwriting_using_Seq2SeqTrainer.ipynb)

## <p id ="Evaluation">Evaluation </p>
Evaluation can be found on this jupyternotebook [Evaluating_TrOCR_base_handwritten_on_the_huData_test_set](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/Evaluating_TrOCR_base_handwritten_on_the_huData_test_set.ipynb)

## <p id="Demo"> Demo :</p>

</html>
</body>