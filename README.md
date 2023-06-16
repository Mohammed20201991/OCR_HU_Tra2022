# TrOCR_HU_2022
<ul dir="auto">
<li><a href="#Introduction">Introduction</a></li>
<li><a href="#How_to_use_it?">How do I use it?</a></li>
<li><a href="#illusturation">illusturation</a></li>
<li><a href="#DataSets">DataSets</a></li>
<li><a href="#Fine-tuning">Fine-tuning</a></li>
<li><a href="#Inference">Inference</a></li>
<li><a href="#Evaluation">Evaluation</a></li>
<li><a href="#Demo">Demo</a></li>
<li><a href="#"></a></li>
</ul>

**<p id="Introduction">Introduction:</p>** 
The project has been done for **Enhancing State-of-the-Art Language Models for Hungarian Handwritten Text Recognition**. By leveraging the Transformer architecture for both visual models (CV) for image understanding and language models (LM) for wordpiece-level text generation and its end-to-end architecture. The baseline model is TrOCR.

**<p id="illusturation"> illusturation </p>**
### Sample for Leavearging vision(CV) with Languge Model(LMs) (self-dited) source <a href="https://arxiv.org/abs/1810.04805">BERT</a> & <a href ="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">GPT-2</a> Models. 
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/illustration%20for%20%20%20Hungarain%20Transformer.PNG)
### OCR Pipeline(Self-Made)
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/OCR_pipline.png)
<!-- The TrOCR is currently implemented with the Fairseq library. We hope to convert the models to the Huggingface format later. -->
The OCR models are provided in the Huggingface format.[[Models](https://huggingface.co/AlhitawiMohammed22)]

<hr>
<h2 id="DataSets"> DataSets </h2>
The data we used contains Humman data that was collected and synthetic data that we generated. To see how the generation step has been done visit<a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HuTRDG"> HuTRDG </a>. The dataset has been split into 3 sets: train (80%), validation (10%) to optimize the used hyper-parameters, and test (10%) to see how much the trained and finetuned models generalize. 

## 1- Humman data (DH-Lab)
The baseline models are trained with a proprietary dataset. The dataset is private data that contains images (in jpg format). These images have been segmented by lines and annotated with the corresponding text in the text file. The annotations contain the image name,
the status whether is it ok or not, and many other meta parameters(they are not important in our task), and the last will contains the text for this image (features) separating each word by (|) characters, the (+) sign is used to concatenate the next line with the current sentence. As shown below.

<p>sampling image:</p>
<img src="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/RALK987_1865_817_121_001-031.jpg" alt="alt" border-radius: 8px>
<h4 style="font-size:50px;"> And the corresponding text:tott űlése határozata folytán</h4> 

## 2- Human data Augmentation in an Efficient way (DH-Lab Aug.)
You can find the script that represents the toolkit for data augmentation <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HTR_Aug">HTR_Aug. </a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YA-VV-RZYf9MOV8a6b-kXtQyPdnOkG57?usp=sharing)
## 3- Synthetic Data
This dataset has been generated synthetically by developing an existing toolkit for Handwritten Text Data Generation(HTDG) <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HuTRDG">HuTRDG</a>

## The methodology used follows the figure below: 
The models have been developed in two stages: the first is pre-training on synthetic Hungarian data, and the second is fine human data from DH-Lab on the resulting checkpoints from stage one.
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/Methodolgy_used.PNG)

## The Evaluation Metrics
### 1- Character Error Rate(CER) <a href="https://huggingface.co/spaces/AlhitawiMohammed22/CER_Hu-Evaluation-Metrics">demo</a>
### 2- Word Error Rate(WER) <a href="https://huggingface.co/spaces/AlhitawiMohammed22/WER_Hu-Evaluation-Metrics">demo</a>

<h2 id="How_to_use_it?"> How do I use it? </h2>

## Installation
### Make the new virtual environment 
```
!pip install virtualenv
!python3.8 -m venv env
```
For Activating Virtual Environment `source env/bin/activate`
~~~bash
git clone https://github.com/Mohammed20201991/OCR_HU_Tra2022.git
cd OCR_HU_Tra2022
cd HuTrOCR
pip install -r requirements.txt
~~~
## Task: (only one-stage) Model selection experimental could be found [MODEL_SELECTION.md]() file 
## Task: (only one-stage) Fine-tuning all TrOCR baseline models on the DH-Lab dataset results could be found at [Fine-tuning all TrOCR](https://github.com/Mohammed20201991/Draft/blob/main/Fine_tune_all_TrOCR.md)
## Task: Two stages(Pre-training & Fine-tuning) Words-level results could be found at [WORDS_LEVEL.md](https://github.com/Mohammed20201991/Draft/blob/main/WORDS_LEVEL.md) 
## Task: Two stages(Pre-training & Fine-tuning) lines_hu_v4 results could be found at [LINES_HU_V4.md](https://github.com/Mohammed20201991/Draft/blob/main/LINES_HU_V4.md) 
## Task: Two stages (pre-training and fine-tuning) lines_hu_v2_1 results: 
### 1- Pre-training Test Results:
<table>
<tr>
    <th>Model Name</th>
    <th>CER(%)</th>
    <th>WER(%)</th>
</tr>
  <tr>
    <td>TrOCR-large-handwritten</td>
    <td>1.792 </td>
    <td>4.944</td>
  </tr>
  <tr>
    <td>Deit+Roberta-base</td>
    <td> 2.327</td>
    <td>6.2332</td>
  </tr>

 <tr>
    <td>Deit-PULI-BERT</td>
    <td> 2.129 </td>
    <td>4.691</td>
  </tr>

  <tr>
    <td></td>
    <td> </td>
    <td></td>
  </tr> 
</table>

### 2-Fine-tuning Test Results: 
<table>
<tr>
    <th>Model Name</th>
    <th>Data</th>
    <th>CER(%)</th>
    <th>WER(%)</th>
</tr>
  <tr>
    <td>TrOCR-large-handwritten (only fine-tune stage one [English weights])</td>
    <td>DH-Lab</td>
    <td> 5.764</td>
    <td>23.297</td>
  </tr>

 <tr>
    <td>TrOCR-large-handwritten (only fine-tune stage one [English weights])</td>
    <td>DH-Lab(Augumented)</td>
    <td> 6.473</td>
    <td> 22.211 </td>
  </tr>

  <tr>
    <td>TrOCR-large-handwritten </td>
    <td>DH-Lab</td>
    <td> **3.681** </td>
    <td> **16.189** </td>
  </tr>

  <tr>
    <td>TrOCR-large-handwritten </td>
    <td>DH-Lab(Augumented)</td>
    <td> 5.221 </td>
    <td> 18.46 </td>
  </tr>
  
  <tr>
    <td>Deit+Roberta-base</td>
    <td>DH-Lab</td>
    <td> 8.374 </td>
    <td>29.121</td>
  </tr>

<tr>
    <td>Deit+Roberta-base</td>
    <td>DH-Lab(Augumentation)</td>
    <td> 4.889 </td>
    <td> 18.558 </td>
  </tr>
  
 <tr>
    <td>Deit-PULI-Bert</td>
    <td>DH-Lab</td>
    <td> 5.381  </td>
    <td> 16.091 </td>
  </tr>

  <tr>
    <td>Deit-PULI-Bert</td>
    <td>DH-Lab(Augumentation)</td>
    <td> 6.123 </td>
    <td> 16.357 </td>
  </tr> 
</table>

## TrOCR large-handwritten 
### Re-Pretraining on lines_hu_v2_1
~~~bash
python3 train.py pass/list/of/args
~~~

### Evaluation on lines_hu_v2_1
~~~bash
python3 test.py pass/arg(s)/
~~~


## An Inference Example
Please see details in [pic_inference.py](https://github.com/microsoft/unilm/blob/master/trocr/pic_inference.py).

**<p id="Demo"> Demo </p>** 

## Citation


## License
- The source code is free.
- The dataset is not allowed to use as it is private data and was given only for academic research.

### Contact Information
Email: Mohamedabid09@gmail.com

For other communications related to TrOCR, please contact Lei Cui (`lecu@microsoft.com`), and Furu Wei (`fuwei@microsoft.com`).

## Acknowledgement 
- f
- g
- g
- g 


## References
- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282), Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei, ```AAAI 2023```.
- The Official implementation for TrOCR <a href = "https://github.com/microsoft/unilm/tree/master/trocr">this repo</a>
- The Human dataset I am using is a private dataset provided by the Hungarian Digital Heritage Lab <a href="https://dh-lab.hu/en/kezdolap-english/">(DH-Lab)</a>. Written by Auther(<a href="http://magyar-irodalom.elte.hu/sulinet/igyjo/setup/portrek/arany/arany.htm">János Arany</a>)
- <a href="http://acta.bibl.u-szeged.hu/78417/">Jönnek a nagyok! BERT-Large, GPT-2 és GPT-3 nyelvmodellek magyar nyelvre </a> or <a href="https://huggingface.co/NYTK/PULI-BERT-Large">PULI-BERT-Large</a>
- <a href="https://huggingface.co/facebook/deit-small-distilled-patch16-224">Deit</a> , <a href="https://arxiv.org/abs/2012.12877">Training data-efficient image transformers & distillation through attention</a>
- <a href="https://arxiv.org/abs/1907.11692">RoBERTa: A Robustly Optimized BERT Pretraining Approach
</a> , <a href="https://huggingface.co/roberta-base">Roberta-base</a>
