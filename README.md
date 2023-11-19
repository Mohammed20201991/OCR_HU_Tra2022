# TrOCR_HU_2022  <img src="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/Syndbad.PNG" width="150" height="75" /> 
<ul dir="auto">
<li><a href="#Introduction">Introduction</a></li>
<li><a href="#illusturation">illusturation</a></li>
<li><a href="#OCR_Pipeline">OCR Pipeline</a></li>
<li><a href="#DataSets">DataSets</a></li> 
<li><a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HTR_Aug"> Data Augmentation in Efficient way </a></li> 
<li><a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HuTRDG">Toolkit Handwritten Text Recognition Data Generator </a> </li> 
<li><a href="#Methodology">Methodology</a></li>
<li><a href="#EvaluationMetrics">Evaluation Metrics</a></li> 
<li><a href="#How_to_use_it?">How do I use it?</a>
<ol>
  <li> <a href="#TrOCR_large_handwritten">TrOCR large handwritten </a> </li>
    <ul>
      <li> <a href="#TrOCR_large_handwrittenPre-train"> Pre-train </a> </li>
      <li> <a href="#TrOCR_large_handwrittenFine-tune"> Fine-tune </a>  </li>
    </ul>
  <li> <a href="#LeaveargingDeitRoberta-base"> Leavearging Deit with Roberta-base </a> </li>
    <ul>
      <li> <a href="#LeaveargingDeitRoberta-basePre-train"> Pre-train </a> </li>
      <li> <a href="#LeaveargingDeitRoberta-baseFine-tune"> Fine-tune </a> </li>
    </ul>
  <li> <a href="#LeaveargingDeitPULI-BERT"> Leavearging Deit with PULI-BERT </a> </li>
    <ul>
      <li><a href="#LeaveargingDeitPULI-BERTPre-train"> Pre-train </a> </li>
      <li> <a href="#LeaveargingDeitPULI-BERTFine-tune"> Fine-tune </a>  </li>
    </ul>
</ol>
</li>
<li><a href="#Inference">Inference</a></li>
<li><a href="#Demo">Demo</a></li>
<li><a href="#Acknowledgement"> Acknowledgement </a></li>
<li><a href="https://github.com/Mohammed20201991/MFileSiplite">More processing scripts </li>
</ul>

**<p id="Introduction">Introduction:</p>** 
The project has been done for **Enhancing State-of-the-Art Language Models for Hungarian Handwritten Text Recognition**. By leveraging the Transformer architecture for both visual models (CV) for image understanding and language models (LM) for wordpiece-level text generation and its end-to-end architecture. The baseline model is TrOCR.

**<p id="illusturation"> illusturation </p>**
### Sample for Leavearging vision(CV) with Languge Model(LMs) (self-Edited) source <a href="https://arxiv.org/abs/2012.12877">Deit</a>, <a href="https://arxiv.org/abs/1810.04805">BERT</a> & <a href ="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">GPT-2</a> Models. 
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/illustration%20for%20%20%20Hungarain%20Transformer.PNG)
**<p id="OCR_Pipeline">OCR Pipeline (Self-Made)</p>**
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/OCR_pipline.png)
<!-- The TrOCR is currently implemented with the Fairseq library. We hope to convert the models to the Huggingface format later. -->
The OCR models are provided in the Huggingface format.[[Models](https://huggingface.co/AlhitawiMohammed22)]

<hr>
<h2 id="DataSets"> DataSets </h2>
The data we used contains Humman data that was collected and synthetic data that we generated. To see how the generation step has been done visit<a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HuTRDG"> HuTRDG </a>. The dataset has been split into 3 sets: train (80%), validation (10%) to optimize the used hyper-parameters, and test (10%) to see how much the trained and finetuned models generalize. 

## 1- Humman data (DH-Lab)
The baseline models are trained with a proprietary dataset. The dataset is private data that contains images (in jpg format). These images have been segmented by lines and annotated with the corresponding text in the text file. The annotations contain the image name,
the status, whether is okay or not, and many other meta parameters (they are not important in our task), and the last will contain the text for this image (features) separating each word by (|) characters, the (+) sign is used to concatenate the next line with the current sentence. As shown below.

<p>Sampling image:</p>
<img src="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/RALK987_1865_817_121_001-031.jpg" alt="alt" border-radius: 8px>
<h4 style="font-size:50px;"> And the corresponding text:tott űlése határozata folytán</h4> 

## 2- Human data Augmentation efficiently (DH-Lab Aug.)
You can find the script that represents the toolkit for data augmentation <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HTR_Aug">HTR_Aug. </a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YA-VV-RZYf9MOV8a6b-kXtQyPdnOkG57?usp=sharing)
## 3- Synthetic Data
This dataset has been generated synthetically by developing an existing toolkit for Handwritten Text Data Generation(HTDG) <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HuTRDG">HuTRDG</a>

<h2 id="Methodology">The methodology used follows the figure below: </h2>
The models have been developed in two stages: the first is pre-training on synthetic Hungarian data, and the second is human data from DH-Lab on the resulting checkpoints from stage one.

![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/Methodolgy_used.PNG)

<h2 id="EvaluationMetrics">The Evaluation Metrics </h2>
<ol>
  <li> Character Error Rate (CER) <a href="https://huggingface.co/spaces/AlhitawiMohammed22/CER_Hu-Evaluation-Metrics">demo</a> </li>
  <li> Word Error Rate (WER) <a href="https://huggingface.co/spaces/AlhitawiMohammed22/WER_Hu-Evaluation-Metrics">demo</a></li>
</ol>

<h2 id="How_to_use_it?"> How do I use it? </h2>

## Installation
### Make the new virtual environment 
```
pip install virtualenv
python3.8 -m venv env
```
For Activating Virtual Environment `source env/bin/activate`
~~~bash
git clone https://github.com/Mohammed20201991/OCR_HU_Tra2022.git
cd OCR_HU_Tra2022
cd HuTrOCR
pip install -r requirements.txt
~~~

<h3> Task: (only one-stage) Model selection experiments could be found in the <a href=""> MODEL_SELECTION.md  </a> file <h3>

<h3> Task: (only one-stage) <a href="https://github.com/Mohammed20201991/Draft/blob/main/Fine_tune_all_TrOCR.md"> Fine-tuning all TrOCR </a> baseline models on the DH-Lab dataset results could be found at </h3>
<h3> Task: Two stages (pre-training & Fine-tuning) Word-level results could be found at <a href="https://github.com/Mohammed20201991/Draft/blob/main/WORDS_LEVEL.md"> WORDS_LEVEL.md </a> </h3> 

<h3>Task: Two stages (pre-training and Fine-tuning) lines_hu_v4 results could be found at <a href="https://github.com/Mohammed20201991/Draft/blob/main/LINES_HU_V4.md"> LINES_HU_V4.md  </a> </h3> 

<h3>Task: Two stages (pre-training and fine-tuning) lines_hu_v2_1 results: </h3>
<h4> 1- Pre-training Test Results (First Stage): </h4>
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

<h4>2-Fine-tuning Test Results (Second Stage): </h4>
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
    <td> <u> 3.681<\u> </td>
    <td> <u>16.189 <\u> </td>
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

<h3 id="TrOCR_large_handwritten">TrOCR large-handwritten </h3>
<h4 id="TrOCR_large_handwrittenPre-train"> Re-Pretraining on Synthetic lines_hu_v2_1 dataset </h4>

`export CUDA_VISIBLE_DEVICES=3`
~~~bash
python3 train.py 
--text_path="../Data/lines_hu_v2_1/train.jsonl"
--images_path="../Data/lines_hu_v2_1/images/"
--epochs=25
--train_batch_size=100
--eval_batch_size=100
--logging_steps=500
--save_steps=5000
--eval_steps=5000
--learning_rate=5e-5
--gradient_checkpointing=True
--full_train=False
~~~
Setting `--full_train=False` because we are doing Re-Pre-training
### Evaluation on lines_hu_v2_1 (Test set)
i- If you want to use the models from <a href ="https://huggingface.co/">HuggingFace library<a>. For accessing private models <a href="https://huggingface.co/AlhitawiMohammed22">Reach our models upon request</a> type in your terminal `huggingface-cli login` Then copy and past the provided `token ="xxxx"`. 

ii- `export CUDA_VISIBLE_DEVICES=3`

iii- For testing use the command below or you can  [![Open it in  Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1gGply3tQEmQMqkhOovMLAOEx0pVBQMGt/view?usp=sharing)
~~~bash
python3 test.py 
--text_path="../Data/lines_hu_v2_1/test.jsonl"
--images_path="../Data/lines_hu_v2_1/images/"
--get_by_model_id=False
~~~


<h3 id="TrOCR_large_handwrittenFine-tune">Fine-tune the DH-Lab dataset on the pre-trained Synthetic checkpoint (lines_hu_v2_1)</h3>
  
`export CUDA_VISIBLE_DEVICES=3`

~~~bash
python3 train.py 
--epochs=25
--train_batch_size=8
--eval_batch_size=8
--logging_steps=500
--save_steps=1000
--eval_steps=500
--learning_rate=5e-5
--full_train=False
--processor_dir="/Models/TrOCR_large_handwritten/processor"
--ft_model_id="/Models/TrOCR_large_handwritten/checkpoint-xxxx"
--max_length=64
--working_dir="Models_ft/TrOCR_large_handwritten_ft/"
~~~
Setting `--full_train=False` because we are doing Fine-tuning 
### Evaluation on DH-Lab (Test set)
`export CUDA_VISIBLE_DEVICES=3`

~~~bash
python3 test.py 
--get_by_model_id=False
--load_model_from_checkpoint_dir="./Models_ft/TrOCR_large_handwritten_ft/checkpoint-xxxx"
--max_length=64
~~~

<h3 id="LeaveargingDeitRoberta-base">Leavearging Deit with PULI-BERT model </h3>
<h4 id="LeaveargingDeitRoberta-basePre-train">Pretraining on Synthetic lines_hu_v2_1 dataset, first-stage </h4>

`export CUDA_VISIBLE_DEVICES=3`
~~~bash
python3 train.py 
--text_path="../Data/lines_hu_v2_1/train.jsonl"
--images_path="../Data/lines_hu_v2_1/images/"
--epochs=25
--train_batch_size=24
--eval_batch_size=24
--logging_steps=100
--save_steps=5000
--eval_steps=5000
--learning_rate=4e-5
--leveraging=True
--max_length=96
--working_dir="Models/PULI-BERT_Deit/"
~~~

### Evaluation on lines_hu_v2_1 (Test set)
`export CUDA_VISIBLE_DEVICES=3`

~~~bash
python3 test.py 
--text_path="../Data/lines_hu_v2_1/test.jsonl"
--images_path="../Data/lines_hu_v2_1/images/"
--get_by_model_id=False
--load_model_from_checkpoint_dir="./Models/PULI-BERT_Deit"
--max_length=96
~~~

<h3 id="LeaveargingDeitRoberta-baseFine-tune">Fine-tune the DH-Lab dataset on the pre-trained Synthetic checkpoint (lines_hu_v2_1) second-stage </h3>

`export CUDA_VISIBLE_DEVICES=3`

~~~bash
python3 train.py 
--epochs=25
--train_batch_size=24
--eval_batch_size=24
--logging_steps=100
--save_steps=2000
--eval_steps=1000
--learning_rate=4e-5
--full_train=False
--processor_dir="/Models/PULI-BERT_Deit/processor"
--ft_model_id="/Models/PULI-BERT_Deit/checkpoint-xxxx"
--max_length=64
--working_dir="Models_ft/PULI-BERT_Deit_ft/"
~~~
Setting `--full_train=False` because we are doing Fine-tuning 
### Evaluation on DH-Lab(Test set)
`export CUDA_VISIBLE_DEVICES=3`

~~~bash
python3 test.py 
--get_by_model_id=False
--load_model_from_checkpoint_dir="./Models_ft/PULI-BERT_Deit_ft/checkpoint-xxxx"
--max_length=64
~~~

<h3 id="LeaveargingDeitPULI-BERT">Leavearging Deit with Roberta-base model </h3>
<h4 id="LeaveargingDeitPULI-BERTPre-train"> Pretraining on Synthetic lines_hu_v2_1 dataset, first-stage </h4>

`export CUDA_VISIBLE_DEVICES=3`
~~~bash
python3 train.py 
--text_path="../Data/lines_hu_v2_1/train.jsonl"
--images_path="../Data/lines_hu_v2_1/images/"
--epochs=25
--train_batch_size=32
--eval_batch_size=32
--logging_steps=100
--save_steps=5000
--eval_steps=5000
--learning_rate=4e-5
--nlp_model_dir="Roberta-base"
--leveraging=True
--max_length=96
--working_dir="Models/Roberta-base_Deit/"
~~~

### Evaluation on lines_hu_v2_1 (Test set)
`export CUDA_VISIBLE_DEVICES=3`

~~~bash
python3 test.py 
--text_path="../Data/lines_hu_v2_1/test.jsonl"
--images_path="../Data/lines_hu_v2_1/images/"
--get_by_model_id=False
--load_model_from_checkpoint_dir="./Models/Roberta-base_Deit"
--max_length=96
~~~

<h3 id="LeaveargingDeitPULI-BERTFine-tune">Fine-tune the DH-Lab dataset on the pre-trained Synthetic checkpoint (lines_hu_v2_1) second-stage</h3>

`export CUDA_VISIBLE_DEVICES=3`
~~~bash
python3 train.py 
--epochs=25
--train_batch_size=24
--eval_batch_size=24
--logging_steps=100
--save_steps=2000
--eval_steps=1000
--learning_rate=4e-5
--full_train=False
--processor_dir="/Models/Roberta-base_Deit/processor"
--ft_model_id="/Models/Roberta-base_Deit/checkpoint-xxxx"
--max_length=64
--working_dir="Models_ft/Roberta-base_Deit_ft/"
~~~
Setting `--full_train=False` because we are doing Fine-tuning 
### Evaluation on DH-Lab(Test set)
`export CUDA_VISIBLE_DEVICES=3`

~~~bash
python3 test.py 
--get_by_model_id=False
--load_model_from_checkpoint_dir="./Models_ft/PULI-BERT_Deit_ft/checkpoint-xxxx"
--max_length=64
~~~


<h3 id="Inference">An Inference Example </h3>

Please see details in [pic_inference.py](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/infrence.py).Or you can see [![Open it in  Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zeI9cK070UnMQkbZC-D1EHPJPml3ChZ0?usp=sharing)
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/Inference_samples.PNG)
<h3 id="Demo"> Demo </h3>

See <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/Image2text_OCR.ipynb">Image2Text Script</a> or [![Open I2T Pipeline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YzpWIEj0oscoP-1AVih7aospqOEO9J4f?usp=sharing)
For OCR_live_demo using Gradio, see: [![Open in  Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LoGv2cPoJrLZQcHT_yds_pcKqBznGXuW?usp=sharing)

Video could be found here : <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/Videos/OCR_HU_Tra2022_%20HTR%20Transformer%20for%20Hungarian%20Language%20-%2024%20June%202023.mp4"> <img src="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/free-video-icon-818-thumb.png" width="50" height="40" /> </a>

## License
- The source code is free.
- The dataset is not allowed to be used as it is private data and was given only for academic research.

### Contact Information
Email: Mohamedabid092@gmail.com

<h3 id="Acknowledgement"> Acknowledgement </h3>

- **Gyöngyössy Natabara Máté** , Email: natabara@inf.elte.hu My Supervisor during AI Project Labs and thesis  <a href="https://www.elte.hu/">Eötvös Loránd University </a>
- **Dr. János Botzheim**, Email: botzheim@inf.elte.hu         My Supervisor during AI Project Labs. <a href="https://www.elte.hu/">Eötvös Loránd University<a>
- **Szekrényes István and Nemeskey Dávid** Hungarian Digital Heritage Lab Researchers (<a href="https://elte-dh.hu/en/digital-cultural-heritage/">DH-Lab</a>) where they provide me the historical handwriting dataset benchmark of János Arany & valuable A100 8GPUS 80GB  

<h3 id ="Contribution"> Contribution </h3>
<ul>
  <li>The thesis includes results from months-long GPU-optimal runs, where the   most 
novel technologies, as intended, are utilized </li>
  <li> Addressing Double start token in many international models <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/JupLab/AddressingDoubleStartTokenIssue.ipynb"> see notebook</a></li>
  <li> Generate more than two million <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HuTRDG"> synthetic datasets </a> for the Hungarian language </li>
  <li> The results overcome the state-of-the-Art TrOCR Model for Hungarian handwriting recognition. </li>
  <li> Leveraging new state-of-the-art vision-language models in OCR architecture </li>
  <li> What else is left? for future work and it is open for contribution  </li>
  <ol>
  <li> Replace the GPT-2 Hungarian based in TrOCR architecture </li>
    
  <li> Use TrOCR model in Parallel with <a href="https://huggingface.co/NYTK/PULI-GPT-2"> PULI-GPT-2 </a> see the draft notebook <a href="https://colab.research.google.com/drive/1ZNxbJddY6BrRSO41aZxAYu3ad1NPUKv0?usp=sharing"> Parallel </a> 
  </li>

  <li> Generate more variations of synthetic data or collect more human-annotated data <a href="https://github.com/Mohammed20201991/OCR_HU_Tra2022/tree/main/HTR_Aug">HTR_Aug. </a>    </li>
</ol>
</ul>

<h3 id="References">References</h3>

- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282), Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei, ```AAAI 2023```.
- <a href="http://acta.bibl.u-szeged.hu/78417/">Jönnek a nagyok! BERT-Large, GPT-2 és GPT-3 nyelvmodellek magyar nyelvre </a> or <a href="https://huggingface.co/NYTK/PULI-BERT-Large">PULI-BERT-Large</a>
- <a href="https://huggingface.co/facebook/deit-small-distilled-patch16-224">Deit</a> , <a href="https://arxiv.org/abs/2012.12877">Training data-efficient image transformers & distillation through attention</a>
- <a href="https://arxiv.org/abs/1907.11692">RoBERTa: A Robustly Optimized BERT Pretraining Approach
</a> , <a href="https://huggingface.co/roberta-base">Roberta-base</a>
- The Official implementation for TrOCR <a href = "https://github.com/microsoft/unilm/tree/master/trocr">this repo</a>
- The Human dataset I am using is a private dataset provided by the Hungarian Digital Heritage Lab <a href="https://dh-lab.hu/en/kezdolap-english/">(DH-Lab)</a>. Written by Auther(<a href="http://magyar-irodalom.elte.hu/sulinet/igyjo/setup/portrek/arany/arany.htm">János Arany</a>)
- Tools used: <a href="https://www.python.org/"> Python </a>, <a href="https://huggingface.co/">Huggingface</a>, <a href="https://pytorch.org/">Pytroch </a>, <a href="https://code.visualstudio.com/"> VScode </a> <a href="https://www.tensorflow.org/tensorboard">Tensorbord</a>, and <a href="https://www.linux.org/"> Linux OS </a> <a href="https://ubuntu.com/"> Ubunto distribution  </a>
