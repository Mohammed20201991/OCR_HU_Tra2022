# TrOCR_HU_2022
############# **TrOCR Finetuning** #############

Description:

Example source code to finetune and test TrOCR models for handwritten text recognition

Content:

- train.py: training script for finetuning pre-trained models
- fun.py : models that has all processing , load data and configuratiions for TrOCR
- training-data-ex: example dataset (image segments and the corresponding text) for finetuning.
- models/: contians list of a finetuned models on the example dataset (including Arany200 )
- script/test: script for testing

**How to use it?**

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
<h3> illusturation on how TrOCR Archtecture look like  </h3>

![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/TrOCR-finetune/images/illustration%20for%20Hungarain%20Transformer.PNG)

 <!DOCTYPE html>
<html>
<head>

</head>
<body>

<h1>Transformers Models Table</h1>

<table id="customers">
  <tr>
    <th>Rank</th>
    <th>Model name</th>
    <th>Encoder Transformer</th>
    <th>Decoder Transformers</th>
  </tr>
  <tr>
    <td>1</>
    <td>TrOCRHu𝐿𝑎𝑟𝑔𝑒</td>
    <td>ViT</td>
    <td>CLM</td>
  </tr>
  <tr>
    <td>2</td>
    <td>TrOCRHu𝑝𝑟𝑖𝑛𝑡𝑒𝑑</td>
    <td>ViT</td>
    <td>CLM</td>
  </tr>
  <tr>
    <td>3</td>
    <td>HuMu𝐿𝑎𝑟𝑔𝑒𝑆𝑡𝑎𝑔𝑒1 </td>
    <td>ViT</td>
    <td>CLM</td>
  </tr>
  <tr>
    <td>4</td>
    <td>TrOCR-base-ver021-aicup-ft</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>5</td>
    <td>BeiT_CLM</td>
    <td>BeiT</td>
    <td>CLM</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Swin_CLM</td>
    <td>Swin</td>
    <td>CLM</td>
  </tr>
  <tr>
    <td>7</td>
    <td>DeiT_CLM</td>
    <td>DeiT</td>
    <td>CLM</td>
  </tr>
  <tr>
    <td>8</td>
    <td>SwinHuBert</td>
    <td>Swin</td>
    <td>HuBert</td>
  </tr>
  <tr>
    <td>9</td>
    <td>SwinBart</td>
    <td>Swin</td>
    <td>Bart</td>
  </tr>
  <tr>
    <td>10</td>
    <td>ViTHuBert</td>
    <td>ViT</td>
    <td>HuBert</td>
  </tr>
 
 <tr>
    <td>11</td>
    <td>Vitdistilbert</td>
    <td>ViT</td>
    <td>Distilbert</td>
  </tr>
 <tr>
    <td>12</td>
    <td>hubert-m-wiki-s128</td>
    <td></td>
    <td></td>
  </tr>
 <tr>
    <td>13</td>
    <td>hubert-medium-wiki</td>
    <td>ViT</td>
    <td>Hubert</td>
  </tr>
 <tr>
    <td>14</td>
    <td>BeitHuBert</td>
    <td>BeiT</td>
    <td>HuBert</td>
  </tr>
 
 <tr>
    <td>15</td>
    <td>DeitHubert</td>
    <td>DeiT</td>
    <td>HuBert</td>
  </tr>
 <tr>
    <td>16</td>
    <td>ViTgpt2</td>
    <td>ViT</td>
    <td>gpt2</td>
  </tr>
 
 <tr>
    <td>17</td>
    <td>ViTgpt2smallhu poem</td>
    <td>ViT</td>
    <td>gpt2 on hu poem</td>
  </tr>
 
 <tr>
    <td>18</td>
    <td>ViTRoberta</td>
    <td>ViT</td>
    <td>XLRberta</td>
  </tr>
 
 <tr>
    <td>19</td>
    <td>ViTmGPT</td>
    <td>ViT</td>
    <td>mGPT</td>
  </tr>
 
 
</table>

</body>
</html>



