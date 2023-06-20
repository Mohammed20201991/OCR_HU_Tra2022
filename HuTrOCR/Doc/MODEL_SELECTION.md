## Task Model Selection was fine-tuned on (DH-Lab) only the second stage is used here.
<h1 id ="Transformers Models Table">Transformers Models Table</h1>

<table style="width:100%">
<caption>List of (vision)Encoder Transformers Models </caption>
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
  <caption>List of Decoder Transformers Languge Models(LMs)</caption>
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
    <td>53.890</td>
    <td>92.857</td>
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
    <td>Swin+huBert</td>
    <td><a href="#swin">Swin </a></td>
    <td><a href="#hubert"> HuBert </a></td>
    <td>77.364</td>
    <td>93.339</td>
  </tr>
  <tr>
    <td>9</td>
    <td>Swin+Bart</td>
    <td> <a href="#swin">Swin </a></td>
    <td> <a href="#bart"> Bart </a></td>
    <td>136.22</td>
    <td>100.48 </td>
  </tr>
  <tr>
    <td>10</td>
    <td>ViT+huBert</td>
    <td><a href="#vit">Vit </a></td>
    <td> <a href="#hubert"> HuBert </a></td>
    <td>77.977</td>
    <td>95.442</td>
  </tr>
 
 <tr>
    <td>11(i)</td>
    <td>Vit+distilbert</td>
    <td><a href="#vit">ViT </a></td>
    <td> <a href="#distilledbert"> m-Distilbert </a> </td>
    <td>80.437</td>
    <td>95.877</td>
 </tr>
 
 <tr>
    <td>11(ii)</td>
    <td>Vit+distilbert</td>
    <td><a href="#vit">ViT </a></td>
    <td> <a href="#distilledbert"> m-Distilbert </a> </td>
    <td>79.591</td>
    <td>95.964</td>
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
    <td> <a href="#hubert">+hubert</a> </td>
    <td>82.194</td>
    <td>99.561</td>
  </tr>
 <tr>
    <td>14</td>
    <td>Beit+huBert</td>
    <td><a href="#beit"> BeiT </a></td>
    <td><a href="#hubert"> HuBert </a></td>
    <td>98.146</td>
    <td>96.319</td>
  </tr>
 
 <tr>
    <td>15</td>
    <td>Deit+hubert</td>
    <td> <a href="#deit"> DeiT </a></td>
    <td> <a href= "#hubert"> HuBert </a></td>
    <td>80.95</td>
    <td>104</td>
  </tr>
 <tr>
    <td>16</td>
    <td>ViT+GPT2</td>
    <td> <a href="#vit">ViT </a></td>
    <td> <a href="#gpt-2"> GPT-2</a></td>
    <td>362</td>
    <td>281 </td>
  </tr>
 
 <tr>
    <td>17</td>
    <td>ViT+gpt2smallhu poem</td>
    <td><a href="#vit">ViT </a></td>
    <td><a href="#roberta"> gpt2 on hu poem</a></td>
    <td>362</td>
    <td>281</td>
  </tr>
 
 <tr>
    <td>18</td>
    <td>ViT+Roberta</td>
    <td><a href="#vit">ViT </a></td>
    <td><a href="#roberta"> XLRoberta </a></td>
    <td>198.75</td>
    <td>405.02</td>
  </tr>
 
 <tr>
    <td>19</td>
    <td>ViT+mGPT</td>
    <td> <a href="#vit">ViT </a></td>
    <td> <a href="#mgpt">mGPT </a></td>
    <td>163</td>
    <td>787</td>
  </tr>
 <tr>
    <td></>
    <td>Vit+Bert</td>
    <td > <a href="#vit"> ViT </a></td>
    <td><a href="#bert"> Bert </a> </td>
    <td>76.780</td>
    <td>96.757</td>
  </tr>
  
   <tr>
    <td></>
    <td>Deit+Bert</td>
    <td > <a href="#deit"> Deit </a></td>
    <td><a href="#bert"> Bert </a> </td>
    <td>98.266</td>
    <td>99.439</td>
  </tr>
  
  <tr>
    <td></>
    <td>Deit+xlm-Roberta-large</td>
    <td > <a href="#deit"> Deit </a></td>
    <td><a href="#roberta"> xlm-Roberta-large </a> </td>
    <td>917.29</td>
    <td>1739.551</td>
  </tr>
</table>

## Fine-tuning All TrOCR Baseline Models on IAM Data (line levels)
### 1- Evaluation Results: 
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/all_TrOCR_Models.PNG)

### 2- Test Results:
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/all_TrOCR_Models_test_results.PNG)
## Fine-tuning Models on IAM Data (Sentence levels)
### Evaluation Results: 
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/IAM_Sentence_level.PNG)

## Fine-tuning Models on SROIE Dataset:
### Evaluation Results: 
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/fine-tuning%20combinations%20of%20models%20on%20SROIE.PNG)

## Mixed IAM + SROIE 
### Evaluation Results: 
![plot](https://github.com/Mohammed20201991/OCR_HU_Tra2022/blob/main/HuTrOCR/images/Mixed_IAM_SROIE.PNG)
</body>
</html>
