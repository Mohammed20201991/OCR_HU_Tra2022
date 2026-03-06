## Encoder–Decoder Model Selection

We evaluated several **vision–language encoder–decoder combinations** using open-source pre-trained models from the **Hugging Face (HF) framework**. 
These models were fine-tuned on the **Hungarian DH-Lab dataset**, and performance was measured on the **evaluation set**.

Most models were initialized with **English pre-trained weights**, while some were initialized with **Hungarian-pretrained variants**.

### Training Configuration

- **Learning rate:** `2e-5`
- **Batch size:** `16`
- **Epochs:** `12`

### Observations

- **GPT-family decoders** did not converge effectively. This is mainly due to architectural differences between autoregressive GPT models and the vision encoder setup, which requires additional integration mechanisms.

- **BERT-family decoders** showed **slow but stable convergence**, making them suitable candidates for further experiments.

- Based on these results, **ViT + BERT** was selected as a promising configuration for subsequent experiments.

### Selected Architecture

The selected configuration follows a **TrOCR-style architecture**, where a **Vision Transformer (ViT)** acts as the encoder and a **text decoder** (e.g., BERT) generates the output sequence.

Table summrize this initial expaiermnt
<table>
<thead>
<tr>
<th>R</th>
<th>Transformers Arch.</th>
<th>Steps (K)</th>
<th>Train Loss</th>
<th>Val Loss</th>
<th>CER</th>
<th>WER</th>
</tr>
</thead>
<tbody>
<tr><td>1</td><td>Swin + HuBERT</td><td>4</td><td>1.7345</td><td>3.763</td><td>77.364</td><td><b>93.339</b></td></tr>
<tr><td>8</td><td>Swin + BART</td><td>4</td><td>4.4888</td><td>15.85</td><td>136.22</td><td>100.48</td></tr>
<tr><td>2</td><td>ViT + HuBERT</td><td>9</td><td>0.9146</td><td>4.682</td><td>77.977</td><td>95.442</td></tr>
<tr><td>0</td><td>ViT + BERT</td><td>8</td><td>1.468</td><td>2.884</td><td><b>76.780</b></td><td>96.757</td></tr>
<tr><td>3</td><td>ViT + DistilBERT</td><td>5</td><td>2.393</td><td>2.907</td><td>79.591</td><td>95.964</td></tr>
<tr><td>-</td><td>ViT + m-DistilledBERT</td><td>4</td><td>2.3462</td><td>2.901</td><td>80.437</td><td>95.877</td></tr>
<tr><td>4</td><td>ViT + HuBERT-m-wiki-seq128</td><td>2</td><td>2.959</td><td>3.502</td><td>80.691</td><td>97.721</td></tr>
<tr><td>6</td><td>ViT + HuBERT-m-wiki</td><td>2</td><td>3.046</td><td>3.54</td><td>82.194</td><td>99.561</td></tr>
<tr><td>7</td><td>BEiT + HuBERT</td><td>4</td><td>5.7333</td><td>6.003</td><td>98.146</td><td>96.319</td></tr>
<tr><td>5</td><td>DeiT + HuBERT</td><td>4</td><td>1.7329</td><td>6.357</td><td>80.95</td><td>104</td></tr>
<tr><td>11</td><td>ViT + GPT-2</td><td>8</td><td>1.9851</td><td>2.903</td><td>362</td><td>281</td></tr>
<tr><td>12</td><td>ViT + GPT-2 Small HU</td><td>1</td><td>3.448</td><td>4.105</td><td>592.69</td><td>643.18</td></tr>
<tr><td>10</td><td>ViT + RoBERTa</td><td>4</td><td>4.7962</td><td>12.37</td><td>198.75</td><td>405.02</td></tr>
<tr><td>9</td><td>ViT + mGPT</td><td>1</td><td>4.409</td><td>4.560</td><td>163</td><td>787</td></tr>
<tr><td>13</td><td>BERT + DeiT</td><td>5</td><td>0.1724</td><td>6.686</td><td>98.266</td><td>99.439</td></tr>
<tr><td>14</td><td>XLM-RoBERTa + ViT</td><td>16</td><td>1.5276</td><td>13.968</td><td>917.29</td><td>1739.551</td></tr>
</tbody>
</table>
