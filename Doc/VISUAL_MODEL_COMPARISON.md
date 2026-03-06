# Visual Model Comparison (Encoder) Experiment

In this experiment, several **state-of-the-art vision models** were evaluated as the **encoder component** in the TrOCR architecture. 
The goal was to compare different visual encoders while keeping the **decoder fixed as a Causal Language Model (CLM)**. This allows us to observe how different vision encoders influence OCR performance.

All models were trained using the same training configuration:

- **Learning rate:** `2e-5`
- **Epochs:** `12`
- **Batch size:** `16`

The experiment was conducted using the **DH-Lab human-annotated dataset**, and the results reported below correspond to the **evaluation set before dataset splitting**.

## Results

| Encoder | Steps (K) | Train Loss | Val Loss | CER | WER |
|-------|-------|-------|-------|-------|-------|
| ViT | 8 | 0.002 | 0.360 | **4.624** | **20.0** |
| BEiT | 5 | 0.003 | 0.339 | 4.684 | 20.676 |
| Swin | 5 | 0.004 | 0.350 | 4.876 | 21.64 |
| DeiT | 5 | 0.0023 | 0.358 | **4.624** | 20.869 |

## Observations

After a single training iteration, the results suggest that **changing the encoder architecture does not significantly influence overall OCR performance** when the decoder remains fixed.

- **ViT** achieved the **best results** in terms of both **Character Error Rate (CER)** and **Word Error Rate (WER)**.
- **DeiT** achieved the **same CER as ViT**, indicating competitive performance.
- **BEiT** and **Swin** produced slightly higher error rates but still demonstrated stable performance.

These results indicate that **large pre-trained vision models provide sufficiently strong image representations**, meaning the encoder component already performs image understanding effectively.

## Model Selection Considerations

Although **ViT achieved the best results in this experiment**, **DeiT** was selected for later experiments due to its strong compatibility with **RoBERTa-base** and **PULI-BERT** models when exploring encoder–decoder combinations.

Additionally, DeiT follows a **student–teacher training paradigm**, which can provide better scalability when increasing the dataset size.

From the literature review, **BEiT-3** is currently considered one of the strongest vision-language models. However, within the scope of this experiment and dataset, the tested models showed **comparable encoder performance**.

## Conclusion

The experiment suggests that **modern pre-trained vision encoders perform similarly well for OCR tasks**. Therefore, further improvements should focus primarily on the **text generation component (decoder)** rather than the visual encoder.
