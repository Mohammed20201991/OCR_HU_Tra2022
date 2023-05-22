import requests
from PIL import Image
from transformers import TrOCRProcessor ,VisionEncoderDecoderModel
import gradio as gr
from PIL import Image

# url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# image

ENGLISH_MODEL = False
if ENGLISH_MODEL:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# HUNGARAIN_MODEL
else:
    check_point = '/home/ngyongyossy/mohammad/OCR_HU_Tra2022/1/trocr_large_lines_v2_1_ft_on_dh-lab_aug/checkpoint-140000'
    processor = '/home/ngyongyossy/mohammad/OCR_HU_Tra2022/1/trocr_large_lines_v2_1_ft_on_dh-lab_aug/processor'
    processor = TrOCRProcessor.from_pretrained(processor)
    model = VisionEncoderDecoderModel.from_pretrained(check_point)
# calling the processor is equivalent to calling the feature extractor
# pixel_values = processor(image, return_tensors="pt").pixel_values
# print(pixel_values.shape)



# generated_ids = model.generate(pixel_values,max_length= 64)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens= False)[0]
# print(generated_text)

# load image examples from the Hungarain database
# {"file_name": "RALK987_1865_817_161_001-016.jpg", 
# "text": "fáradozásit a tudomány érdekében mindenkor nagy érdek-"# }
# {"file_name": "RALK987_1865_817_297_003-029.jpg", 
# "text": "bátor vagyok kérdésbe tenni, hogy jár-e ezek-"}
# {"file_name": "RALK987_1865_817_296_001-014.jpg", 
# "text": "és Pólya József r. tagoknak adatott, ki birálat végett."}

urls = ['/home/ngyongyossy/mohammad/Data/DH-Lab_lines/images/RALK987_1865_817_161_001-016.jpg',
        '/home/ngyongyossy/mohammad/Data/DH-Lab_lines/images/RALK987_1865_817_297_003-029.jpg',
        '/home/ngyongyossy/mohammad/Data/DH-Lab_lines/images/RALK987_1865_817_296_001-014.jpg']
for idx, url in enumerate(urls):
  image = Image.open(url)
#   image = Image.open(requests.get(url, stream=True).raw)
  image.save(f"image_{idx}.png")

def process_image(image):
    # prepare image
    pixel_values = processor(image, return_tensors="pt").pixel_values
    # generate (no beam search)
    generated_ids = model.generate(pixel_values,pad_token_id=processor.tokenizer.eos_token_id)
    # decode 
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # generated_text

title = "Interactive demo: HuTrOCR"
description = "Demo for Hungarain TrOCR, (DH-Lab (Arany200) : 3.681 % CER, 16.189% WER) an encoder-decoder model consisting of an image Transformer encoder and a text Transformer decoder for state-of-the -art optical character recognition (OCR) on single-text line images. This particular model is pre-trained on Synthestic data and then fine-tuned on Hungarrain,a dataset of annotated handwritten images. To use it, simply upload an image or use the example image below and click 'submit'. Results will show up in a few seconds."
article = "<p style='text-align: center'> <a href='https://github.com/Mohammed20201991'>Github Repo</a></p>"
examples =[["image_0.png"], ["image_1.png"], ["image_2.png"]]

iface = gr.Interface(fn=process_image, 
                     inputs=gr.inputs.Image(type="pil"), 
                     outputs=gr.outputs.Textbox(),
                     title=title,
                     description=description,
                     article=article,
                     examples=examples)
iface.launch(debug=True,share=True) #,debug=False

