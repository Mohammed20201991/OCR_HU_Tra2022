# Data Augmentation in Efficient ways
<!DOCTYPE html>
<html>
<head>
</head>
<body>

<ul dir="auto">
<li><a href="#Description">Description</a></li>
<li><a href="#How to use it?">How to use it?</a></li>
<li><a href="#illusturation">Illusturation</a></li>
<li><a href="#references">References</a></li>
</ul>

**<p id="Description">Description:</p>**

Augmentation approaches by useing Computer Vision(CV) methods, which will increase the meager dataset we received from DH-Lab.
Small adjustments are made every single time a data point goes through training, reducing the likelihood of over-fitting as well as improving generalization
This method is expiermental for now we are using basic augmentation methods there are many advantges the could be used fron OpenCV lib and Albumation.

**Content:**

- ocr_data_augmentations_hu : Augumentation script for HTR 
- requirements.txt : Contains all necessory dependencies
- Collab Notebook : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YA-VV-RZYf9MOV8a6b-kXtQyPdnOkG57?usp=sharing)
**<p id= "How to use it?"> How to use it?</p>**

~~~bash  
!git clone https://github.com/Mohammed20201991/OCR_HU_Tra2022.git
!cd OCR_HU_Tra2022/HTR_Aug/
~~~

<p>You need to install requirements using the command : </p>

~~~bash  
!pip install -r requirements.txt
~~~

Start data augmentation

~~~bash  
!python3 ocr_data_augmentations_hu.py
~~~

<h6 id="illusturation"> Procedure method for data augmentation(Self-made) </h6>

![plot]()

<h6> Figure show Different Augmentation methods (Self-made) </h6>

![plot]()
**<p id="references">References:</p>**
<ul dir="auto">
<li><a href="https://opencv.org/">OpenCV</a></li>
<li><a href="https://albumentations.ai">Albumentations</a></li>
<li><a href="https://towardsdatascience.com/effective-data-augmentation-for-ocr-8013080aa9fa">effective-data-augmentation-for-ocr</a></li>
</ul>
</html>
</body>