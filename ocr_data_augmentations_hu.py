# requierments albumentations,
# pip install -U albumentations
import os , time ,random ,shutil ,requests ,json ,zipfile , time 
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
# -------------- BEGIN OF CONFIGURATION ---------------------
WORKING_DIR = '/home/ngyongyossy/mohammad/OCR_HU_Tra2022/data_aug/'
TEXT_PATH = f'{WORKING_DIR}source/labels.jsonl'
SOURCE_IMAGE_PATH = F'{WORKING_DIR}source/images/'
AUG_LABELS_PATH = f'{WORKING_DIR}destination/aug_labels.jsonl'
HANDLE_MISSING    = False
MANY_AUGMENTATION = True
# -------------- END   OF CONFIGURATION ---------------------
# Function that gets PIL image and returns augmented PIL image
def augment_img(img):
  # only augment 3/4th the images
  if random.randint(1, 4) > 3:
      return img  
  # convert to numpy for opencv
  img = np.asarray(img)     

  # morphological alterations
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  if random.randint(1, 5) == 1:
    # dilation because the image is not inverted
    img = cv2.erode(img, kernel, iterations=random.randint(1, 2))
  if random.randint(1, 6) == 1:
    # erosion because the image is not inverted
    img = cv2.dilate(img, kernel,iterations=random.randint(1, 1))
  
  transform = A.Compose([      
    A.OneOf([
      # add black pixels noise
      A.OneOf([
             A.RandomRain( brightness_coefficient= 0.9, 
                           drop_length=1, drop_width=1, 
                           drop_color = (0, 0, 0), 
                           blur_value=2, 
                           rain_type = 'drizzle', 
                           p=0.05), 
              A.RandomShadow(p=1),
              A.PixelDropout(p=1),
              ],  p=0.8),

      # add white pixels noise
      A.OneOf([
              A.PixelDropout(dropout_prob=0.5,drop_value=255,p=1),
             A.RandomRain( brightness_coefficient=1.0, 
                           drop_length=2, 
                           drop_width=2, 
                           drop_color = (252, 255, 205), 
                           blur_value=1, 
                           rain_type = None, p=1), 
             ], p=0.9),
            ], p=1),

    # transformations
    A.OneOf([
            A.ShiftScaleRotate( shift_limit=0.1, 
                                scale_limit=0.25, 
                                rotate_limit=2, 
                                border_mode=cv2.BORDER_CONSTANT, 
                                value=(255,255,255),p=1
                              ),
            A.ShiftScaleRotate( shift_limit=0.1, 
                                scale_limit=0, 
                                rotate_limit=3, 
                                border_mode=cv2.BORDER_CONSTANT, 
                                value=(255,255,255),p=1
                              ),
            A.ShiftScaleRotate( shift_limit=0.02, 
                                scale_limit=0.15, 
                                rotate_limit=5, 
                                border_mode=cv2.BORDER_CONSTANT, 
                                value=(255,255,255),p=1
                              ),  
            A.Affine(           shear=random.randint(-5, 5),
                                mode=cv2.BORDER_CONSTANT, 
                                cval=(255,255,255), p=1
                    )          
            ], p=0.5),
    A.Blur(blur_limit=5,p=0.25),
                        ])
  
  img = transform(image=img)['image']  
  return Image.fromarray(img)

train_text = TEXT_PATH 

# --------------------------------------------------
def load_jsonl():
    return pd.read_json(
                        path_or_buf = train_text,
                        lines=True,
                        )
print(train_text)



"""### Show Random sample"""
def ShowSingleSample(): 
  idx = random.randint(0,len(df))
  print(df['text'][idx])
  print(df['file_name'][idx])
  Image.open(SOURCE_IMAGE_PATH + df['file_name'][idx]).convert("RGB")



"""Let's run the above function on some samples"""
def ShowRandomSample():
  # take sample and do augmentations
  sample_amount =  2 # 8
  # to do choose randomly
  idx1 = random.randint(0,len(df))
  idx2 = random.randint(0,len(df))
  idx3 = random.randint(0,len(df))
  idx4 = random.randint(0,len(df))

  image1 = Image.open(SOURCE_IMAGE_PATH + df['file_name'][idx1]).convert("RGB")
  image2 = Image.open(SOURCE_IMAGE_PATH + df['file_name'][idx2]).convert("RGB")
  image3 = Image.open(SOURCE_IMAGE_PATH + df['file_name'][idx3]).convert("RGB")
  image4 = Image.open(SOURCE_IMAGE_PATH + df['file_name'][idx4]).convert("RGB")
  plt.figure(figsize=(20,25))
  i = 1
  plt.subplot(sample_amount, 2, i).imshow(image1)
  plt.subplot(sample_amount, 2, i).set_title('source image')
  plt.subplot(sample_amount, 2, i).set_axis_off() 
  plt.subplot(sample_amount, 2, i + 1).imshow(augment_img(image1))
  plt.subplot(sample_amount, 2, i + 1).set_title('augmented image')
  plt.subplot(sample_amount, 2, i + 1).set_axis_off() 
  i = 3
  plt.subplot(sample_amount, 2, i).imshow(image2)
  plt.subplot(sample_amount, 2, i).set_title('source image')
  plt.subplot(sample_amount, 2, i).set_axis_off() 
  plt.subplot(sample_amount, 2, i + 1).imshow(augment_img(image2))
  plt.subplot(sample_amount, 2, i + 1).set_title('augmented image')
  plt.subplot(sample_amount, 2, i + 1).set_axis_off()
  i = 5
  plt.subplot(sample_amount, 2, i).imshow(image3)
  plt.subplot(sample_amount, 2, i).set_title('source image')
  plt.subplot(sample_amount, 2, i).set_axis_off() 
  plt.subplot(sample_amount, 2, i + 1).imshow(augment_img(image3))
  plt.subplot(sample_amount, 2, i + 1).set_title('augmented image')
  plt.subplot(sample_amount, 2, i + 1).set_axis_off() 
  i = 7
  plt.subplot(sample_amount, 2, i).imshow(image4)
  plt.subplot(sample_amount, 2, i).set_title('source image')
  plt.subplot(sample_amount, 2, i).set_axis_off() 
  plt.subplot(sample_amount, 2, i + 1).imshow(augment_img(image4))
  plt.subplot(sample_amount, 2, i + 1).set_title('augmented image')
  plt.subplot(sample_amount, 2, i + 1).set_axis_off()


"""## Create a new folder to store the augmented images and the labels file."""
def mkdir1():    
  aug_imgs = f'{WORKING_DIR}destination/aug_imgs'
  if not os.path.exists(aug_imgs):
      os.makedirs(aug_imgs)


# This will be the last resulting dir after combine source with augmented (destenation)
def mkdir2():
  
  output_imgs_folder = f'{WORKING_DIR}output_dir'
  if not os.path.exists(output_imgs_folder):
      os.makedirs(output_imgs_folder)



def dataAug():
  """## Create a new dataframe to store the new file names and corresponding labels."""

  new_df = pd.DataFrame(columns=['file_name', 'text'])
  print(new_df)
  source_imags_path = SOURCE_IMAGE_PATH 
  # To save the augmented images in the output_imgs_folder directory and create a new new_labels.jsonl file with the updated file names, you can modify the for loop as follows:
  sample_amount = 4
  # to add tag to see progress
  for idx in tqdm(range(len(df))):
      image_path =  os.path.join(source_imags_path, df['file_name'][idx])
      image = Image.open(image_path).convert("RGB")
      for i in range(sample_amount):
          augmented_image = augment_img(image)          
          augmented_image_path = os.path.join(  f'{WORKING_DIR}destination/aug_imgs', 
                                                f'aug_{str(idx * sample_amount + i)}.jpg'
                                              )
          augmented_image.save(augmented_image_path)
          # encode "text" file as a Dict format before writing to dataframe 
          dictionary = { 
                         "file_name": os.path.basename(augmented_image_path),
                          "text": df['text'][idx]
                        }
          df_dictionary = pd.DataFrame([dictionary])
          new_df = pd.concat([new_df, df_dictionary], ignore_index=True)

  new_df.to_json( AUG_LABELS_PATH,
                  lines=True, 
                  orient="records", 
                  force_ascii=False)
  print('\n',new_df.head())



# output_dir = f'{WORKING_DIR}output_dir'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)


"""**2.1.1 - Morphological (dilate)**"""

def augment_img_dilate(img):
  img = np.asarray(img)     #convert to numpy
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,1))
  # dilation because the image is not inverted
  img = cv2.erode(img, kernel, iterations=random.randint(2, 4))
  return Image.fromarray(img) 


""" **2.1.2 - Morphological (erode) **"""
def augment_img_erode(img):
  img = np.asarray(img)     
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
  # erosion because the image is not inverted
  img = cv2.dilate(img, kernel,iterations=random.randint(1,1)) 
  return Image.fromarray(img)


""" **2.2 - Black Pixel Noise** """
def augment_img_RandomRain_black(img):
  img = np.asarray(img)     
  transform = A.Compose([
  #add black pixels noise: RandomRain
  A.RandomRain( brightness_coefficient=.9, 
                drop_length=1, drop_width=1, 
                drop_color = (0, 0, 0), 
                blur_value=2, 
                rain_type = 'drizzle', p=1)
    ])
  img = transform(image=img)['image']  
  return Image.fromarray(img)


def augment_img_RandomShadow(img):
  img = np.asarray(img)     
  transform = A.Compose([
  #add black pixels noise: RandomShadow
  A.RandomShadow(p=1)
  #A.PixelDropout(p=1),
  ])
  img = transform(image=img)['image']  
  return Image.fromarray(img) 


def augment_img_PixelDropout_black(img):
  img = np.asarray(img)    
  transform = A.Compose([
  #add black pixels noise: PixelDropout
  A.PixelDropout(p=1)
  ])
  img = transform(image=img)['image']  
  return Image.fromarray(img) 


"""**2.3 - White Pixel Noise**"""
def augment_img_RandomRain_white(img):
  img = np.asarray(img)     #convert to numpy
  transform = A.Compose([
  #add white pixels noise: RandomRain
  A.RandomRain( brightness_coefficient=1.0, 
                drop_length=2, 
                drop_width=2, 
                drop_color = (253, 255, 205), 
                blur_value=1, 
                rain_type = None, 
                p=1)
  ])
  img = transform(image=img)['image']     
  return Image.fromarray(img)



def augment_img_PixelDropout_white(img):
  img = np.asarray(img)     
  transform = A.Compose([
  # add white pixels noise: PixelDropout
  A.PixelDropout(  dropout_prob=0.4,
                   drop_value=250,p=1
                )
  ])
  img = transform(image=img)['image']  
  return Image.fromarray(img)  


"""**2.4 - Transformations**"""
def augment_img_ShiftScaleRotate(img):
  img = np.asarray(img)     
  transform = A.Compose([
  # add white pixels noise: PixelDropout
  A.OneOf([
            A.ShiftScaleRotate( shift_limit=0,
                                scale_limit=0.2,
                                rotate_limit=2,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255,255,255),p=1
                              ),

            A.ShiftScaleRotate( shift_limit=0.05,
                                scale_limit=0,
                                rotate_limit=3,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255,255,255),p=1
                              ),

            A.ShiftScaleRotate( shift_limit=0.02,
                                scale_limit=0.1,
                                rotate_limit=4,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255,255,255),p=1
                              ),  

            A.Affine(           shear=random.randint(-5, 5),
                                mode=cv2.BORDER_CONSTANT,
                                cval=(255,255,255), p=1)          
          ], p=1)
                        ])

  img = transform(image=img)['image']   
  return Image.fromarray(img)


def augment_img_Blur(img):
  img = np.asarray(img)     
  transform = A.Compose([
                          A.Blur(blur_limit=3,p=1),
                        ])
  img = transform(image=img)['image']  
  return Image.fromarray(img)



"""## Process Missing"""
def processMissing():

  path = f'{WORKING_DIR}output_dir/'
  def load_jsonl(path):
      return pd.read_json(
                          path_or_buf = f'{path}all_labels.jsonl',
                          lines=True) 
  df = load_jsonl(path)

  # ---------------------------

  def is_dir_exist(filename): 
      path = f'{WORKING_DIR}output_dir/'
      path_to_file = f'{path}imgs/'+ filename 
      path = Path(path_to_file)
      return path.is_file() 

  list_fn = [
              df['file_name'][idx]
              for idx in range(len(df))
              if not is_dir_exist(df['file_name'][idx])
            ]
  print('list of file names that exist in labels but not in imgs dir: \n', list_fn)

  for i in list_fn:
      df.drop(df[df['file_name'] == i ].index, inplace = True)

  time.sleep(3)
  print("Data frame after processed" , df.head(10))

  # save resulting df 
  reddit = df.to_dict(orient= "records")
  print(type(reddit) , len(reddit))
  # we have list of dict[{},{},{}]
  with open(f"{path}all__labels.jsonl","w") as f:
      for line in reddit:
          f.write(json.dumps(line,ensure_ascii=False) + "\n")


"""## images compression if needed """
def compressImages():
  # path to the directory you want to zip
  dir_path = f"{WORKING_DIR}output_dir/imgs/"
  # name of the zip file you want to create
  zip_filename = "images.zip"

  # create a ZipFile object with write mode
  with zipfile.ZipFile(zip_filename, 'w') as zip:
      # iterate over all files in the directory
      for root, dirs, files in os.walk(dir_path):
          for file in files:
              # get the full path of the file
              file_path = os.path.join(root, file)
              # add the file to the zip file
              zip.write(file_path, os.path.relpath(file_path, dir_path))


if __name__=='__main__':
   
   df = load_jsonl()
   print(f'length of df : {len(df)} \n',df.head())

  #  ShowSingleSample()

  #  ShowRandomSample()

   mkdir1()
   mkdir2()
   # to check how many images do we heve to be Aug.
   print(len(
              os.listdir(SOURCE_IMAGE_PATH)
            )
        )
   if MANY_AUGMENTATION: 
    dataAug()

   else:

      """ # 2- Functions to do 1 augmentation only
      * In this way you can do one type of augumention at each time in spreate way

      **2.1 Morphological Alterations**
      """
      # Read an image 
      path = '/content/drive/MyDrive/TrsOCR_utorial/training-data-ex/img/' 
      image = Image.open(path + df['file_name'][0]).convert("RGB")
      print(image)

      img = np.asarray(image)     
      image = Image.fromarray(img)
      print(image)

      augment_img_dilate(image)

      augment_img_erode(image) 

      augment_img_RandomRain_black(image)

      augment_img_RandomShadow(image)

      augment_img_PixelDropout_black(image)

      augment_img_RandomRain_white(image)

      augment_img_PixelDropout_white(image)

      augment_img_ShiftScaleRotate(image)

      augment_img_Blur(image)

  #  If needed 
  #  compressImages()

   if HANDLE_MISSING: 
    processMissing()
