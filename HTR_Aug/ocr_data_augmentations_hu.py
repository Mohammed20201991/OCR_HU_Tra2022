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

def augment_img(img):
  '''
  Function that gets PIL image and returns augmented PIL image depend on 
  how many augmented images you want "e.g" if we want 4 times augmented data 
  we augment only 3/4th the images and keep the original one 
  '''
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

# --------------------------------------------------
def load_jsonl(path):
    '''
    Function for data loading(labels), Where data saved in JSON Line format \n
    In (image_name,text) pair And returns Datafreame 
    '''
    return pd.read_json(
                        path_or_buf = train_text,
                        lines=True,
                        )

# --------------------------------------------------
def ShowSingleSample():
  '''
  Function shows randomly selected sample(only one) 
  '''
  idx = random.randint(0,len(df))
  print(df['text'][idx])
  print(df['file_name'][idx])
  Image.open(SOURCE_IMAGE_PATH + df['file_name'][idx]).convert("RGB")

# --------------------------------------------------
def ShowRandomSample():
  '''
  Let's run the augment_img function on some samples \n
  Take sample randomly and do augmentations
  '''
  sample_amount =  2 # 8
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

# --------------------------------------------------
def mkdir_1():
  '''
  Create a new folder to store the augmented images and the labels file.
  '''    
  aug_imgs = f'{WORKING_DIR}destination/aug_imgs'
  if not os.path.exists(aug_imgs):
      os.makedirs(aug_imgs)

# --------------------------------------------------
def mkdir_2():
  '''
  This will be the last resulting dir after combine source with augmented (destenation)
  '''
  output_imgs_folder = f'{WORKING_DIR}output_dir'
  if not os.path.exists(output_imgs_folder):
      os.makedirs(output_imgs_folder)

# --------------------------------------------------
def dataAug():
  '''
  Create a new dataframe to store the new file names and corresponding labels.\n
  sLoop through the data frame and call augment_img function at the end save returned results 
  '''

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

# --------------------------------------------------
def augment_img_dilate(img):
  '''
  Function for Morphological(dilate)
  '''
  img = np.asarray(img)    
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,1))
  # dilation because the image is not inverted
  img = cv2.erode(img, kernel, iterations=random.randint(2, 4))
  return Image.fromarray(img) 

# --------------------------------------------------
def augment_img_erode(img):
  '''
  Function for Morphological (erode)
  '''
  img = np.asarray(img)     
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
  # erosion because the image is not inverted
  img = cv2.dilate(img, kernel,iterations=random.randint(1,1)) 
  return Image.fromarray(img)

# --------------------------------------------------
def augment_img_RandomRain_black(img):
  '''
  Function for Black Pixel Noise, Where RandomRain used 
  '''
  img = np.asarray(img)     
  transform = A.Compose([
  # add black pixels noise: RandomRain
  A.RandomRain( brightness_coefficient=.9, 
                drop_length=1, drop_width=1, 
                drop_color = (0, 0, 0), 
                blur_value=2, 
                rain_type = 'drizzle', p=1)
    ])
  img = transform(image=img)['image']  
  return Image.fromarray(img)

# --------------------------------------------------
def augment_img_RandomShadow(img):
  '''
  Function for add black pixels noise: RandomShadow
  '''
  img = np.asarray(img)     
  transform = A.Compose([
                          A.RandomShadow(p=1)
                          #A.PixelDropout(p=1),
                        ])
  img = transform(image=img)['image']  
  return Image.fromarray(img) 

# --------------------------------------------------
def augment_img_PixelDropout_black(img):
  '''
  add black pixels noise: PixelDropout
  '''
  img = np.asarray(img)    
  transform = A.Compose([
                          A.PixelDropout(p=1)
                        ])
  img = transform(image=img)['image']  
  return Image.fromarray(img) 

# --------------------------------------------------
def augment_img_RandomRain_white(img):
  '''
  Function for White Pixel Noise
  '''
  img = np.asarray(img)     
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

# --------------------------------------------------
def augment_img_PixelDropout_white(img):
  '''
  Function for adding white pixels noise: PixelDropout
  '''
  img = np.asarray(img)     
  transform = A.Compose([
                          A.PixelDropout(  dropout_prob=0.4,
                                          drop_value=250,p=1,
                                        )
                        ])
  img = transform(image=img)['image']  
  return Image.fromarray(img)  

# --------------------------------------------------
def augment_img_ShiftScaleRotate(img):
  '''
  Function for doing Transformations here we should be aware with parameters like rotation 
  '''
  img = np.asarray(img)     
  transform = A.Compose([
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

# --------------------------------------------------
def augment_img_Blur(img):
  '''
  Function for Add Blur to image 
  '''
  img = np.asarray(img)     
  transform = A.Compose([
                          A.Blur(blur_limit=3,p=1),
                        ])
  img = transform(image=img)['image']  
  return Image.fromarray(img)

# --------------------------------------------------
def processMissing():
  '''
  Function for Process Missing, After the data has  been augmented we do check if there is(are) missing images\n
  So in this case we drop it from labels file (train.jsonl) 
  '''
  path = f'{WORKING_DIR}output_dir/'
  def load_jsonl(path):
      return pd.read_json(
                          path_or_buf = f'{path}all_labels.jsonl',
                          lines=True) 
  df = load_jsonl(path)
  
 # ++++++++++++++++++++++++++++++++++++++++++++++
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

# --------------------------------------------------
def compressImages():
  '''
  Function for images compression if needed \n
  Path to the directory you want to zip
  ''' 
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

# --------------------------------------------------
if __name__=='__main__':

   train_text= TEXT_PATH
   df = load_jsonl(train_text)
   print(f'length of df : {len(df)} \n',df.head())

  #  ShowSingleSample()

  #  ShowRandomSample()

   mkdir_1()
   mkdir_2()
   # to check how many images do we heve to be Augmented.
   print(len(
              os.listdir(SOURCE_IMAGE_PATH)
            )
        )
   if MANY_AUGMENTATION:
    "1- Use mixed augumentation" 
    dataAug()

   else:
      '''
      2- Functions to do one augmentation only in each time 
      * In this way you can do one type of augumention at each time in spreate way
     '''
      # Read an image 
      path = f'{SOURCE_IMAGE_PATH}' 
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
