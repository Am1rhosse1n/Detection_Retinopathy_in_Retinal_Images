import cv2 as cv
import numpy as np
import glob
import random
from scipy.fftpack import dct



def zigzagOrder(x,scale):
    matrix = x
    rows= matrix.shape[0]
    columns= matrix.shape[1]
    	
    solution=[[] for i in range(rows+columns-1)]
    
    for i in range(rows):
    	for j in range(columns):
    		sum=i+j
    		if(sum%2 ==0):
    
    			#add at beginning
    			solution[sum].insert(0,matrix[i][j])
    		else:
    
    			#add at end of the list
    			solution[sum].append(matrix[i][j])
    		
    			
    # print the solution as it as
    result = []
    for i in solution:
    	for j in i:
    		result.append(j)
    result.pop(0)
    
    lenth = len(result)
    n = int(lenth*scale)
#    print("N is: ",n)
    del result[-n:]
    return result

    


image_n = []
image_a = []
masks_n = []
masks_a = []

lable = []
dataset = []

# Load images

for img in glob.glob("dataset/abnormal/*.png"):
    im = cv.imread(img,0)
    image_a.append(im)

for img in glob.glob("dataset/normal/*.png"):
    im = cv.imread(img,0)
    image_n.append(im)

for img in glob.glob("dataset/abnormal_mask/*.png"):
    im = cv.imread(img,0)
    masks_a.append(im)

for img in glob.glob("dataset/normal_mask/*.png"):
    im = cv.imread(img,0)
    masks_n.append(im)
    
"""
 Preprocess: 1)Apply Masks   2)Downsample   3)DCT2D    4)Zigzag Order
             5)Discard Less Important 99% Component    6)Build Lables
"""
#Abnormal
for i in range(len(image_a)):
    temp = image_a[i] * (masks_a[i]/255)
    temp = cv.resize(temp,None, fx=.3 , fy=.3 , interpolation = cv.INTER_NEAREST)
    dct2d = dct(dct(temp.T).T)
    dct2d_zigzag = zigzagOrder(dct2d,0.99)
    dataset.append(dct2d_zigzag)
    lable.append(0)  # Abnormal = 0

#Normal    
for i in range(len(image_n)):
    temp = image_n[i] * (masks_n[i]/255)
    temp = cv.resize(temp,None, fx=.3 , fy=.3 , interpolation = cv.INTER_NEAREST)    
    dct2d = dct(dct(temp.T).T)
    dct2d_zigzag = zigzagOrder(dct2d,0.99)
    dataset.append(dct2d_zigzag)
    lable.append(1) # Normal = 1

#Shuffle Data
shuffle = []
for i in range(130):
    shuffle.append(i)
random.shuffle(shuffle)

dataset_v1 = []
lable_v1 = []
for i in shuffle:
    dataset_v1.append(dataset[i])
    lable_v1.append(lable[i])


#np.save('dataset.npy',dataset_v1)
#np.save('lables.npy',lable_v1)


