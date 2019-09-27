import shutil
import pandas as pd
from joblib import Parallel, delayed
import subprocess
import os
import cv2
import joblib
from skimage import io
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pafy
from skimage import transform
from sklearn.manifold import Isomap
import logging
ausnahme={
    'out/cropped/buntefilmbilder_album7_S18_B8_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S19_B6_1_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S24_B6_0_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S19_B8_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S7_B6_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S22_B6_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S19_B6_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S10_B3_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S11_B6_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S22_B8_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S23_B8_1_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S7_B8_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S11_B8_0_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S17_B8_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S10_B1_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S3_B6_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S17_B6_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S6_B8_1_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S7_B8_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S5_B8_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S22_B6_0_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S18_B8_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S18_B6_0_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S11_B8_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S24_B6_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S23_B6_0_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S17_B8_1_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S2_B2_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S29_B2_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S2_B7_0_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S15_B3_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S3_B8_1_cropped.jpg':'m',
    'out/cropped/buntefilmbilder_album7_S3_B8_0_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_Titelblatt_0_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S23_B6_1_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S10_B1_1_cropped.jpg':'w',
    'out/cropped/buntefilmbilder_album7_S17_B6_1_cropped.jpg':'w'}
imegefolder='FOLDER/WITH/PICTURES'

def findface(ordner,filename):

    if filename.endswith('jpg'):
        print(os.path.join(ordner,filename.split('.')[0]+'.csv'))
        try:
            dummy = pd.read_csv((os.path.join(ordner,filename.split('.')[0]+'.csv')),sep=',')
            root, ext = os.path.splitext(filename)
            dummy['file'] = ''
            for i in range(len(dummy)):
                dummy.set_value(i,'file', 'out/cropped/'+filename)

            dummy.index=dummy['file']
            print(dummy)
            return dummy

        except Exception as exp:
            print('-----------------------Problem', exp)


cwd=os.getcwd()
print('/home/erik/OpenFace/build/bin/FaceLandmarkImg -fdir "' +bilderordner + '" -wild 1 -out_dir out/'+str(bilderordner))
subprocess.call('/home/erik/OpenFace/build/bin/FaceLandmarkImg -fdir "' +bilderordner + '" -wild 1 -out_dir out/'+str(bilderordner),shell=True)
ordner=os.path.join(cwd , 'out',str(bilderordner))
downloaded_videos = Parallel(n_jobs=4)(delayed(findface)(ordner, filename) for filename in os.listdir(bilderordner))
faces = pd.concat(downloaded_videos)
dfsex= pd.read_csv('BunteFilmBilder-1.csv', sep='\t')
for face in faces.itertuples():
    geteilt = face[0].split('/')[2].split('_')
    if len(geteilt) > 3:
        print(len(geteilt))
        print(geteilt[0] + '_' + geteilt[1] + '_' + geteilt[2] + '_' + geteilt[3])
        name = geteilt[0] + '_' + geteilt[1] + '_' + geteilt[2] + '_' + geteilt[3]
        dfsexausw = dfsex[dfsex.Dateiname == name]
        if len(dfsexausw) == 1:
            dfsexausw.reset_index(drop=True, inplace=True)
            print(dfsexausw.get_value(0, 'Geschlecht'))
            faces.set_value(face[0], 'Geschlecht', dfsexausw.get_value(0, 'Geschlecht'))
        else:
            if face[0] in ausnahme.keys():
                faces.set_value(face[0], 'Geschlecht', ausnahme[face[0]])

faces.to_pickle('faces_sex.p')
faces.to_csv('faces_sex.csv', sep='\t')
# ['file', 'AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']]
dfposition=faces[[' pose_Rx',' pose_Ry',' pose_Rz']]
print(dfposition)
matrix = dfposition.as_matrix()
imf = matrix
np.shape(imf)
from sklearn import manifold
xy = manifold.TSNE(n_components=3).fit_transform(imf)
#print(corpus_map)
imap = Isomap()
corpus_map = imap.fit_transform(matrix)
#print(corpus_map)
plt.scatter(corpus_map[:,0],corpus_map[:,1])
x = corpus_map[:,0].copy()
y = corpus_map[:,1].copy()

# normalise the x values to be between -1 and +1:
x *= (1/np.max(  abs(x)   ))
# shift them so they are between 0 and 2
x += 1
x *= 2300

y *= (1/np.max(  abs(y)   ))
y += 1
y *= 2300
from skimage import transform
from skimage import io
import cv2 as cv

myCanvas = np.zeros((5000, 5000, 3))


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


# myCanvas = create_blank(5000, 5000, rgb_color=(255, 255, 255))

import tqdm

for i in tqdm.tqdm(range(len(dfposition.index.tolist()))):
    px = int(x[i])
    py = int(y[i])
    print("out/" + dfposition.index.tolist()[i])
    imr = cv.imread(dfposition.index.tolist()[i].replace('\\', '/'), cv.IMREAD_GRAYSCALE)
    auswahl=faces[faces.index==dfposition.index.tolist()[i]]
    auswahl.reset_index(drop='True', inplace=True)
    if auswahl.get_value(0, 'Geschlecht')=='m':
        imr= cv.applyColorMap(imr, cv.COLORMAP_WINTER)
        print('m')
        io.imsave("out/" + dfposition.index.tolist()[i], imr)
    elif auswahl.get_value(0, 'Geschlecht')=='w':
        imr= cv.applyColorMap(imr, cv.COLORMAP_HOT)
        print('w')
    imr=transform.resize(imr, (128, 128, 3))
    myCanvas[px:px + 128, py:py + 128] = imr
io.imsave('Output_facedirection.jpg', myCanvas)
dfposition=faces[[' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r']]
print(dfposition)
matrix = dfposition.as_matrix()
imf = matrix
np.shape(imf)
from sklearn import manifold
from sklearn.cluster import KMeans
xy = manifold.TSNE(n_components=3).fit_transform(imf)
#print(corpus_map)
km = KMeans(n_clusters=4)
clusters=km.fit_predict(matrix)
imap = Isomap()
corpus_map = imap.fit_transform(matrix)

#print(corpus_map)
plt.scatter(corpus_map[:,0],corpus_map[:,1])
x = corpus_map[:,0].copy()
y = corpus_map[:,1].copy()

# normalise the x values to be between -1 and +1:
x *= (1/np.max(  abs(x)   ))
# shift them so they are between 0 and 2
x += 1
x *= 2300

y *= (1/np.max(  abs(y)   ))
y += 1
y *= 2300
from skimage import transform
from skimage import io
import cv2 as cv

myCanvas = np.zeros((5000, 5000, 3))


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


# myCanvas = create_blank(5000, 5000, rgb_color=(255, 255, 255))

import tqdm

for i in tqdm.tqdm(range(len(dfposition.index.tolist()))):
    px = int(x[i])
    py = int(y[i])
    print("out/" + dfposition.index.tolist()[i].replace('\\', '/')+'.jpg')
    imr = cv.imread(dfposition.index.tolist()[i].replace('\\', '/'), cv.IMREAD_GRAYSCALE)
    auswahl=faces[faces.index==dfposition.index.tolist()[i]]
    auswahl.reset_index(drop='True', inplace=True)
    if auswahl.get_value(0, 'Geschlecht')=='m':
        imr= cv.applyColorMap(imr, cv.COLORMAP_WINTER)
        print('m')
        io.imsave("out/" + dfposition.index.tolist()[i], imr)
    elif auswahl.get_value(0, 'Geschlecht')=='w':
        imr= cv.applyColorMap(imr, cv.COLORMAP_HOT)
        print('w')
    imr=transform.resize(imr, (128, 128, 3))
    myCanvas[px:px + 128, py:py + 128] = imr
io.imsave('Output_faceemotion.jpg', myCanvas)
dfposition=faces[[' AU01_c',' AU02_c',' AU04_c',' AU05_c',' AU06_c',' AU07_c',' AU09_c',' AU10_c',' AU12_c',' AU14_c',' AU15_c',' AU17_c',' AU20_c',' AU23_c',' AU25_c',' AU26_c',' AU45_c']]
print(dfposition)
matrix = dfposition.as_matrix()
imf = matrix
np.shape(imf)
from sklearn import manifold
xy = manifold.TSNE(n_components=3).fit_transform(imf)
#print(corpus_map)
imap = Isomap()
corpus_map = imap.fit_transform(matrix)
#print(corpus_map)
plt.scatter(corpus_map[:,0],corpus_map[:,1])
x = corpus_map[:,0].copy()
y = corpus_map[:,1].copy()

# normalise the x values to be between -1 and +1:
x *= (1/np.max(  abs(x)   ))
# shift them so they are between 0 and 2
x += 1
x *= 2300

y *= (1/np.max(  abs(y)   ))
y += 1
y *= 2300
from skimage import transform
from skimage import io
import cv2 as cv

myCanvas = np.zeros((5000, 5000, 3))


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


# myCanvas = create_blank(5000, 5000, rgb_color=(255, 255, 255))

import tqdm

for i in tqdm.tqdm(range(len(dfposition.index.tolist()))):
    px = int(x[i])
    py = int(y[i])
    print("out/" + dfposition.index.tolist()[i].replace('\\', '/')+'.jpg')
    imr = cv.imread(dfposition.index.tolist()[i].replace('\\', '/'), cv.IMREAD_GRAYSCALE)
    auswahl=faces[faces.index==dfposition.index.tolist()[i]]
    auswahl.reset_index(drop='True', inplace=True)
    if auswahl.get_value(0, 'Geschlecht')=='m':
        imr= cv.applyColorMap(imr, cv.COLORMAP_WINTER)
        print('m')
        io.imsave("out/" + dfposition.index.tolist()[i], imr)
    elif auswahl.get_value(0, 'Geschlecht')=='w':
        imr= cv.applyColorMap(imr, cv.COLORMAP_HOT)
        print('w')
    imr=transform.resize(imr, (128, 128, 3))
    myCanvas[px:px + 128, py:py + 128] = imr
io.imsave('Output_faceemotion_bool.jpg', myCanvas)

