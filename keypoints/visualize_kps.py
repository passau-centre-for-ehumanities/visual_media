import pandas as pd
from gender_guesser import detector
import numpy as np
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import subprocess
from joblib import Parallel, delayed
import os
import json

imglocations="/media/snjuk/DATA/virtual_history/"
imglocationsorgs="/media/snjuk/DATA/Virtual-History/"
dfkps=pd.read_pickle('dfkps_all.p')

dfmeta=pd.read_json('virtual-history.json')
dfmeta.rename(columns={'url':'album_url'},inplace=True)
dfmetapia=pd.read_excel('FilmBilderRecherche.xlsx')
print(dfmetapia.album_url[0])
dfmeta=pd.concat([dfmeta.drop(['images'], axis=1),dfmeta['images'].apply(pd.Series)],axis=1)
dfmeta=pd.concat([dfmeta.drop([0], axis=1),dfmeta[0].apply(pd.Series)],axis=1)
dfmeta['key']=dfmeta['path'].str.replace('full/','')
d = detector.Detector()
def get_genders(names):
    results = []
    for name in names:
        results.append(d.get_gender(name.split(" ")[0]))
    return results
dfmeta['gender'] = dfmeta['names'].apply(lambda x: get_genders(x))
print(dfmeta['gender'])

dfmeta.to_csv('dfmeta.csv',sep='\t')
dfkps['ind']=dfkps['Bild'].astype(str)+dfkps['Person'].astype(str)

dfkpsx=dfkps[dfkps.Achse=='x']
dfkpsy=dfkps[dfkps.Achse=='y']
dfkpsy.drop(['Achse','Bild','Person','Bildbreite','Bildlänge'],axis=1, inplace=True)
dfkps_xy=pd.merge(dfkpsx, dfkpsy, on=['ind'])
dfkps_xy.set_index('ind',inplace=True)

dfkps_meta=pd.merge(dfmeta,dfkps_xy,left_on='key',right_on='Bild', how='left')
print(len(dfmeta), len(dfkps))
dfkps_meta=pd.merge(dfkps_meta,dfmetapia,left_on='album_url',right_on='album_url', how='left')
print(len(dfkps_meta))
#for line in dfkps_meta.itertuples():
#    auswahl=dfmetapia
#    for word in line.album_url.split('/')[-1].split('-'):
#        auswahl=auswahl[auswahl.Serie.str.contains(word.lower())]
    #print(auswahl,line.album_url.split('/')[-1])
dfkps_meta_with_pics = dfkps_meta.dropna(axis=0, how='any',subset=['0_y','1_y',	'2_y',	'3_y',	'4_y',	'5_y',	'6_y',	'7_y',	'8_y',	'9_y',	'10_y',	'11_y',	'12_y',	'13_y',	'14_y',	'15_y',	'16_y'])
print(len(dfkps_meta_with_pics))
# for col in dfkps_meta_with_pics:
#     if col not in [ 'image_urls',            'names',           'number',
#               'album_url',                  0,         'checksum',
#                    'path',              'url',              'key',
#                    'Bild',            'Achse',           'Person','Unnamed: 2',
#              'Hersteller',  'Produktionsland', 'Verbreitungszeit',
#                   'Links','gender','Serie']:
#         if 'x' in str(col):
#             print('Rechne Werte von',col, ' durch Bildbreite')
#             dfkps_meta_with_pics[col]=dfkps_meta_with_pics[col]/dfkps_meta_with_pics['Bildbreite']
#         elif 'x' in str(col):
#             print('Rechne Werte von', col, ' durch Bildlänge')
#             dfkps_meta_with_pics[col] = dfkps_meta_with_pics[col] / dfkps_meta_with_pics['Bildlänge']

import math

def dotproduct(v1, v2):
  #print(v1,v2)
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def angle(v1, v2):
  #print(dotproduct(v1, v2) / (length(v1) * length(v2)))
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def calc_angle_per_row(row, point1, point2, point3):
    #print(row[point1+"_x"],row[point2+"_x"],row[point3+"_x"],row['Bild'])
    if (str(row[point1+"_x"]) != 'nan') and (str(row[point2+"_x"]) != 'nan') and (str(row[point3+"_x"]) != 'nan'):
        #print(row[point2+"_x"]-row[point1+"_x"], row[point2+"_y"]-row[point1+'_y'], (row[point3+'_x']-row[point1+'_x'], row[point3+'_y']-row[point1+'_y']))
        return angle_between((row[point2+"_x"]-row[point1+"_x"], row[point2+"_y"]-row[point1+'_y']), (row[point3+'_x']-row[point1+'_x'], row[point3+'_y']-row[point1+'_y']))*180/math.pi
    else:
        return -1

def calc_left_elbow_angle(row):
    return calc_angle_per_row(row, '7', '5', '9')

def calc_right_elbow_angle(row):
    return calc_angle_per_row(row, '8', '6', '10')

def calc_left_shoulder_angle(row):
    return calc_angle_per_row(row, '5', '7', '6')

def calc_right_shoulder_angle(row):
    return calc_angle_per_row(row, '6', '8', '5')
def calc_right_hip_shoulder_knee_angle(row):
    return calc_angle_per_row(row, '12', '6', '14')
def calc_left_hip_shoulder_knee_angle(row):
    return calc_angle_per_row(row, '11', '5', '13')
#def calc_right_shoulder_angle(row):
#    return calc_angle_per_row(row, '6', '8', '5')
def calc_left_hip_angle(row):
    return calc_angle_per_row(row, '11', '12', '13')
def calc_right_hip_angle(row):
    return calc_angle_per_row(row, '12', '11', '14')
def calc_left_knee_angle(row):
    return calc_angle_per_row(row, '13', '11', '15')
def calc_right_knee_angle(row):
    return calc_angle_per_row(row, '14', '12', '16')
def calc_right_shoulder_hip_angle(row):
    return calc_angle_per_row(row, '5', '6', '11')
def calc_left_shoulder_hip_angle(row):
    return calc_angle_per_row(row, '6', '5', '12')
def calc_nose_eys_angle(row):
    return calc_angle_per_row(row, '0', '1', '2')
def calc_left_eye_nose_ear_angle(row):
    return calc_angle_per_row(row, '1', '3', '0')
def calc_right_eye_nose_ear_angle(row):
    return calc_angle_per_row(row, '2', '4', '0')
def calc_left_schoulder_hip(row):
    return calc_angle_per_row(row, '5', '11', '6')
def calc_right_schoulder_hip(row):
    return calc_angle_per_row(row, '6', '12', '5')
print(dfkps_meta_with_pics)
dfkps_meta_with_pics['left_elbow_angle'] = dfkps_meta_with_pics.apply(calc_left_elbow_angle, axis=1)
dfkps_meta_with_pics['right_elbow_angle'] = dfkps_meta_with_pics.apply(calc_right_elbow_angle, axis=1)
dfkps_meta_with_pics['left_shoulder_angle'] = dfkps_meta_with_pics.apply(calc_left_shoulder_angle, axis=1)
dfkps_meta_with_pics['right_shoulder_angle'] = dfkps_meta_with_pics.apply(calc_right_shoulder_angle, axis=1)
dfkps_meta_with_pics['left_hip_angle'] = dfkps_meta_with_pics.apply(calc_left_hip_angle, axis=1)
dfkps_meta_with_pics['right_hip_angle'] = dfkps_meta_with_pics.apply(calc_right_hip_angle, axis=1)
dfkps_meta_with_pics['left_knee_angle'] = dfkps_meta_with_pics.apply(calc_left_knee_angle, axis=1)
dfkps_meta_with_pics['right_knee_angle'] = dfkps_meta_with_pics.apply(calc_right_knee_angle, axis=1)
dfkps_meta_with_pics['right_shoulder_hip_angle'] = dfkps_meta_with_pics.apply(calc_right_shoulder_hip_angle, axis=1)
dfkps_meta_with_pics['left_shoulder_hip_angle'] = dfkps_meta_with_pics.apply(calc_left_shoulder_hip_angle, axis=1)
dfkps_meta_with_pics['nose_eys_angle'] = dfkps_meta_with_pics.apply(calc_nose_eys_angle, axis=1)
dfkps_meta_with_pics['left_eye_nose_ear_angle'] = dfkps_meta_with_pics.apply(calc_left_eye_nose_ear_angle, axis=1)
dfkps_meta_with_pics['right_eye_nose_ear_angle'] = dfkps_meta_with_pics.apply(calc_right_eye_nose_ear_angle, axis=1)
dfkps_meta_with_pics['right_schoulder_hip'] = dfkps_meta_with_pics.apply(calc_right_schoulder_hip, axis=1)
dfkps_meta_with_pics['left_schoulder_hip'] = dfkps_meta_with_pics.apply(calc_left_schoulder_hip, axis=1)
print('Berechnung der Winkel abgeschlossen')
angles=['right_schoulder_hip','left_schoulder_hip','left_elbow_angle','right_elbow_angle','left_shoulder_angle','right_shoulder_angle','left_hip_angle','right_hip_angle','left_knee_angle','right_knee_angle','right_shoulder_hip_angle','left_shoulder_hip_angle','nose_eys_angle','left_eye_nose_ear_angle','right_eye_nose_ear_angle']
def findface(ordner,filename):
    cwd=os.getcwd()
    if filename.endswith('csv'):
            #print('Bearbeite Datei mit Namen:',filename)
        #print(os.path.join(ordner,filename.split('.')[0]+'.csv'))
#        try:
            dummy = pd.read_csv((os.path.join(cwd,ordner,filename)),sep=',')
            root, ext = os.path.splitext(filename)
            dummy['file'] = ''
            dummy['oldfile']=root+'.jpg'
            for i in range(len(dummy)):
                dummy.set_value(i,'file', root+'_aligned/face_det_'+str(i).zfill(6)+'.bmp')

            dummy.index=dummy['file']
            #print(dummy)
            return dummy

#        except Exception as exp:
#            print('-----------------------Problem', exp)
#            pass

#subprocess.call(
#        '/home/snjuk/OpenFace/build/bin/FaceLandmarkImg -fdir "' +imglocationsorgs + '" -wild -multi-view 1 -mloc "model/main_ceclm_general.txt" -out_dir out/virt_virthistory',
#        shell=True)
img_faces = Parallel(n_jobs=3)(
    delayed(findface)('out/virt_virthistory/', filename) for filename in os.listdir('out/virt_virthistory'))
faces = pd.concat(img_faces)
faces.to_csv('faces_virt_history.csv', sep='\t')
dfkps_meta_with_pics.to_csv('dfkps_meta_with_pics.csv', sep='\t')
dfkps_meta_with_pics.to_pickle('dfkps_meta_with_pics.p')
dffaces_meta=pd.merge(dfmeta,faces,left_on='key',right_on='oldfile', how='left')
dffaces_meta=pd.merge(dffaces_meta,dfmetapia,left_on='album_url',right_on='album_url', how='left')
dffaces_meta.to_csv('dffaces.csv','\t')
cols=[]
for col in dffaces_meta.columns:
    #print(col)
    if 'AU' in str(col):
        if 'r' in col:
            cols.append(col)
cols.append('file')
print(cols)
#dummy = dummy.ix[:, ['file', 'AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']]
dffaces_meta = dffaces_meta.dropna(axis=0, subset=[' AU01_r'])
dffaces_au = dffaces_meta[cols]
dffaces_au.set_index('file',inplace=True, drop=True)
print(len(dffaces_au))
dffaces_au[(dffaces_au != 0).all(1)]
print(len(dffaces_au))
print('dffaces', len(dffaces_au))
print('dffaces', len(dffaces_au))
cols.append('gender')
dffaces_au2 = dffaces_meta[cols]
dffaces_au2.to_pickle('dffaces_au.p')
def print_gender(dfmatrix, zeit,col):
    print(dfmatrix.columns)
    dfmatrix.fillna(-1, inplace=True)

    matrix = dfmatrix.as_matrix()
    imf = matrix
    np.shape(imf)
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
    x *= 4800

    y *= (1/np.max(  abs(y)   ))
    y += 1
    y *= 4800

    myCanvas = np.zeros((10000, 10000, 3))


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
    dfkps_meta_with_pics_auswahl=dfkps_meta_with_pics[dfkps_meta_with_pics[col]==zeit]
    for i in tqdm.tqdm(range(len(dfkps_meta_with_pics_auswahl.index.tolist()))):
        px = int(x[i])
        py = int(y[i])
        bild = imglocations + str(dfkps_meta_with_pics_auswahl.key.tolist()[i])+str(dfkps_meta_with_pics_auswahl.Person.tolist()[i])+'_kp.jpg'
        #print(bild)
        imr = cv.imread(bild, cv.IMREAD_GRAYSCALE)
        #print(dfkps_meta_with_pics.index.tolist()[i],dfkps_meta_with_pics.gender.tolist()[i])
        #print('imr',imr)
        try:
            if len(dfkps_meta_with_pics_auswahl.gender.tolist()[i])>0:
                if dfkps_meta_with_pics_auswahl.gender.tolist()[i][0]=='male':
                    imr= cv.applyColorMap(imr, cv.COLORMAP_WINTER)
                    #print('m')
                    #io.imsave("out/" + dfkps_meta_with_pics.Bild.tolist()[i], imr)
                elif dfkps_meta_with_pics_auswahl.gender.tolist()[i][0]=='female':
                    imr= cv.applyColorMap(imr, cv.COLORMAP_HOT)
                    #print('w')
        except:
            #print('Fehler bei:',dfkps_meta_with_pics.gender.tolist()[i][0])
            pass
        imr=transform.resize(imr, (128, 128, 3))
        myCanvas[px:px + 128, py:py + 128] = imr
    io.imsave('Ergebnisse_Keypoints/'+str(zeit)+'_gender_out_kps_virtual-history.jpg', myCanvas)
def print_clusters(dfmatrix, col, dfmeta,dataname):
    print(dfmatrix.columns)
    dfmatrix.fillna(-1, inplace=True)

    matrix = dfmatrix.as_matrix()
    imf = matrix
    np.shape(imf)
    import cv2 as cv

    cluster_num = 4
    km = KMeans(n_clusters=cluster_num, random_state=5)
    clusters = km.fit_predict(matrix)
    print("Silhouette score:", silhouette_score(matrix, clusters))

    #print(corpus_map)
    imap = Isomap(n_components=2)
    corpus_map = imap.fit_transform(matrix)
    #print(corpus_map)
#    plt.scatter(corpus_map[:,0],corpus_map[:,1])
    x = corpus_map[:,0].copy()
    y = corpus_map[:,1].copy()
    #z = corpus_map[:,2].copy()

    # normalise the x values to be between -1 and +1:
    x *= (1/np.max(  abs(x)   ))
    # shift them so they are between 0 and 2
    x += 1
    x *= 4800

    y *= (1/np.max(  abs(y)   ))
    y += 1
    y *= 4800
    from skimage import transform
    from skimage import io
    import cv2 as cv

    myCanvas = np.zeros((10000, 10000, 3))


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
    positions_pics={}
    for i in tqdm.tqdm(range(len(dfmeta.index.tolist()))):
        px = int(x[i])
        py = int(y[i])
        #pz = int(z[i])

        #print([type(x[i]),type(y[i]),type(int(clusters[i])),type(dfkps_meta_with_pics.gender.tolist()[i])])
        if str(dfmeta.file.tolist()[i]) != 'nan':
            positions_pics[dfmeta.key.tolist()[i].split('.')[0]]=[float(x[i]),float(y[i]),int(clusters[i]),dfmeta.gender.tolist()[i]]
            if dataname=='_kps':
                bild = imglocations + str(dfmeta.key.tolist()[i])+str(dfmeta.Person.tolist()[i])+'_kp.jpg'
            else:
                bild = os.path.join('out/virt_virthistory/',dfmeta.file.tolist()[i])
            #print(bild)
            imr = cv.imread(bild, cv.IMREAD_GRAYSCALE)
            #print(dfkps_meta_with_pics.index.tolist()[i],clusters[i])
            color= {
                0: cv.COLORMAP_AUTUMN,
                1: cv.COLORMAP_BONE,
                2: cv.COLORMAP_COOL,
                3: cv.COLORMAP_SUMMER,
                4: cv.COLORMAP_SPRING,
                5: cv.COLORMAP_OCEAN,
                6: cv.COLORMAP_PINK,
                7: cv.COLORMAP_RAINBOW,
                8: cv.COLORMAP_JET,
                9: cv.COLORMAP_HSV,
                10: cv.COLORMAP_WINTER,
                11: cv.COLORMAP_HOT,
                12: cv.COLORMAP_PARULA,
            }
            #print(clusters[i])
            #print('color:',color)
            #print(color[clusters[i]])
            #print(imr)
            imr = cv.applyColorMap(imr, color[clusters[i]])
            imr=transform.resize(imr, (128, 128, 3))
            myCanvas[px:px + 128, py:py + 128] = imr
    io.imsave(col+dataname+'_all_cluster_out_kps_virtual-history.jpg', myCanvas)
    with open('data'+dataname+'.json', 'w') as outfile:
        json.dump(positions_pics, outfile)
    with open('data2'+dataname+'.json', 'w') as outfile:
        json.dump(positions_pics, outfile,indent=4, sort_keys=True)

    for zeit in dfmeta[col].unique():
        myCanvas = np.zeros((10000, 10000, 3))
        dfmeta_auswahl=dfmeta[dfmeta[col]==zeit]
        for i in tqdm.tqdm(range(len(dfmeta.index.tolist()))):
            if len(dfmeta_auswahl[dfmeta_auswahl.index==dfmeta.index.tolist()[i]])>0:
                px = int(x[i])
                py = int(y[i])
                if dataname == '_kps':
                    bild = imglocations + str(dfmeta.Bild.tolist()[i])+str(dfmeta.Person.tolist()[i])+'_kp.jpg'
                else:
                    bild = os.path.join('out/virt_virthistory/',dfmeta.file.tolist()[i])
                #print(bild)
                imr = cv.imread(bild, cv.IMREAD_GRAYSCALE)
                #print(dfmeta.index.tolist()[i],clusters[i])
                #print('imr',imr)
                imr = cv.applyColorMap(imr, color[clusters[i]])
                imr=transform.resize(imr, (128, 128, 3))
                myCanvas[px:px + 128, py:py + 128] = imr
        io.imsave('Ergebnisse_Keypoints/'+col+'_'+str(zeit)+dataname+'_cluster_out_kps_virtual-history.jpg', myCanvas)

#dfkps_meta_with_pics=dfkps_meta_with_pics[dfkps_meta_with_pics['Verbreitungszeit']==zeit]
dfmatrix=dfkps_meta_with_pics[angles]
dfkps_meta_with_pics['file']=dfkps_meta_with_pics['key']
print(dfkps_meta_with_pics.columns)
print_clusters(dfmatrix, 'Verbreitungszeit',dfkps_meta_with_pics,'_kps')
print_clusters(dfmatrix, 'Produktionsland',dfkps_meta_with_pics,'_kps')
print_clusters(dffaces_au, 'Verbreitungszeit',dffaces_meta,'_faces')
print_clusters(dffaces_au, 'Produktionsland',dffaces_meta,'_faces')


for zeit in dfkps_meta_with_pics.Verbreitungszeit.unique():
    print(zeit)
    dfkps_meta_with_pics_auswahl=dfkps_meta_with_pics[dfkps_meta_with_pics['Verbreitungszeit']==zeit]
    dfmatrix=dfkps_meta_with_pics_auswahl[angles]
    if len(dfmatrix)>0:
        print_gender(dfmatrix,zeit,'Verbreitungszeit')
for zeit in dfkps_meta_with_pics.Produktionsland.unique():
    print(zeit)
    dfkps_meta_with_pics_auswahl=dfkps_meta_with_pics[dfkps_meta_with_pics['Produktionsland']==zeit]
    dfmatrix=dfkps_meta_with_pics_auswahl[angles]
    if len(dfmatrix)>0:
        print_gender(dfmatrix,zeit)
