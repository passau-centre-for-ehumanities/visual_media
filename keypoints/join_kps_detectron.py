import pickle
import cv2 as cv
from sklearn.manifold import Isomap

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dpi=200
keypoints = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]
kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]

index = np.arange(0, 0)
dfkps = pd.DataFrame(index=index, columns=[],)

imgfolder='/home/erik/Zigarettenbilder/scraper/images/Virtual-History/'
dfkps['Bildbreite']=-1
dfkps['Bildlänge']=-1
keypointfolder='ergebnisse_keypoints_detectron'
print(dfkps.columns)
for file in os.listdir(keypointfolder):
    print(file, len(dfkps))
    if file.endswith('dfkps.p'):
        dfkps_neu = pd.read_pickle(os.path.join(keypointfolder , file))
        dfkps_neu[14] = np.nan
        dfkps_neu[15] = np.nan
        dfkps_neu[16] = np.nan
        dfkps_neu_x = dfkps_neu[dfkps_neu.Achse == 'x']
        if len(dfkps)>0:
            for linie in dfkps_neu_x.itertuples():
                #print('-------------------',linie.Bild)
                test=dfkps_neu[dfkps_neu.Bild==linie.Bild]
                oriImg = cv.imread(imgfolder + linie.Bild)  # B,G,R order
                toleranzwert=max(oriImg.shape)/100*3
                print(oriImg.shape)
                dfind = dfkps[(dfkps.Bild == linie.Bild) & (dfkps.Achse == 'x')]
                print(dfind,linie)
                dfkps.loc[dfind.index, 'Bildbreite']= oriImg.shape[0]
                dfkps.loc[dfind.index, 'Bildlänge']= oriImg.shape[1]

                #print(str(test.Person.max()))
                if str(test.Person.max()) != 'nan':
                    if test.Person.max() > 0:
                        iterlist=range(test.Person.max())
                    elif test.Person.max() == 0:
                        iterlist=[0]
                    for i in iterlist:
                        testperson=test[test.Person==i]
                        testperson.set_index('Achse',inplace=True)
                        test2=dfkps[(dfkps.Bild==linie.Bild)]
                        if test2.Person.max()>0:
                            iterlist2=range(test2.Person.max())
                        elif test2.Person.max()==0:
                            iterlist2=[0]
                        for z in iterlist2:
                            testperson2=test2[test2.Person==z]
                            testperson2.set_index('Achse',inplace=True)
                            if (len(testperson)>0) and (len(testperson2)>0):
                                maxabw=0
                                for col in testperson2:
                                    if col not in ['Bild','Achse','Person','Bildbreite','Bildlänge']:
                                        pers1x=testperson.get_value('x',col)
                                        pers1y=testperson.get_value('y',col)
                                        pers2x=testperson2.get_value('x',col)
                                        pers2y=testperson2.get_value('y',col)
                                        if (str(pers1x)!='nan')and (str(pers1y)!='nan') and (str(pers2x)!='nan') and (str(pers2y)!='nan'):
                                            #print(pers1x,pers1y,abs(pers1x-pers2x))
                                            #print(pers2x, pers2y,abs(pers1y-pers2y))
                                            xpers=abs(pers1x-pers2x)
                                            ypers=abs(pers1y-pers2y)
                                            if xpers>maxabw:
                                                maxabw=xpers
                                            if ypers>maxabw:
                                                maxabw=ypers
                                #print('z=',z, 'i= ',i, linie.Bild,maxabw)
                                if (maxabw<toleranzwert):
                                    #print('in ',linie.Bild, ' ist detectron p ',z,' mit multi person ', i,' identisch',testperson.columns)
                                    for spalte in testperson2:
                                        for zeile in ['x','y']:
                                            if spalte not in ['Bild','Achse','Person','Bildbreite','Bildlänge']:
                                                #print(testperson2.get_value(zeile,spalte),testperson.get_value(zeile,spalte))
                                                if str(testperson2.get_value(zeile,spalte)) == 'nan':
                                                    if str(testperson.get_value(zeile,spalte)) != 'nan':
                                                        print('Hier könnte man den Wert',spalte,' ersetzen')
                                                        dfind=dfkps[(dfkps.Bild==linie.Bild)&(dfkps.Person==z)&(dfkps.Achse==zeile)]
                                                        dfkps.set_value(dfind.index[0],spalte,testperson.get_value(zeile,spalte))
                                                        dfkps.to_csv('KPS_neu.csv', sep='\t')
                                                        cmap = plt.get_cmap('rainbow')
                                                        colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
                                                        im = cv.cvtColor(oriImg, cv.COLOR_BGR2RGB)
                                                        fig = plt.figure(frameon=False)
                                                        fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
                                                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                                                        ax.axis('off')
                                                        fig.add_axes(ax)
                                                        ax.imshow(im)
                                                        plt.autoscale(False)

                                                        for l in range(len(kp_lines)):
                                                            i1 = kp_lines[l][0]
                                                            i2 = kp_lines[l][1]
                                                            #print('!-!',i1,i2)
                                                            dfpers=dfkps[(dfkps.Bild == linie.Bild) & (dfkps.Person == z)]
                                                            dfpers.set_index('Achse', inplace=True)
                                                            if str(dfpers.get_value('x',i1))!='nan' and str(dfpers.get_value('x',i2))!='nan':
                                                                x = [dfpers.get_value('x',i1), dfpers.get_value('x',i2)]
                                                                y = [dfpers.get_value('y',i1), dfpers.get_value('y',i2)]
                                                                line = plt.plot(x, y)
                                                                plt.setp(line, color=colors[l], linewidth=2.0, alpha=0.7)
                                                            if str(dfpers.get_value('x', i1)) != 'nan':
                                                                plt.plot(
                                                                    dfpers.get_value('x', i1), dfpers.get_value('y',i1), '.', color=colors[l],
                                                                    markersize=3.0, alpha=0.7)
                                                            if str(dfpers.get_value('x', i2)) != 'nan':
                                                                plt.plot(
                                                                    dfpers.get_value('x', i2), dfpers.get_value('y',i2), '.', color=colors[l],
                                                                    markersize=3.0, alpha=0.7)
                                                        if str(dfpers.get_value('x', keypoints.index('right_shoulder'))) != 'nan' and str(dfpers.get_value('x', keypoints.index('left_shoulder'))) != 'nan':
                                                            #print(dfpers.get_value('x', keypoints.index('right_shoulder')))
                                                            #print(dfpers.get_value('x', keypoints.index('left_shoulder')))
                                                            x = [(dfpers.get_value('x', keypoints.index('right_shoulder'))+dfpers.get_value('x', keypoints.index('left_shoulder')))/2.0, dfpers.get_value('x', keypoints.index('nose'))]
                                                            y = [(dfpers.get_value('y', keypoints.index('right_shoulder'))+dfpers.get_value('y', keypoints.index('left_shoulder')))/2.0, dfpers.get_value('y', keypoints.index('nose'))]
                                                            line = plt.plot(x, y)
                                                            plt.setp(line, color=colors[len(kp_lines)], linewidth=2.0, alpha=0.7)
                                                            if str(dfpers.get_value('x', keypoints.index('right_hip'))) != 'nan' and str(dfpers.get_value('x', keypoints.index('left_hip'))) != 'nan':
                                                                x = [(dfpers.get_value('x', keypoints.index('right_shoulder')) +
                                                                      dfpers.get_value('x', keypoints.index('left_shoulder'))) / 2.0,
                                                                     (dfpers.get_value('x', keypoints.index('right_hip')) +
                                                                      dfpers.get_value('x', keypoints.index('left_hip'))) / 2.0]
                                                                y = [(dfpers.get_value('y', keypoints.index('right_shoulder')) +
                                                                      dfpers.get_value('y', keypoints.index('left_shoulder'))) / 2.0,
                                                                     (dfpers.get_value('y', keypoints.index('right_hip')) +
                                                                      dfpers.get_value('y', keypoints.index('left_hip'))) / 2.0]
                                                                line = plt.plot(x, y)
                                                                plt.setp(
                                                                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                                                                    alpha=0.7)

                                                        fig.savefig('ergebnisse_keypoints_detectron/Bilder/'+linie.Bild+'_besser.jpg', dpi=dpi)
                                                        plt.close('all')
        else:
            dfkps=dfkps_neu
print(dfkps)
dfkps.to_pickle('dfkps_all.p')
dfkps.to_csv('dfkps_all.csv', sep='\t')
