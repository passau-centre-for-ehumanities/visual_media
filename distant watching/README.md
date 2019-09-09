# Distand Watching
This is a manual for our Workflow to train and use Detectron models for your own specific research questions on a Linux OS. 

## First step:
Install Detectron using this [walk through](https://github.com/passau-centre-for-ehumanities/visual_media/edit/master/howtos/install_detectron.md)

## Second step:
Install label me from [this repository](https://github.com/wkentaro/labelme)

## Third step:
Presupposed, that you already constructed an Ontology (or other Schema) of the Symboles you want to train Detectron on, you can start now woth the anotation of the symbols in your training corpus. Our test showed, that it is neccesary to anotate at least 80 symboles to get good training results with Detectron. 

## Fourth step:
After anotating your training set, split your images and jsons into two different folders (every image need to have a sorresponding json in the other folder and vice versa) and apply our process_images script to them.
The script has the following parameters:
"--jsons": specifies the folder with the anotation files.
"--images": specifies the folder with the image files.
"--output", specifies the folder, where the output is to be saved.

This script will will split your training corpus in three different corpora: a training corpus (70% of all pictures), an evaluation corpus (15% of all pictures) and an analysis corpus (15% of all pictures). The first two are needed for the training, the last one is supposed to be an unseen test corpus to test the model.

## Sixth step:

After creating the training corpus, you need to add them to the Detectron datasets:
- Create a folder in: "Detectron/detectron/datasets/data/" named after your training corpus.
- Copy the train, val, and analysis folders and jsons into your folder.
- add the following lines to the dataset ditionary within the dataset_catalog.py file which is located in: "Detectron/detectron/datasets/" and replace NAMEOFYOURFOLDER with the actual name of your folder.
```
    'NAMEOFYOURFOLDER_train': { 
        _IM_DIR: 
            _DATA_DIR + '/NAMEOFYOURFOLDER/train_images',
        _ANN_FN: 
            _DATA_DIR + '/NAMEOFYOURFOLDER/train.json' }, 
    'NAMEOFYOURFOLDER_val': { 
        _IM_DIR: 
            _DATA_DIR + '/NAMEOFYOURFOLDER/val_images', 
        _ANN_FN: 
            _DATA_DIR + '/NAMEOFYOURFOLDER/val.json' 
    },
    'NAMEOFYOURFOLDER_analysis': { 
        _IM_DIR: 
            _DATA_DIR + '/NAMEOFYOURFOLDER/analysis_images', 
        _ANN_FN: 
            _DATA_DIR + '/NAMEOFYOURFOLDER/analysis.json' 
    },
```
## Seventh step:

Create a configuration yaml: you can use one of the many configuration yamls as boilerplait, which can be found in "Detectron/configs/" (or use the one, we added here).
The most important lines, which have to be adjusted are:
NUM_CLASSES: the number of your classes plus one (beckground class)
NUM_GPUS: the number of available GPUs
TRAIN/DATASETS: the name of your train dataset, which you added to the dataset_catalog.py. It should look like that: ('NAMEOFYOURFOLDER_train',)
TEST/DATASETS: same as TRAIN/DATASETS, only with the name of your test dataset in the dataset_catalog.py
MAX_ITER: the amount of maximal iterations of the training algorithm. Wich number is the best to choos depents on a number of variables.
OUTPUT_DIR: add here the folder where the results are to be safed.

## Eighth step

Start training within the Detectron folder with comand line:
```
python2 tools/train_net.py --cfg YOURCONFIG.yaml
```
## Ninth step 
After producing a model, you can it apply with this commands on your corpus:

In case you have a video corpus, use this python script:
```
run_on_frames.py --video_folder "FOLDERS/OF/VIDEOFRAMEFOLDERS" --model-cfg YOURCONFIG.yaml --outputdir FOLDER/OF/VIDEOOUTPUTFOLDERS --output YOUROUTPUTFILE.json

```
This algorithm requires, that you first extracted the frames from the videos and saved all video frames of a video in one folder:

./OUTPUTFOLDER/VIDEO1.mp4/000000001.jpg

./OUTPUTFOLDER/VIDEO1.mp4/000000002.jpg

./OUTPUTFOLDER/VIDEO1.mp4/000000003.jpg

:

./OUTPUTFOLDER/VIDEO2.mp4/000002532_0.png

./OUTPUTFOLDER/VIDEO2.mp4/000003019_0.png

./OUTPUTFOLDER/VIDEO2.mp4/000003019_1.png

## Tenth step
```
python2 run_on_images.py --input /FOLDER/OF/YOUR/IMAGES --model-cfg ../YOUCONFIG.yaml --save-res=True
```
--safe-res specifies, if the results are to be visualized or not.

