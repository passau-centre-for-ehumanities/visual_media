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
