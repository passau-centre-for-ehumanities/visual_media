# Walk through for posture analysis in pictures

## First step:
Install Detectron using this [walk through](https://github.com/passau-centre-for-ehumanities/visual_media/edit/master/howtos/install_detectron.md)

## Second step:

Start the extract_keypoints.py in the Detectron/detectron-folder with the following parameters:
--input: folder with your visual korpus
--model_cfg: specifies the yaml-file, in which the config-information for detectron are stored. there are several keypoint yamls in the /configs/12_2017_baseline/ folder. look for yamls with keypoint in the name.
--save-res: bolean, which specifies, if the keypoint-detection-results should be visualized. If =True, a folder is created, coposed of the name of the used config-yaml and the analysed folder. If you want to use more then one keypoint-detection net, you can modify the "batch_keypoint_detection.sh" to automate the process.
The algorithm produces a pickled file named after the folder, which was analyzed and the config-yaml.

## third step:
If you have more then one keypoint-detection result, you can combine the results with join_kps_detectron.py
you need to put all the results in one folder and put the name of the folder in line 53 (keypointfolder). you also need to modify the variable in line 50. put here the folder of your visual corpus.

## Fourth step:
visualize_kps.py is used to visualize the results. Lines 14-43 are used to enrich the keypoint-data with metadata and the rest of the script depends on this metadata. As this is always depending on the data, one project has, you probably have to rewrite the script acording to your needs.
