# Workflow How to use OpenFace for Facial Behaviour

## First step:
Install OpenFace according to the installation [instructions](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation)

## Second step:
Edit and execute the script face_analysis.py. 
You need to edit the variable "imagefolder" in line 54. This is the folder, where your images are stored.
Please, keep in mind, that this script is designed, to integrate the output of the facial analysis with metadata from another table (line 81-98). We let our solution in the source text but this bit should probably be solved individually. Please, open an issue, if you have trouble to do so, we are happy toi assist.
If used properly, the script should produce 3 different visualizations:
* Output_facedirection.jpg: is a visualization of the corpus according to facial direction.
* Output_faceemotion.jpg: is a visualization of the corpus according to facial behaviour (float values)
* Output_faceemotion_bool.jpg: is a visualization of the corpus according to facial behaviour (bool values)
