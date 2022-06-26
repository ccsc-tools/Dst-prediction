This ReadMe explains the requirements and getting started to run the Dst index prediction using the Deep Learning network DSTT.

Prerequisites:

Python, Tensorflow, and Cuda:
The initial work and implementation of DSTT was done using Python version 3.9.7, Tensorflow 2.6.1 and GPU Cuda version cuda_11.4.r11.4.
Therefore, in order to run the default out-of-the-box models to run some predictions, you should use the exact version of Python and Tensorflow. 
Other versions are not tested, but they should work if you have the environment set properly to run deep learning jobs.

Python Packages:
The following python packages and modules are required to run DSTT:
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow==2.6.1
tensorflow-estimator==2.6.0
tensorflow-gpu==2.6.0
tensorflow-probability==0.14.1
numpy==1.21.5
pandas==1.3.4
keras==2.6.0
scikit-learn
sklearnmatplotlib==3.4.3
matplotlib-inline==0.1.3
seaborn==0.11.2
scipy==1.7.1

To install the required packages, you may use Python package manager "pip" as follow:
1.	Copy the above packages into a text file,  ie "requirements.txt"
2.	Execute the command:
pip install -r requirements.txt
Note: There is a requirements file already created for you to use that includes all packages with their versions. 
       The files are located in the root directory of the DSTT_Package.
Note: Python packages and libraries are sensitive to versions. Please make sure you are using the correct packages and libraries versions as specified above.

Cuda Installation Package:
You may download and install Cuda v 11.0 from https://developer.nvidia.com/cuda-11.0-download-archive

Package Structure
After downloading the zip files from github repository: https://github.com/deepsuncode/Dst-prediction the DSTT package includes the following folders and files:
 
 ReadMe.txt                    - this ReadMe file.
 requirements.txt              - includes Python required packages for Python version 3.9.7.
 models                        - directory for newly trained models. 
 default_models                - includes default trained models used during the initial work of DSTT.
 logs                          - includes the logging inforation.
 data                          - includes a list of DSTT data sets that can be used for training and prediction.
 results                       - will include the prediction result file(s).
 default_results               - will include the prediction result files from the initial work of DSTT produced by the default models.
 figures                       - will include the prediction result figures.
 default_figures               - will include the prediction result figures from the initial work of DSTT produced by the default models.
 						Note: The figures are saved as PNG files which can be viewed individually using PNG viewer in case the figures are not displayed due to any system or environment issues.
 DSTT_test.py                  - Python program to test/predict a trained model.
 DSTT_train.py                 - Python program to train a model and save it to the "models" directory.
 DSTT_plot_results_figures.py  - Python program to redraw the DSTT figures from existing predictions that exist in the "results" directory.
 Other files are included as utilities files for training and testing.
 
Running a Test/Prediction Task:
To run a test/prediction, you should use the existing data sets from the "data" directory. 
 	DSTT_test.py is used to run the test/prediction. 
Type: python DSTT_test.py :
	Without any option will test all the short term hours 1-6 hours ahead, save the prediction results, save and display the figures.

Type: python DSTT_test.py 4 :
	provide a number h-hour ahead, for example 4, to h=4 ahead hours, save the prediction results, save and display the figures.
	Available numeric options are: 1,2,3,4,5, or 6

Running a Training Task:
	DSTT_train.py is used to run the training. 
	Examples to run a training job:
	python DSTT_train.py	
	without any options to run a training job with default parameters to train all the short-term ,1-6 hours, and save them to the "models" directory.

	python DSTT_train.py 4 
	provide a number h-hour ahead, for example 4, to h=4 ahead hours, save the trained model to the "models" directory.
	Available numeric options are: 1,2,3,4,5, or 6

Running Replotting the Graphs Task:
To redraw the graphs using the predictions that are saved in the "results" directory use DSTT_plot_results_figures.py program.
 
Type: python DSTT_plot_results_figures.py :
	Without any option will re-draw all the short term hours 1-6 hours ahead, save and display the figures.

Type: python DSTT_plot_results_figures.py 4 :
	provide a number h-hour ahead, for example 4, to h=4 ahead hours, save and display the figures.
	Available numeric options are: 1,2,3,4,5, or 6
