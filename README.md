# Robosec
This package integrate face recognition and speech recognition module on ros platform.

# Development Environment:
- Ubuntu 18.04
- Ros Melodic
- Python>=3.10.0 environment

# Prerequisites:
## Install Anaconda:
### 1. First download the corresponding installation package [Anaconda](https://www.anaconda.com/download#linux)
### 2. Then install anaconda (for example, the version may vary)
``` $ bash ~/Downloads/Anaconda3-2021.05-Linux-x86_64.sh ```
### 3. Find the .bashrc file 
Go to the home folder and press ctrl+h
### 4. Edit the ~/.bashrc file and add the following PATH export to the end of the file
``` export PATH=/home/mustar/anaconda3/bin:$PATH ```
### 5. Source the ~/.bashrc file after saving and exit terminal
``` $ source ~/.bashrc ``` <br><br>

## Create virtual environments for python with conda:
### 1. Check conda is installed and in your PATH
1. Open a terminal
2. Enter ``` conda -V ``` into the terminal command line and press enter.
3. If conda is installed you should see somehting like the following.
```
$ conda -V
conda 3.7.0
```
### 2. Create a virtual environment 
``` $ conda create -n yourenvname python=3.10 anaconda ```
### 3. Activate your virtual environment
``` $ source activate yourenvname ``` <br><br>

## Install the required library
``` 
$ source activate yourenvname
$ pip install -r requirements.txt
```
### Install the speechbrain.inference librart
```
pip install git+https://github.com/speechbrain/speechbrain.git@develop 
pip install pydub noisereduce
pip install ffmpeg-python
```


## Steps to run this project
### 1. First, open a new terminal and type the following command in the terminal:
``` $ roslaunch usb_cam usb_cam-test.launch ```
### 2. Open a new terminal and type the following command to run the face_recognition_node:
```
$ source activate yourenvname
$ source devel/setup.bash
$ rosrun robot_pkg face_recognition_node.py
```
### 3. Open a new terminal and type the following command to run the voice_recognition_node:
```
$ source activate yourenvname
$ source devel/setup.bash
$ rosrun robot_pkg voice_recognition_node.py
```
