# Capstone-LEGOLAS

## Contributors
This is a CS 496 Capstone Project at Loyola University Maryland.
#### Students:
Clark, Brendan

Versteegen, Audrey

#### Client:
Dr. Mary Lowe, Professor at Loyola Universtity Maryland 

## Project Description
The LEGO-based Low-cost Autonomous Scientist (LEGOLAS) is a low cost solution to the time consuming nature of physical science research. LEGOLAS combines machine learning and robotics to predict and validate pH corrolations between solutions. There is, however a little to be said about the robot's navigation. Our client aquired a LEGOLAS robot from the University of Maryland   

## Insatlation Instructions
dataset can be found at https://drive.google.com/file/d/1pEE_Kq6wtxsWpd7Atc1lKsCQilUrcppn/view?usp=drive_link
Trained model can be found at https://studentsloyola-my.sharepoint.com/:u:/g/personal/arversteegen_loyola_edu/EYWHuGRhSgVLg4ie0CfrDBUBJZzpZamqoQdZceh2qHtEsA?e=CobIiI 
## How to Run
### Step 1:
1. download our legolas files from our github
2. download anaconda from https://www.anaconda.com/products/individiual

### Step 2:
Launch anaconda prompt and type the following:
1. conda create -n python_3.12.3 python=3.12.3 (type "y" when it asks to proceed)
2. conda activate python_3.12.3
3. python -m pip install ipykernel

### Step 3:
1. Open jupyter notebook
2. locate where the legolas files are and open "LegolasDemo.ipynb" under the "LEGOLAS Scripts" folder
2. on the top bar select "kernel" -> "change kernel" -> "py3.12.3-kernel"

### Step 4:
In your anaconda prompt type the following commands:
1. pip install opencv-python
2. pip install pyyaml
3. pip install rpyc
4. pip install matplotlib
5. pip install scipy
6. pip install torchvision
7. pip install paramiko

### Step 5:
1. make sure you are on the same network as the legolas
2. run the 2nd cell in the jupyter notebook
3. after that's done run any of the cells to test the legolas

#### Note:
if you are to use manual.py instead of the jupyter notebook make sure that before you launch manual.py you type in "conda activate python_3.12.3" to use the enviornment we just setup

## How to Test
