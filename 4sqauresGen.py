IMAGESIZE = 10
BLOCKSIZE = 2
STEPINCREMENT = 2
ROUND = 0
DIR = "ImageSize_"+str(IMAGESIZE)+"_BlockSize_"+str(BLOCKSIZE)+"_Step_"+str(STEPINCREMENT)
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import os

fig1 = plt.figure()

ax1 = fig1.add_subplot(111, aspect='equal')

if not os.path.exists(DIR):
   
    os.mkdir(DIR)
plt.xlim(-IMAGESIZE, IMAGESIZE)
plt.ylim(-IMAGESIZE, IMAGESIZE)

def addRectangle(x,y):
    ax1.add_patch(
            patches.Rectangle(
                (x , y),   # (x,y)
                BLOCKSIZE,          # width
                BLOCKSIZE,          # height
            )
        )

for block1 in iter(np.arange(-IMAGESIZE, IMAGESIZE, STEPINCREMENT)):
    for block2 in iter(np.arange(-IMAGESIZE, IMAGESIZE, STEPINCREMENT)):
        for block3 in iter(np.arange(-IMAGESIZE, IMAGESIZE, STEPINCREMENT)):
            for block4 in iter(np.arange(-IMAGESIZE, IMAGESIZE, STEPINCREMENT)):
        


                ax1.cla()
                addRectangle(block1, 0)
                addRectangle(block2, 0)
                addRectangle(0, block3)
                addRectangle(0, block4)
                block1, block2, block3, block4 =  np.round(block1,ROUND), np.round(block2,ROUND), np.round(block3,ROUND), np.round(block4,ROUND)
                
                fileName = str(block1)+"_"+str(block2)+"_"+str(block3)+"_"+str(block4)
                fig1.savefig(DIR+"/"+fileName+".png")

