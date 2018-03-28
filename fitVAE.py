from os import listdir
import os
from os.path import isfile, join
import pickle
import cv2
import numpy as np
import vae
import tensorflow as tf
from matplotlib import pyplot as plt
I = 6 #ImageSize
DIR = "ImageSize_"+str(I)+"_BlockSize_2_Step_2_black"
path = os.getcwd()+"/"+DIR+"/"


imgList = []
print("LOADING DATA......")
flag = True
for i in range(-I, I, 2):
    print("Currently loading i="+"\t"+str(i))
    
    for j in range(-I, I, 2):
        for k in range(-I, I, 2):
            for l in range(-I, I, 2):
                

                img = cv2.imread(path+str(i)+"_"+str(j)+"_"+str(k)+"_"+str(l)+".png")

                if flag:
                    origShape = np.shape(img)
                    flag = False
                img = np.reshape(img, -1)
                
                imgList.append(img/255)
               
print("LOADING COMPLETE")

dim_img = np.shape(imgList)[1]



print("PROCESSING IMAGELIST")
imgList = np.array(imgList)
imgListMeans = np.array([np.mean(imgList[:,i]) for i in range(dim_img)])
imgListStds = np.array([np.std(imgList[:,i]) for i in range(dim_img)]) + 1e-8*np.ones(dim_img)

Data = np.column_stack([ (imgList[:,i] - imgListMeans[i])/imgListStds[i] for i in range(dim_img)])
#Data = imgList/255 (This is without normalization)
np.random.shuffle(Data)
#print("STORING IMAGE DATA")
#fobj = open(DIR+"_DATA.pkl", "ab")
#for i in Data:
#    pickle.dump(Data[i], fobj)
#fobj.close()




numTrainPoints = int(0.8*np.shape(Data)[0])
numValPoints = int(0.2*np.shape(Data)[0])
trainSet = Data[:numTrainPoints]
valSet = Data[numTrainPoints:]


#default args
n_hidden = 50
dim_z = 20
keep_prob = 0.9
learn_rate = 1e-3
batchSize = 100

 
x = tf.placeholder(tf.float32, shape=[None, dim_img], name='img')

# dropout
#keep_prob = tf.placeholder(tf.float32, name='keep_prob')



# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x, x, dim_img, dim_z, n_hidden, keep_prob)

# optimization
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _iter in range(100):
        print("*******ITERATION"+"\t"+str(_iter)+"********")
        
        trainX = [trainSet[i] for i in np.random.choice(np.arange(0, numTrainPoints), batchSize, replace = False)]
        valX = [valSet[i] for i in np.random.choice(np.arange(0,numValPoints), batchSize, replace = False)]

        
        train_loss = sess.run(loss, feed_dict={x:trainX})
        val_loss = sess.run(loss, feed_dict={x:valX})
        _,  loss_likelihood, loss_divergence = sess.run(
                        (train_op,  neg_marginal_likelihood, KL_divergence),
                        feed_dict={x: trainX})
        print("Train Loss \t"+str(train_loss))

        print("Val Loss \t" +str(val_loss))
        print("neg_ML \t" +str(loss_likelihood))
        print("KL \t" +str(loss_divergence))

    print("Training Completed")

    for i in range(100):
        
        z = tf.random_normal(np.array([1,dim_z]), 0, 1, dtype=tf.float32)
        y = vae.decoder(z, dim_img, n_hidden)

        img = np.reshape(((sess.run(y)*imgListStds)+imgListMeans)*255, origShape)
        #img = np.reshape(sess.run(y)*255, origShape)
        
        
        cv2.imwrite("genImages/image"+str(i)+".png", img)

    print("Generation Completed")

    fobj =open(DIR+"_Record.pkl", "rb")
    SizeDict = pickle.load(fobj)
    KLHistory = []
    fileSizes = []
    
    for key in sorted(SizeDict.keys()):


        imList = []
        dataFiles = [f +".png" for f in SizeDict[key]]
        for _file in dataFiles:
            img = cv2.imread(path+_file)
            img = np.reshape(img, -1)
            imList.append(img/255)
        imList = np.array(imList)
        
        imData = np.column_stack([ (imList[:,i] - imgListMeans[i])/imgListStds[i] for i in range(dim_img)])
        fileSizes.append(key)
        KLHistory.append(sess.run(KL_divergence, feed_dict={x:imData}))

        print(fileSizes)
   
        plt.plot(fileSizes, KLHistory)
        plt.savefig("KL_vs_FileSize.png")