import tensorflow as tf
import os
import cv2
import numpy as np
import pandas as pd

csvPath = './aideeplearningproject2022/test.csv'
modelPaht = './catClassify_Xception.h5'
imgFolderPath = './aideeplearningproject2022/dataset/Images'
testCsv = pd.read_csv(csvPath, sep=',')
model = tf.keras.models.load_model(modelPaht)

testX = np.array([cv2.imread(os.path.join(imgFolderPath, imgPath)) for imgPath in testCsv.iloc[:, 0]])
testX = np.array([cv2.resize(img, (150, 150)) for img in testX])
testY = np.array([])

for img in testX:
    x = np.reshape(img, (1, ) + img.shape)
    pred = model.predict(x).argmax()
    testY = np.append(testY, pred)

dfY = pd.DataFrame(testY, columns=['classify'])
dfY = pd.concat([testCsv, dfY], axis=1)
dfY.to_csv('./20216710_김남혁_221213_03_32_00.csv', index=None)

