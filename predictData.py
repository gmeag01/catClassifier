import tensorflow as tf
import os
import numpy as np
import pandas as pd
import cv2

'''
csvPath : 테스트파일 csv 파일 경로
modelPath : 저장한 모델(h5파일) 경로
imgFolderPath : 이미지 폴더 경로로
'''

csvPath = './aideeplearningproject2022/test.csv'
modelPath = './catClassify_VGG16_spare.h5'
imgFolderPath = './aideeplearningproject2022/dataset/Images'
testCsv = pd.read_csv(csvPath, sep=',')
model = tf.keras.models.load_model(modelPath)

testX = np.array([cv2.imread(os.path.join(imgFolderPath, imgPath)) for imgPath in testCsv.file])
# testX = np.array([cv2.resize(img, (224, 224)) for img in testX])  # 모델 입력 크기 맞춰서 resizing, 필요하면 주석풀고 사용
testY = []

# 불러온 모델로 테스트 데이터 예측
for img in testX:
    img_tensor = img / 255.
    x = np.expand_dims(img_tensor, axis=0)
    pred = np.argmax(model.predict(x))
    testY.append(pred)

testY = np.array(testY)
dfY = pd.DataFrame(testY, columns=['classify'])
dfY = pd.concat([testCsv, dfY], axis=1)
dfY.to_csv('./20216710_김남혁_221224_02_59_00.csv', index=None)  # csv 파일로 저장