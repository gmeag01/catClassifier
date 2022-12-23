# 해당 파일은 사용하지 않았음

import os
import shutil
import pandas as pd

def checkFolders():
    if len(os.listdir('./aideeplearningproject2022/dataset/')) == 1:
        for c in classes:
            os.makedirs(os.path.join('./aideeplearningproject2022/dataset/sortData', c), exist_ok=True)

def sortData(className):
    classLabel = pd.read_csv('./aideeplearningproject2022/train.csv')
    c = classLabel[classLabel.breed == className]
    for idx in range(len(c)):
        src = os.path.join('./aideeplearningproject2022/dataset/Images', c.iloc[idx, 0])
        dst = os.path.join('./aideeplearningproject2022/dataset/sortData/', className)
        shutil.copy(src, dst)

classes = ['Devon Rex', 'Abyssinian', 'Turkish Angora', 'Ragdoll',
            'British Shorthair', 'Russian Blue', 'Birman', 'Persian',
            'Siamese', 'Bombay', 'Maine Coon', 'Bengal', 'Ameriacan Curl',
            'Munchkin', 'Egyptian Mau', 'Singapura']

try:
    checkFolders()
    for c in classes:
        sortData(c)
except Exception() as e:
    print(e)
