#PCS3438 - Inteligencia Artificial
#Lista de Exercícios - Prof. Eduardo Raul Hruschka
#Autor: Rafael Seiji Uezu Higa (9836878)
#------------------------------------------------------------------
#Exercício 2 - KNN

#################################################################################
#Ler arquivo
import numpy as np

dataset = np.genfromtxt("class02.csv",delimiter=",")

datasetSize = np.shape(dataset)[0] - 1
testSetSize = int(datasetSize / 5)
trainingSetSize = datasetSize - testSetSize

trainingSet = np.zeros((trainingSetSize, 101))
testSet = np.zeros((testSetSize, 101))
normSet = np.zeros(trainingSetSize)
acc = np.zeros(5)
normElements = np.zeros((trainingSetSize,100))

for i in range(0,5):
    testSet = dataset[1 + i * testSetSize : (i + 1) * testSetSize + 1][:]

    a = dataset[1 : (i * testSetSize) + 1][:]
    b = dataset[1 + (i+1) * testSetSize : datasetSize + 1][:]
    trainingSet = np.concatenate((a,b),axis=0)
    
    for j in range(0,testSetSize):   #varrendo pra cada exemplo
        normElements[:,0:] = testSet[j,0:100]
        
        normSet = np.linalg.norm(normElements - trainingSet[:,0:100],axis=1)
    
        normIndex = np.argsort(normSet)
        
        votes = trainingSet[normIndex[0:10],100]  #computo das classes

        election, oi = np.histogram(votes, bins=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  #computo do resultado da eleição

        result = np.argmax(election)

        if votes[result] == testSet[j][100]:
            acc[i] += 1
        
    print(acc[i] / testSetSize)


average = sum(acc/testSetSize)/5*100
print("\nacurácia média:\n%.2f" % average, "%")