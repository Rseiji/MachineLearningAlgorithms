#PCS3438 - Inteligencia Artificial
#Lista de Exercícios - Prof. Eduardo Raul Hruschka
#Autor: Rafael Seiji Uezu Higa (9836878)
    
#------------------------------------------------------------------
#Exercício 1 - Naive Bayes

#################################################################################

import numpy as np
import math

dataset = np.genfromtxt("class01.csv",delimiter=",")

trainingNumber = 350
featureNumber = np.shape(dataset)[1]-1  #  =100

trainingSet = dataset[1:trainingNumber+1,:]  #350 exemplos de treino
testSet = dataset[trainingNumber+1:,:]       #o restante: 750 exemplos

#Matrizes auxiliares para executar os cálculos
featureGroups       = np.zeros(   (2, 10, featureNumber)   )
gaussianDenominator = np.zeros((10,featureNumber))  #10 x 100
gaussianExponent    = np.zeros((10,featureNumber))
auxMat              = np.zeros((10,100))
labelCounter        = np.zeros(10)
probbes             = np.zeros(10)


for i in range(0,10):
    classFilter = dataset[1:trainingNumber + 1,100] == i
    labelCounter[i] = sum(dataset[1:trainingNumber + 1,100] == i)
    
    for j in range(0,featureNumber):
        featureGroups[0,i,j] = np.mean(trainingSet[classFilter,j])
        featureGroups[1,i,j] = np.std(trainingSet[classFilter,j])

labelCounter /= trainingNumber

for counter in range(0,2):
    accTest = 0

    if counter == 0:
        exampleAmount = 350    
    else:
        exampleAmount = np.shape(testSet)[0]

    for k in range(0,exampleAmount):
        prob = np.ones( (10,100) )

        gaussianDenominator[:,:] = math.sqrt(2 * math.pi) * featureGroups[1,:,:]
        
        if counter == 0:
            auxMat[:] = np.copy(trainingSet[k,:100].T)
        else:
            auxMat[:] = np.copy(testSet[k,:100].T)
        
        gaussianExponent[:,:] = (-1) * np.power(auxMat[:,:] - featureGroups[0,:,:], 2) / (2 * pow(featureGroups[1,:,:],2))
        
        prob[:,:] *= 1 / gaussianDenominator[:,:] * np.exp(gaussianExponent[:,:]) #* labelCounter[:,:] /sum(prob[:,:])
        
        probbes = np.prod(prob,axis=1) * labelCounter
        temp = sum(probbes)
        probbes = probbes / temp
        
        if counter == 0:
            if trainingSet[k,100] == np.argmax(probbes):
                accTest += 1
        else:
            if testSet[k,100] == np.argmax(probbes):
                accTest += 1

    if counter == 0:
        print("Acurácia do conjunto de treino:")
    else:
        print("Acurácia do conjunto de validação:")
    
    temp2 = accTest / exampleAmount * 100
    print("%.2f" % temp2, "%")
    