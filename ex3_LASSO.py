#PCS3438 - Inteligencia Artificial
#Lista de Exercícios - Prof. Eduardo Raul Hruschka
#Autor: Rafael Seiji Uezu Higa (9836878)
#------------------------------------------------------------------
#Exercício 3 - Regressão Linear com Regularização L1 (LASSO alpha = 1)

#################################################################################
import numpy as np

dataset = np.genfromtxt("reg01.csv",delimiter=",")

phiMatrix = np.zeros((np.shape(dataset)[0] - 2, 11))  #999x11
ErroQuadTraining = 0
ErroQuadValidation = 0
alpha = 1
phiMatrix[0:,0] = np.full_like(phiMatrix[0:,0],1)
temp2 = np.zeros((11,1))

for i in range(0,np.shape(dataset)[0] - 2): #dataset[i,:] é a validação!
    #Matrix phi
    phiMatrix[0:,1:] = np.delete(dataset[1:,:10], i, axis=0)   #999x11
    
    #Matriz t, excluindo-se o ruído gaussiano (observação determinística)
    tMatrix = np.delete(dataset[1:,10], i,axis=0)    #999x1
    
    #Cálculo do parametro w, na variável WML
    #Esses parametros estão dando valores MUITO discrepantes...
    w1 = np.dot(phiMatrix.T,phiMatrix)
    w2 = alpha * np.identity(11) + w1
    w3 = np.linalg.inv(w2)
    w4 = np.dot(w3,phiMatrix.T)
    WML = np.dot(w4,tMatrix)
    
    temp = tMatrix - (np.dot(WML.T, phiMatrix.T)).T   #999x1 - ((1x11) * (11x999))'
    
    temp *= temp
    WMLAbs = np.absolute(WML)
    
    ErroQuadTraining += np.sqrt(1/999*(sum(temp) + alpha * sum(WMLAbs)))

    temp2[0,0] = 1
    temp2[1:11,0] =  dataset[i+1,0:10]
    
    ErroQuadValidation += np.sqrt((np.power(dataset[i+1,10] - np.dot(WML.T,temp2)  ,  2) + alpha * sum(WMLAbs)))

ErroQuadTraining /= (np.shape(dataset)[0] - 2)
ErroQuadValidation /= (np.shape(dataset)[0] - 2)

print("RMSE do Training Set:")
print(ErroQuadTraining)
print("RMSE do Validation Set:")
print(ErroQuadValidation)