#Q1_Q2_Q3
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
import sklearn.discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


mean1 = [4,5,4,5,4,5,4,5,4,5]
mean2 = [-4,-5,-4,-5,-4,-5,-4,-5,-4,-5]

np.random.seed(42)
covpre1 = 5 + 40 * np.random.randn(10, 10)
cov1 = (np.dot(covpre1, covpre1.transpose()))
np.random.seed(42)
covpre2 = 20 + 20 * np.random.randn(10, 10)
cov2 = (np.dot(covpre2, covpre2.transpose()))

np.random.seed(42)
x1 = np.random.multivariate_normal(mean1,cov1, 1000)
np.random.seed(42)
x2 = np.random.multivariate_normal(mean2,cov2, 1000)

X = np.concatenate((x1,x2))
Xc = np.ones(1000)
Xc = np.concatenate((Xc, np.zeros(1000)))

XTrain, XTest , XcTrain, XcTest = train_test_split(X,Xc, test_size=0.2, stratify=Xc)

XTrainMC = X - np.mean(XTrain)


D,E = np.linalg.eig(np.dot(XTrainMC.T,XTrainMC))
sortedD = np.argsort(D)[::-1]
sortedE = E[:, sortedD]

meanSquareError = np.zeros(5,)
classificationError = np.zeros(6,)
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

for i in range(6):
    YTrain = np.dot(XTrain,sortedE[:,0:(10-i)])
    YTest = np.dot(XTest,sortedE[:,0:(10-i)])
    eReduced = E[:,0:(10-i)]
    lda.fit(YTrain,XcTrain)
    prediction = lda.predict(YTest)
    error = sum(abs(prediction - XcTest))
    classificationError[i] = (error/YTest.shape[0]) * 100
    if(i>0):
        XTrainRecon = np.dot(YTrain,eReduced.T)
        meanSquareError[i-1] = mean_squared_error(XTrain,XTrainRecon)
print(classificationError)
print(meanSquareError)
plt.plot(range(1,6), meanSquareError, label = 'Mean Square Error')
plt.legend()
plt.show()
plt.plot(range(1,7), classificationError, label = 'Classification Error(%)')
plt.ylim([0,100])
plt.legend()
plt.show()

