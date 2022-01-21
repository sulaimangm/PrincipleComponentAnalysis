#Here we will perform Principal Component Analysis on a self generated random Data Set We will use SKLearn's in-built Library for LDA for performing classification

#Importing Libraries
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
import sklearn.discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Means for each of the 10 attributes of the 2 classes.
mean1 = [4,5,4,5,4,5,4,5,4,5]
mean2 = [-4,-5,-4,-5,-4,-5,-4,-5,-4,-5]

#Generating Covariance matrices for the 2 classes
np.random.seed(42)
covpre1 = 5 + 40 * np.random.randn(10, 10)
cov1 = (np.dot(covpre1, covpre1.transpose()))
np.random.seed(42)
covpre2 = 20 + 20 * np.random.randn(10, 10)
cov2 = (np.dot(covpre2, covpre2.transpose()))

#Generating randon data from the covariance matrices
np.random.seed(42)
x1 = np.random.multivariate_normal(mean1,cov1, 1000)
np.random.seed(42)
x2 = np.random.multivariate_normal(mean2,cov2, 1000)

#Combining the Data and class values for the 2 classes
X = np.concatenate((x1,x2))
Xc = np.ones(1000)
Xc = np.concatenate((Xc, np.zeros(1000)))

#Splitting Data into Training and Testing Set with a 80:20 Split
XTrain, XTest , XcTrain, XcTest = train_test_split(X,Xc, test_size=0.2, stratify=Xc)

#Mean Centering the Training Data
XTrainMC = X - np.mean(XTrain)

#Calculating Principal Components
D,E = np.linalg.eig(np.dot(XTrainMC.T,XTrainMC)) #Calculating Eigen Values and Eigen Vectors
sortedD = np.argsort(D)[::-1] #Sorting Eigen Values in Decending order
sortedE = E[:, sortedD] #Sorting Eigen Vectors on the bais or the sorted Eigen Values

#Initializing MeanSquareError and Classification Error
meanSquareError = np.zeros(5,)
classificationError = np.zeros(6,)

#Creating Object for LDA
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

#This loop will first reduce dimentions of the sorted Eigen Vectors and then multiply it with the dataset to bring it into the reduced principal component space. But for the
#first loop dimentions will not reduce as i is 0. After reducing the dimensions and bringing it into the principal component space, we will calculate the classification error
#and then reconstruct the reduced data. We will then calculate the mean square error(MSE) of the reconstructed data. MSE will give you the variance between original data and
#reconstructed data.
for i in range(6):
    YTrain = np.dot(XTrain,sortedE[:,0:(10-i)]) #Bring the training data into the principal component space while reducing dimensions
    YTest = np.dot(XTest,sortedE[:,0:(10-i)]) #Bring the testing data into the principal component space while reducing dimensions
    eReduced = E[:,0:(10-i)] #Reducing dimensions of the Eigen Vectors
    lda.fit(YTrain,XcTrain) #Fitting the LDA Model
    prediction = lda.predict(YTest) #Performing Prediction on the model 
    error = sum(abs(prediction - XcTest)) #Calculating number of errors
    classificationError[i] = (error/YTest.shape[0]) * 100 #Calculating percentage of errors
    if(i>0):
        XTrainRecon = np.dot(YTrain,eReduced.T) #Reconstruction Data
        meanSquareError[i-1] = mean_squared_error(XTrain,XTrainRecon) #Calculating Mean Square Error on Reconstructed Data
print(classificationError)
print(meanSquareError)

#Plotting the Mean Square error and the classification error
plt.plot(range(1,6), meanSquareError, label = 'Mean Square Error')
plt.legend()
plt.show()
plt.plot(range(1,7), classificationError, label = 'Classification Error(%)')
plt.ylim([0,100])
plt.legend()
plt.show()

