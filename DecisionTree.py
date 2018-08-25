import pandas as pd  # this library is used to handle the IO and file manipulation
import sklearn as sk    # this library is used to manage the machine learning algorithms, such as training and testing them
from sklearn import model_selection
from sklearn import tree

dataset = pd.read_csv("C:\\Users\Youseef Noaman\PycharmProjects\DecisionTree\iris.csv")

Features = dataset.drop('species', axis=1)  # the features are all the columns except the species, so this command is used to save the dataset except this column
Result = dataset['species']  # this lines means that the output will only equal the species column
# this line will divide the dataset to training and testing parts of features and output
TrainingFeatures, TestingFeatures, TrainingOutput, TestingOutput = model_selection.train_test_split(Features, Result, test_size=0.20)
# those lines will determine the classifier that will be used and train it by giving it the training dataset
classifier = tree.DecisionTreeClassifier()
classifier.fit(TrainingFeatures, TrainingOutput)
# this line will use the trained model to predict the output by giving it the testing features
OutputPrediction = classifier.predict(TestingFeatures)
# this line will output the accuracy of the model by comparing the prediction output with the testing output
print("Accuracy is ", sk.metrics.accuracy_score(TestingOutput, OutputPrediction) * 100)
