import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv(r"car_evaluation.csv")

dataframe.columns = ['buying','maint','doors','persons','lug_boot','safety','classes']
dataframe.buying.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
dataframe.maint.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
dataframe.doors.replace(('2','3','4','5more'),(1,2,3,4), inplace=True)
dataframe.persons.replace(('2','4','more'),(1,2,3), inplace=True)
dataframe.lug_boot.replace(('small','med','big'),(1,2,3), inplace=True)
dataframe.safety.replace(('low','med','high'),(1,2,3), inplace=True)
dataframe.classes.replace(('unacc','acc','good','vgood'),(1,2,3,4), inplace=True)

dataset = dataframe.values
X = dataset[:,0:6]
Y = numpy.asarray(dataset[:,6], dtype="str")

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X_Train, Y_Train)

predicted = clf.predict(X_Train)

error = 1 - accuracy_score(Y_Train, predicted)
print('Error: ',error)

try:
    filename = 'model_trained.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Modelo entrenado creado correctamente...")
except pickle.PickleError:
  print("Error al crear el modelo entreado")

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.30, random_state=5)
test_data = np.vstack((X_Test[:,0], X_Test[:,1], X_Test[:,2], Y_Test))
np.savetxt("test_data.txt",X_Test)