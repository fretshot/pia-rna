import numpy
import pandas
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Cargando muestra de datos
dataframe = pandas.read_csv(r"car_evaluation.csv")

# Agregando nombres a las columnas
dataframe.columns = ['buying','maint','doors','persons','lug_boot','safety','classes']

# Remplazando las etiquetas por numeros
dataframe.buying.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
dataframe.maint.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
dataframe.doors.replace(('2','3','4','5more'),(1,2,3,4), inplace=True)
dataframe.persons.replace(('2','4','more'),(1,2,3), inplace=True)
dataframe.lug_boot.replace(('small','med','big'),(1,2,3), inplace=True)
dataframe.safety.replace(('low','med','high'),(1,2,3), inplace=True)
dataframe.classes.replace(('unacc','acc','good','vgood'),(1,2,3,4), inplace=True)

#print("dataframe.head: ", dataframe.head())
#print("dataframe.describe: ", dataframe.describe())

plt.hist((dataframe.classes))
dataframe.hist()
dataset = dataframe.values

X = dataset[:,0:6]
Y = numpy.asarray(dataset[:,6], dtype="str")

# Spliteando los datos en train y test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)

# Se entrena el clasificador
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X_Train, Y_Train)

# Se predice
predicted = clf.predict(X_Train)

# Calculando el error
error = 1 - accuracy_score(Y_Train, predicted)
print('Error: ',error)

try:
    # el modelo entrenado se guarda en un archivo
    filename = 'model_trained.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Modelo entrenado creado correctamente...")
except pickle.PickleError:
  print("Error al crear el modelo entreado")

# Datos de entrada para la prueba se guardan en un archivo
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.30, random_state=5)
test_data = np.vstack((X_Test[:,0], X_Test[:,1], X_Test[:,2], Y_Test))
np.savetxt("test_data.txt",X_Test)