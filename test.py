import pickle
import numpy as np

# Leyendo X
x = np.loadtxt("test_data.txt")

# Escogiendo un numero aleatorio
ex = np.random.randint(0,500)

# Cargando la red
loaded_model = pickle.load(open("model_trained.sav","rb"))

# IDentificacion del tipo de carro
xt = x[ex,:].reshape(1,-1)
predicted = loaded_model.predict(xt)
print("Las caracteristicas del auto son: ",xt)
print("Se predice que el auto es: ", predicted) 