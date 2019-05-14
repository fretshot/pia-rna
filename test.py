import pickle
import numpy as np

# Leyendo muestra de carros
x = np.loadtxt("test_data.txt")

# Escogiendo un numero aleatorio de auto
ex = np.random.randint(0,500)

# Cargando la clasificador entrenado
loaded_model = pickle.load(open("model_trained.sav","rb"))

# Identificacion del tipo de carro
xt = x[ex,:].reshape(1,-1)
predicted = loaded_model.predict(xt)

print("Las caracteristicas del auto escogido son: ",xt)
if("1" in predicted):
    print("Se predice que el auto es inaccesible")
    print("Mejor optar por otra configuracion...")

if("2" in predicted):
    print("Se predice que el auto es accesible")
    print("El auto esta pasable...")

if("3" in predicted):
    print("Se predice que el auto es bueno")
    print("El auto no te va a defraudar...")

if("4" in predicted):
    print("Se predice que el auto es muy bueno")
    print("Es tan bueno, que lo amarás...")

