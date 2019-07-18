# Proyecto Integrador Redes Neuronales Artificiales
Clasificador de autos

## Introducción
El objetivo de este proyecto es clasificar autos mediante algunos factores importantes como el mantenimiento y la seguridad, y ver si es una buena opción de compra.  
Se seleccionó un modelo clasificador.

Este modelo, evalúa los automóviles de acuerdo con la siguiente estructura conceptual: 
• COSTO: Precio de venta del automóvil. 
• MANTENIMIENTO: Precio del mantenimiento. 
• PUERTAS: Número de puertas 
• PERSONAS: Capacidad de las personas que se pueden llevar. 
• MALETERO: El tamaño del maletero. 
• SEGURIDAD: Estimada de seguridad del coche.

Y da como resultado 3 posibles respuestas:  
• INACCESIBLE 
• ACCESIBLE 
• BUENO 
• MUY BUENO 

## Características de los datos de entrenamiento
Se obtuvo un fichero .csv con 1728 ejemplos para el entrenamiento del clasificador. 

• Atributos: 6 [buying, maint, dors, persons, lug_boot, safety] 
• Clases: 4 [unacc, acc, good, vgood] 
• Tamaño muestral: 1728 
 
Originalmente, los datos no están como datos numéricos, si no, mas bien por etiquetas en inglés. 
Mas adelante se cambió en código los parámetros “vhigh”, “high”, “med” “low” por parámetros numéricos; 1, 2, 3, 4, respectivamente.  
Lo mismo ocurrió con las clases: 
• “unnac” (inaccesible) = 1 
• “acc” (accesible) = 2 
• “good” (bueno) = 3 
• “vgood” (muy bueno) = 4 
