# -*- coding: utf-8 -*-
"""

Cursos Libres CUNIZAB -USAC-
Curso de Introduccion a la IA con Python y Tensorflow
Titulo: Programa de prediccion de IA
Autor: Juan Pablo Salazar Barrios
Fecha de creacion: 26/08/2021

"""

#Importamos librerias
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

#Declaracion de funciones
def prediccion(archivo, red):
    dato = load_img(archivo, target_size = (100, 100))
    dato = img_to_array(dato)
    dato = np.expand_dims(dato, axis = 0)
    arreglo = red.predict(dato) ##[[1,0,0]]
    resultado = arreglo[0] ##[1,0,0]
    respuesta = np.argmax(resultado) #Indice del numero mayor
    if respuesta == 0:
        print("Hay un perro en la imagen!")
    elif respuesta == 1:
        print("Hay un gato en la imagen!")
    elif respuesta == 2:
        print("Hay un gorila en la imagen!")
        
#Declaracion de variables
encabezado = """
.##.....##.####.##....##.########....####....###...
.###...###..##..##...##..##...........##....##.##..
.####.####..##..##..##...##...........##...##...##.
.##.###.##..##..#####....######.......##..##.....##
.##.....##..##..##..##...##...........##..#########
.##.....##..##..##...##..##...........##..##.....##
.##.....##.####.##....##.########....####.##.....##
Developed by: HercolobusGT
"""
menu = """
Menu principal:
    1) Predecir
    2) Creditos
    3) Salir
"""
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
#Cargamos los resultados del entrenamiento de la CNN
cnn = load_model(modelo)
cnn.load_weights(pesos)

seleccion = True

while seleccion:
    print("\n" , encabezado)
    print("\n" , menu)
    llave = input("\nSelecciona un numero del menu: ")
    if llave == "1":
        #Codigo de prediccion
        a = input("Ingresa el nombre del archivo a predecir: ")
        prediccion(a, cnn)
    if llave == "2":
        #Creditos al autor
        print("\n2021 Aplicacion desarrollada en el curso libre de introduccion a la IA en la USAC (CUNIZAB)")
    if llave == "3":
        print("\nGracias por usar nuestra app! Vuelve pronto")
        break
        seleccion = False
    
    
    
    
    
    
    
    
    
    
    
    
    






