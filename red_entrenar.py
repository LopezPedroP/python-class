import os
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizer_v2.adam import Adam

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

#este  es el directorio donde tenemos las imagenes para entrenar
data_entrenamiento = './Data/Entrenamiento'

#este  es el directorio donde tenemos las imagenes para evaluar
data_evaluacion = './Data/Validacion'

#parametros
epocas = 5
altura, longitud = 100,100
batch_size = 32
pasos = 20
pasos_validacion = 200
filtosConv1 = 32
filtosConv2 = 64
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool = (2,2)
clases = 3
lr = 0.005

#pre procesamiento de las imagenes
entrenamiento_dataGen = ImageDataGenerator(
    rescale = 1/255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
    )

validacion_dataGen = ImageDataGenerator(
    rescale = 1/255)

#procesamiento de imagenes entrenar
imagen_entrenamiento = entrenamiento_dataGen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
    )

imagen_validacion = validacion_dataGen.flow_from_directory(
    data_evaluacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
    )

#creando la red comvolucional
cnn = Sequential()

cnn.add(Convolution2D(filtosConv1, tamano_filtro1, padding = 'same', input_shape = (altura, longitud,clases), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tamano_pool))

cnn.add(Convolution2D(filtosConv2, tamano_filtro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tamano_pool))

#iniciamos la clasificacion
cnn.add(Flatten())
cnn.add(Dense(256,'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation = 'softmax' ))

cnn.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = lr), metrics=['accuracy'] )

 
cnn.fit(imagen_entrenamiento,steps_per_epoch=pasos, epochs = epocas, validation_steps = pasos_validacion, validation_data = imagen_validacion )

dir = './Modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
    
cnn.save('./Modelo/modelo.h5')
cnn.save_weights('./Modelo/pesos.h5')









