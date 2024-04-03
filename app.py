import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Carga el modelo
modelo = load_model('model_final.h5')

st.title('Detector de Neumonía en Imágenes de Tórax')

# Subida de la imagen del usuario
uploaded_file = st.file_uploader("Carga una imagen de rayos X de tórax", type=["jpg", "jpeg"])

def cargar_y_preparar_imagen(uploaded_file):
    # Convierte la imagen cargada en una imagen de PIL
    img_pil = image.load_img(uploaded_file, target_size=(224, 224))
    # Convierte la imagen de PIL a un numpy array
    img = image.img_to_array(img_pil)
    # Preprocesamiento de la imagen para el modelo
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    return img

if uploaded_file is not None:
    # Muestra la imagen
    st.image(uploaded_file, caption='Imagen cargada', use_column_width=True)
    # Prepara la imagen
    img = cargar_y_preparar_imagen(uploaded_file)
    # Realiza la predicción
    predicción = modelo.predict(img)
    # Agrega impresiones de depuración para ver la salida de la predicción
    st.write('Debug - Predicción:', predicción)
    
    # Manejo de la salida de la predicción basado en la forma esperada del resultado
    if predicción.ndim == 1:
        # Si la predicción es un array 1D, usa el primer valor para la condición
        resultado = predicción[0]
    elif predicción.ndim == 2:
        # Si la predicción es un array 2D, como (1, 1), usa el primer valor del primer subarray
        resultado = predicción[0, 0]
    else:
        # Si la predicción tiene una forma inesperada, muestra un mensaje de error
        st.error("Error: la forma del resultado de la predicción no es la esperada.")
        resultado = None
    
    # Interpretar la predicción basado en el valor de 'resultado'
    if resultado is not None:
        if resultado < 0.5:
            st.write('Resultado: No se detecta neumonía')
        else:
            st.write('Resultado: Posible neumonía detectada')