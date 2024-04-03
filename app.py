import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Carga el modelo
modelo = load_model('model_final.h5')

st.title('Detector de Neumonía en Imágenes de Tórax')

# Subida de la imagen del usuario
uploaded_file = st.file_uploader("Carga una imagen de rayos X de tórax", type="jpg")

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
    # Interpretar la predicción
    if predicción[0] < 0.5:
        st.write('Resultado: No se detecta neumonía')
    else:
        st.write('Resultado: Posible neumonía detectada')