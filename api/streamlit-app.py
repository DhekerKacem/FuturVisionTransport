import streamlit as st
import requests
from PIL import Image
import io

# URL de votre API Flask
API_URL = "http://127.0.0.1:5000/predict-mask"

st.title("Application de Segmentation Sémantique")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    
    # Convertir l'image en octets
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Envoyer l'image à l'API Flask
    st.write("Envoi de l'image à l'API Flask pour prédiction...")
    response = requests.post(API_URL, data=img_bytes, headers={"Content-Type": "application/octet-stream"})
    
    if response.status_code == 200:
        st.write("Prédiction reçue de l'API Flask")
        
        # Charger l'image prédite à partir de la réponse
        predicted_image = Image.open(io.BytesIO(response.content))
        
        # Afficher l'image prédite
        st.image(predicted_image, caption='Image avec Masque Prédit', use_column_width=True)
    else:
        st.write("Erreur dans la prédiction. Code de réponse:", response.status_code)
