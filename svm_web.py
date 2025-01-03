import streamlit as st
import pickle
import numpy as np

# Memuat file model SVM
model_path = './svm_model.pkl' # Perbaiki jalur file
try:
    with open(model_path, 'rb') as f:
        model_SVM = pickle.load(f)
except FileNotFoundError:
    st.error(f"File model tidak ditemukan di: {model_path}")
    st.stop()

# Judul Aplikasi
st.title('Prediksi Spesies Ikan')

# Dropdown untuk memilih model
model_choice = st.selectbox(
    'Pilih Model untuk Prediksi:',
    ('SVM')  # Tambahkan model lain jika ada
)

# Input untuk setiap fitur ikan
length = st.number_input('Panjang Ikan (length):', min_value=0.0)
weight = st.number_input('Berat Ikan (weight):', min_value=0.0)
w_l_ratio = st.number_input('Rasio Berat ke Panjang (w_l_ratio):', min_value=0.0)

# Tombol untuk memprediksi spesies ikan
if st.button('Prediksi Spesies'):
    features = np.array([[length, weight, w_l_ratio]])
    
    if model_choice == 'SVM':
        model = model_SVM
    
    prediction = model.predict(features)[0]
    st.success(f'Spesies yang Diprediksi: {prediction}')
