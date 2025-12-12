import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Konfigurasi Model ---
CNN_MODEL_URL = 'https://drive.google.com/file/d/1RuGjNcW_Yi1MTjmgQymTKxP9594uzm4d/view?usp=drive_link' # Contoh: Link langsung dari GDrive
CNN_MODEL_PATH = 'model_cnn_85_15.h5' # Nama file lokal yang akan disimpan
SVM_MODEL_PATH = 'svm_ovo_model.pkl'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal'] # Sesuaikan dengan urutan indeks kelas saat training

# --- Fungsi Pemuatan Model (dengan cache) ---
@st.cache_resource
def load_cnn_model(path):
    """Memuat model CNN."""
    if not os.path.exists(path):
        st.info("Mengunduh model CNN. Ini mungkin memakan waktu...")
        
        # Logika unduhan (sangat disederhanakan)
        try:
            # Contoh unduhan sederhana (ganti dengan logika unduhan GDrive/S3 yang sesuai jika diperlukan)
            response = requests.get(CNN_MODEL_URL, stream=True)
            response.raise_for_status() # Cek jika ada error HTTP
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model CNN berhasil diunduh!")
        except Exception as e:
            st.error(f"Gagal mengunduh model dari URL: {e}")
            return None
# Lanjutkan dengan pemuatan model seperti biasa
    full_cnn_model = load_model(path)
    feature_extractor = Model(
        inputs=full_cnn_model.input,
        outputs=full_cnn_model.get_layer('feature_layer').output 
    )
    return feature_extractor

@st.cache_resource
def load_svm_model(path):
    """Memuat model SVM."""
    try:
        svm_model = joblib.load(path)
        return svm_model
    except Exception as e:
        st.error(f"Gagal memuat model SVM: {e}")
        return None

# Muat Model
feature_extractor = load_cnn_model(CNN_MODEL_PATH)
svm_model = load_svm_model(SVM_MODEL_PATH)

# --- Fungsi Prediksi ---
def predict_image(img, feature_extractor, svm_model, class_names):
    """Memproses gambar dan melakukan prediksi hybrid CNN-SVM."""
    
    # 1. Prarproses Gambar
    # Resize ke 224x224 (ukuran input model)
    img = img.resize(IMAGE_SIZE)
    # Konversi ke array numpy
    img_array = image.img_to_array(img)
    # Tambahkan dimensi batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalisasi (sesuai dengan 1./255 yang digunakan saat training)
    img_array = img_array / 255.0

    # 2. Ekstraksi Fitur dengan CNN
    # Ekstrak fitur dari feature_layer
    features = feature_extractor.predict(img_array)
    # Flatten fitur (jika belum) untuk input SVM
    features = features.reshape(features.shape[0], -1)

    # 3. Prediksi dengan SVM
    # Lakukan prediksi kelas
    prediction = svm_model.predict(features)
    
    # Dapatkan nama kelas
    predicted_class_index = prediction[0]
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

# --- Streamlit UI ---
st.title("Aplikasi Klasifikasi Citra Hybrid CNN-SVM")
st.markdown("Unggah gambar untuk memprediksi jenis penyakit kulit (Chickenpox, Measles, Monkeypox, Normal).")

if feature_extractor is not None and svm_model is not None:
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        img = Image.open(uploaded_file)
        st.image(img, caption='Gambar yang Diunggah.', use_column_width=True)
        st.write("")
        st.write("Menganalisis gambar...")

        # Lakukan prediksi
        with st.spinner('Model sedang memproses...'):
            try:
                predicted_label = predict_image(img, feature_extractor, svm_model, CLASS_NAMES)
                
                # Tampilkan hasil
                st.success(f"Prediksi Model: **{predicted_label}**")
                
                # Opsional: Tampilkan indeks kelas untuk debugging
                # st.write(f"Indeks Kelas: {CLASS_NAMES.index(predicted_label)}")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.info("Pastikan file model sudah benar dan terletak di direktori yang sama.")
else:
    st.error("Model tidak berhasil dimuat. Silakan periksa path file model.")

# Footer
st.caption("Aplikasi ini dibuat untuk tujuan demonstrasi model. Selalu konsultasikan dengan ahli kesehatan.")
