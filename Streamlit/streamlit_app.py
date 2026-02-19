import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')


def is_music(uploaded_file):
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    data,sample_rate = sf.read(uploaded_file)
    if len(data.shape) > 1:
        data = np.mean(data,axis=1)
    if sample_rate != 16000:
        data = librosa.resample(data,orig_sr=sample_rate,target_sr=16000)
    
    scores, embeddings, spectrogram = model(data)
    mean_scores = np.mean(scores.numpy(), axis=0)

    class_map_path = model.class_map_path().numpy().decode()
    class_names = [line.strip() for line in open(class_map_path).readlines()[1:]]

    top_class = class_names[np.argmax(mean_scores)]

    return top_class
def app() :
    st.title("Genre Classifier")

    st.text("This Project to learn basic CNN ")
    st.text("Description : User will upload an audio clip and the model will tell which type Genre music it is ")
    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"])

    if uploaded_file is not None:
        result = is_music(uploaded_file)

        # st.write("Detected class:", result)

        if "Music" in result:
            st.success("🎵 Music detected")
        else:
            st.error("❌ Not music")
if __name__ == '__main__':
    app()