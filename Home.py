from utils.utils import save_spectrogram
from utils.models import run

import json
import os
import pandas as pd
import streamlit as st

st.title('Acoustic Classifier')

dataset = st.text_input('relative path to the dataset', "dataset/")
st.text('The relative path is relative to path from where you run the script. \nThe dataset should be in the following format\ndataset/label1/file1.wav')

st.subheader("Preprocessing informations")
st.text("The preprocessing is done using librosa library,"
        "\nThe signal is split into 5 second chunks by default"
        "\nThe mel spectrogram is extracted for each chunk"
        "\nHere you will enter all the informations to precprocess all the audio files for the training."
        "\nIf you don't know what to put, just leave the default values"
        "\nIf you made this part before or you have all the spectrogram pass this part and go to the training part")
file_type = st.text_input("File type", "wav")
signal_length = st.number_input("Signal length in seconds", min_value=2, max_value=15, value=5)
spec_size = st.number_input("spectrogram size", min_value=1, max_value=2048, value=256)
fmin = st.number_input("Minimum frequency", min_value=1, value=500)
fmax = st.number_input("Maximum frequency", min_value=2, value=14000)
args = {"hop_length": 0, "n_mels": spec_size, "fmin": fmin, "fmax": fmax}

if st.button('Preprocess'):
        st.write('Preprocessing...')
        label = [label for label in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, label))]
        files = [[file, label] for label in label for file in os.listdir(os.path.join(dataset, label)) if file.endswith(file_type)]
        number_of_files = len(files)
        st.write('Number of labels: ', len(label))
        st.write('Number of files to preprocess: ', number_of_files)
        my_bar = st.progress(0)
        for i, file in enumerate(files):
                save_spectrogram(file[0], file[1], dataset, signal_length, args)
                my_bar.progress(i / number_of_files)
        st.write('Preprocessing done')
        json.dump(args, open("preprocinfo.json", "w"))

st.subheader("Training informations")
st.text("Here you will enter all the informations to train the model."
        "\nIf you don't know what to put, just leave the default values"
        "\nThe training is done using pytorch library and pytorch lightning framework")

max_epochs = st.number_input("Maximum number of epochs", min_value=1, max_value=1000, value=100)
save_dir = st.text_input("Relative path to the directory where to save the model", "model/")
name_run = st.text_input("Name of the run", "first_run")
batch_size = st.number_input("Batch size", min_value=1, max_value=1000, value=32)
mixup = st.checkbox("Mixup", value=True)
last_run = {"name_run" : name_run, "dataset_dir" : dataset}
with open("last_run.json", "w") as outfile:
        json.dump(last_run, outfile)

if st.button('Train'):
        st.write('Training...')
        run(os.path.join(dataset,"spectrogram"), batch_size, name_run, max_epochs, save_dir, mixup)
        st.write('Training done')
        st.write('test accuracy : ', pd.read_csv(os.path.join("logs", name_run, "version_0","metrics.csv"))["test_acc"].iloc[-1])

