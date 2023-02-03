import json
import os
import streamlit as st

from utils.utils import display_spectrogram_from_path, display_chart

st.title("Statistic and Vision")
st.subheader("Details of the last training")

if st.button('Load last run informations'):
    f = open('last_run.json')
    data = json.load(f)
    fig1 = display_chart()
    st.pyplot(fig1)


st.subheader("Spectrogram")
if st.button('Load spectrograms'):
    f = open('last_run.json')
    data = json.load(f)
    fig2 = display_spectrogram_from_path(os.path.join(data['dataset_dir'], 'spectrogram'))
    st.pyplot(fig2)