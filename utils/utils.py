from PIL import Image

import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_spectrogram(audio : np.array, sr, signal_length : int, args : dict) -> list[np.array]:
    """get spectrogram for each audio chunk

    Args:
        audio (np.array): numpy array of audio signal
        sr (int): sample rate
        signal_length (int): length of the signal in seconds
        args (dict): args for mel spectrogram extraction

    Returns:
        list[np.array]: spectrograms
    """
    # Split signal into five second chunks
    sig_splits = []
    for i in range(0, len(audio), int(signal_length * sr)):
        split = audio[i:i + int(signal_length * sr)]

        # End of signal?
        if len(split) < int(signal_length * sr):
            break
        
        sig_splits.append(split)
        
    # Extract mel spectrograms for each audio chunk
    specs = []
    for chunk in sig_splits:
        
        mel_spec = librosa.feature.melspectrogram(y=chunk, 
                                                  sr=sr, 
                                                  n_fft=1024, 
                                                  **args)
    
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max) 
        
        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        
        specs.append(mel_spec)
        
    return specs

def save_spectrogram(filename : str, label : str, dataset : str, signal_length : int ,args : dict) -> None:
    """save spectrogram for each audio chunk

    Args:
        filename (str): name of the file
        label (str): label of the file
        dataset (str): path to the dataset
        signal_length (int): length of the signal in seconds
        kwargs (_type_): mel spectrogram extraction args
    """
    # Load audio
    audio, sr = librosa.load(os.path.join(dataset, label, filename), sr=None)
    args["hop_length"] = int(sr * signal_length/ (args["n_mels"] -1))
    
    # Get spectrograms
    specs = get_spectrogram(audio, sr, signal_length, args)
    
    # Save spectrograms as PIL image files
    for i,spec in enumerate(specs):
        if not os.path.exists(os.path.join(dataset, "spectrogram",label)):
            os.makedirs(os.path.join(dataset, "spectrogram", label))
        img = Image.fromarray(spec * 255).convert("L")
        img.save(os.path.join(dataset, "spectrogram",label, filename[:-4] + f"_{i}.png"))

def display_chart():
    f = open('last_run.json')
    data = json.load(f)
    logs = os.listdir(os.path.join("logs", data["name_run"]))
    logs = sorted(logs ,key = lambda x: int(x.split("_")[-1]))
    print(logs)
    df = pd.read_csv(os.path.join("logs", data["name_run"], logs[-1], "metrics.csv"))
    df_train = df.iloc[:,0:4].dropna(how = "any").groupby("epoch").mean()
    df_valid = df.iloc[:,3:7].dropna(how = "any").groupby("epoch").mean()
    fig, ax = plt.subplots(2, 2, figsize = (20, 20))
    ax[0,0].plot(df_train.index, df_train["train_loss_step"], label = "train loss")
    ax[0,0].set_title("train loss")
    ax[0,1].plot(df_train.index, df_train["train_acc_step"], label = "train acc")
    ax[0,1].set_title("train acc")
    ax[1,0].plot(df_valid.index, df_valid["valid_loss"], label = "valid loss")
    ax[1,0].set_title("valid loss")
    ax[1,1].plot(df_valid.index, df_valid["valid_acc"], label = "valid acc")
    ax[1,1].set_title("valid acc")
    return fig

def display_spectrogram(specs : list[np.array], labels : list[str]):
    row = len(specs) // 3 if len(specs) % 3 == 0 else len(specs) // 3 + 1    
    fig, axs = plt.subplots(row, 3, figsize = (30, 30))
    print(axs.shape)
    axs = axs.flatten()
    print(axs.shape)
    for i in range(len(specs)):
        axs[i].imshow(specs[i][0].numpy().squeeze())
        #axs[i].set_title(labels[i])
        axs[i].xaxis.set_visible(False)
        axs[i].yaxis.set_visible(False)
    return fig
        
def display_spectrogram_from_path(dataset_dir : str):
    labels = os.listdir(dataset_dir)
    files = [[file for file in os.listdir(os.path.join(dataset_dir, label))] for label in labels]
    fig, axs = plt.subplots((len(labels)), 3, figsize = (len(labels), 30))
    for i in range(len(labels)):
        ax = axs[i]
        label = labels[i]
        file = files[i]
        for j in range(3):
            ax[j].imshow(plt.imread(os.path.join(dataset_dir, label, file[j])))
            ax[j].set_title(label)
            ax[j].xaxis.set_visible(False)
            ax[j].yaxis.set_visible(False)
    return fig