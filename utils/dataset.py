from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os
import torch
import torchvision.transforms as Tv

class BirdDataset(Dataset):
  def __init__(self, files, class_mapping : dict) -> None:
    self.files = files
    self.class_mapping = class_mapping
    self.crop = Tv.CenterCrop(224)
    self.transform = Tv.Compose([Tv.ToTensor(),
                                 #lambda x : x/255.0,
                                 lambda x : torch.vstack((x,x,x))])

  def __len__(self) -> int:
    return len(self.files)  
  
  def normalizefunc(self, spec):
    spec = np.array(spec, dtype = "float32")
    spec = spec / 255.0
    spec = spec - np.mean(spec)
    #spec = spec / (np.std(spec))
    return spec

  def __getitem__(self, idx: int):
    spec = Image.open(self.files[idx])
    spec = self.crop(spec)
    label = self.class_mapping[self.files[idx].split("/")[-2]]
    spec = self.normalizefunc(spec)
    return self.transform(spec), label