from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models import vit_b_32, ViT_B_32_Weights

from utils.dataset import BirdDataset

import json
import os
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
import torch

class Lumiere(pl.LightningModule):
    def __init__(self, num_classes : int, mixup : bool) -> None:
        super().__init__()
        self.model = vit_b_32()
        self.model.heads.head = nn.Linear(768, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.mixup = mixup
        self.num_classes = num_classes
        
        # logger
        self.save_hyperparameters(logger=False)
        
        # metrics
        self.accuracy = tm.Accuracy(task = 'multiclass', num_classes = num_classes)
        self.f1 = tm.F1Score(task = 'multiclass', num_classes = num_classes)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        
        if self.mixup :
            y = torch.nn.functional.one_hot(y, self.num_classes)
        
        logits = self.model(x)
        loss = self.criterion(logits, y)
        
        # metrics
        self.accuracy(logits, y)
        self.f1(logits, y)
        
        # log metrics
        self.log('train_loss', loss, prog_bar=False, logger = True, on_epoch=True)
        self.log('train_acc', self.accuracy, prog_bar=False, logger = True, on_epoch=True)
        self.log('train_f1', self.f1, prog_bar=False, logger = True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
            
        
        logits = self.model(x)
        loss = self.criterion(logits, y)
        
        # metrics
        self.accuracy(logits, y)
        self.f1(logits, y)
        
        # log metrics
        self.log('valid_loss', loss, prog_bar=False, logger = True, on_epoch=True)
        self.log('valid_acc', self.accuracy, prog_bar=False, logger = True, on_epoch=True)
        self.log('valid_f1', self.f1, prog_bar=False, logger = True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        
        # metrics
        self.accuracy(logits, y)
        self.f1(logits, y)
        
        # log metrics
        self.log('test_loss', loss, prog_bar=False, logger = True, on_epoch=True)
        self.log('test_acc', self.accuracy, prog_bar=False, logger = True, on_epoch=True)
        self.log('test_f1', self.f1, prog_bar=False, logger = True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-6)
        return [optimizer], [lr_scheduler]
    

def run(dataset_dir : str, batch_size : int, name_run : str, max_epochs : int, save_dir : str, mixup : bool) -> None:
    # dataset
    labels = [i for i in os.listdir(dataset_dir)]
    class_map = {i : labels.index(i) for i in labels}
    print(class_map)
    #export the classmap in JSON
    with open("class_map.json", "w") as outfile:
        json.dump(class_map, outfile)
    files = [os.path.join(dataset_dir, label, file) for label in labels for file in os.listdir(os.path.join(dataset_dir, label)) if file.endswith("png")]
    
    print(files[0][0])
    
    dataset_train, dataset_test = train_test_split(files, test_size = 0.2, random_state = 42)
    dataset_train, dataset_valid = train_test_split(dataset_train, test_size = 0.2, random_state = 42)
    
    dataset_train = BirdDataset(dataset_train, class_map)
    dataset_valid = BirdDataset(dataset_valid, class_map)
    dataset_test = BirdDataset(dataset_test, class_map)
    
    #DataLoaders
    
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
    dataloader_valid = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False, num_workers = 4)
    dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 4)
    
    #model
    model_pl = Lumiere(len(class_map), False)
    callbacks = [ModelCheckpoint(dirpath = os.path.join(save_dir, name_run), monitor = 'valid_loss', save_top_k = 1, mode = 'min'), 
                EarlyStopping(monitor = 'valid_loss', patience = 4, mode = 'min')]
    
    #logger
    logger = CSVLogger("logs", name = name_run, version=0)
    print(logger.log_dir)
    
    #trainer
    trainer = Trainer(accelerator = 'auto',
                  max_epochs = max_epochs,
                  logger = logger,
                  callbacks= callbacks,)
    
    #training
    trainer.fit(model = model_pl, 
             train_dataloaders = dataloader_train, 
             val_dataloaders = dataloader_valid)
    
    #testing
    trainer.test(model = model_pl, 
                dataloaders = dataloader_test)