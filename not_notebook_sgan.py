import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy import ndimage

from src.generator import Generator
from src.discriminator import Discriminator
from src.sgan import SGAN

BATCH_SIZE = 128

gan = SGAN(verbosity=False)

from src.utils import image_dataset_from_directory
dir = "images/"

#divide by two because discriminator is taking other half from generator
train_ds = image_dataset_from_directory(dir,
  validation_split=0.2, subset="training",
  seed=123,
  labels="inferred",label_mode="int"
  ,image_size=(128, 128), color_mode= "rgb",
    batch_size=BATCH_SIZE//2)

val_ds = image_dataset_from_directory(dir,
  validation_split=0.2, subset="validation",
  seed=123,
  labels="inferred",label_mode="int"
  ,image_size=(128, 128), color_mode= "rgb",
    batch_size=BATCH_SIZE//2)

gan.train(X=train_ds, y=val_ds, epochs=1000, batch_size=BATCH_SIZE)
