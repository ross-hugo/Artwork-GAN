import numpy as np
import os

# The next line is only needed for AMD GPUs
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras.preprocessing.image as img

image = img.load_img('images/claude_monet/claude_monet_6.jpg')
np_image = img.img_to_array(image)

print(np_image.shape)
image.show()