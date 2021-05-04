import os
from PIL import Image

with os.scandir('images/raw') as it:
  for dir in it:
    with os.scandir('images/raw/' + dir.name) as it:
      if not os.path.exists('images/' + dir.name):
        os.mkdir('images/' + dir.name.lower())
      for file in it:
        image = Image.open('images/raw/' + dir.name + '/' + file.name)
        image = image.resize((128, 128))
        image.save('images/'+ dir.name.lower() + '/' + file.name.lower())