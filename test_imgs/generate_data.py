import os
from PIL import Image
import numpy as np

def load_image( infilename ) :
  img = Image.open( infilename )
  img.load()
  data = np.asarray( img, dtype="int32" )
  return data

def save_image( npdata, outfilename ) :
  img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
  img.save( outfilename )



def main():
  filename = 'test_output.png'
  image_data = load_image(filename)
  image_data = image_data[:,:,0:1]
  print(image_data.shape)
  image_data = np.reshape(image_data, (image_data.shape[0], image_data.shape[1]))
  outname_pattern = list("image0000")
  for i in range(0, 101):
    num = str(i)
    for j in range(len(num)):
      outname_pattern.pop()

    for j in range(len(num)):
      outname_pattern.append(num[j])
    save_image(image_data, "./generated/"+''.join(outname_pattern)+".png")

main()