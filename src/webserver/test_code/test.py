#export GARS_PROJ = "/zooper2/gars/GARS/src"
#run this line or put at the end of ~/.bashrc
import os
import sys
sys.path.append(os.environ['GARS_PROJ'])  #append path for util 
sys.path.append(os.path.join(os.environ['GARS_PROJ'], 'art_generate')) #append path for rec system 
from util import *
from art_rec_bog import ArtRecSystem

#generate true loads in stable diffusion 
rec = ArtRecSystem(metric='cosine', art_generate=True)

image, prompt, words = rec(rating=0)

image_path = "generated_art.png"
image.save(image_path)
print(prompt)
print(words)

#python3.8 test.py (make sure 3.8 is used for other stuff to fix library issue)
