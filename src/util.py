#class made to distinguish varibles 
from enum import StrEnum
import os
class var(StrEnum):
     OPENFACE_PATH = "sudo docker exec openface_docker /home/openface-build/build/bin/"
     GARS_PROJ = os.environ['GARS_PROJ']

        