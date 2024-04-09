import os

OPENFACE_PATH = "docker exec openface_docker /home/openface-build/build/bin/"
video_path = "uploaded_video.mp4"
output_dir = "openface_output"

command = f'{OPENFACE_PATH}FeatureExtraction -f "{video_path}" -out_dir {output_dir}'
os.system(command)
