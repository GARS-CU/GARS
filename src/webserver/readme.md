**Running Steps**

* Copy "Laptop_Code" directory onto local machine
* Open 2 sessions on Kahan, go to webserver directory, and do the following on both: ```srun --nodes=1-1 --gres=gpu:1 --mem 15G --pty bash```
* On one session, run the docker command for openface in terminal mode, and leave it as is: ```docker run --name openface_docker -w /openface_dump/ -v ~/openface_dump/:/openface_dump -it --rm algebr/openface:latest```
* On the other session, the docker container for the app might be built. In that case just run: ```docker run -p 8000:8000 -v ~/openface_dump/:/app/openface_dump engagement_app
```. Otherwise, you'll need to build the container, so run: ```docker build -t engagement_app .``` 
* Now, go to local machine and within the "Laptop_Code" directory, either run app.py with python3 or just do flask run. Follow the localhost link and record a video using the button. The video stops recording once the test on the screen says that it is processing. This should send that video over to Kahan for processing.
