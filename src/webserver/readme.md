**Running Steps**

* Copy "Laptop_Code" directory onto local machine
* Open 2 sessions on Kahan, go to webserver directory on both. On one run the following srun command to request the titanrtx GPU: ```srun --nodes=1-1 --gres=gpu:titanrtx:1 --mem 16G --pty bash```
* On another session, just do a normal srun command: ```srun --nodes=1-1 --gres=gpu:1 --mem 16G --pty bash```
* On the session without the titanrtx GPU requested, run the docker command for openface in terminal mode, and leave it as is: ```docker run --name openface_docker -w /openface_dump/ -v ~/openface_dump/:/openface_dump -it --rm algebr/openface:latest```
* On the other session just run ```python3.8  app.py```
* Now, go to local machine and within the "Laptop_Code" directory, either run app.py with python3 or just do flask run. Follow the localhost link and record a video or enter a score manually. The video stops recording once the test on the screen says that it is processing. This should send that video over to Kahan for processing.
