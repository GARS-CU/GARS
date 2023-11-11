# To Do List

## A place to keep track of what we're currently working on and what still needs to be done

### Build classifier according to https://www.sciencedirect.com/science/article/pii/S0045790621002597
- Datasets Used
  - FER-2013: https://www.kaggle.com/datasets/msambare/fer2013
  - MES dataset: https://github.com/Harsh9524/MES-Dataset/tree/main
- Find alternative image based engagement dataset to train focus classifier (it looks like the labels they made for the FER dataset aren't public)
- Train emotion detection classifier on FER dataset https://www.kaggle.com/datasets/msambare/fer2013
- Evaluate on datasets such as CK+ and DaISEE

### Build classifier according to https://thesai.org/Downloads/Volume14No3/Paper_71-Convolutional_Neural_Network_Model.pdf
- Datasets Used
  - DAiSEE dataset: https://drive.google.com/file/d/1yrk_wyhZ-c7q0Mcyyi888ylFkl_JDELi/view?usp=share_link
- Figure out how to extract features using OpenFace library and perform feature selection using either PCA or SVD
- Try to get performance comparable to what's in paper and evaluate performance on other video based engagement datasets like CK+

