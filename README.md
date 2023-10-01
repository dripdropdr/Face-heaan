# Face-heaan


## 1. Overview
- To implement a face ID using **homomorphic encryption**.

  <img width="500" alt="Screenshot 2023-05-20 at 2 20 35 PM" src="https://github.com/jeewonkimm2/Face-heaan/assets/108987773/553ff032-7d6f-4385-ac0f-022e70f9a9f2">
  
  #### Overall Structure
  
    - Face Identification
    - Face Registration
    - Similarity Measurement : Cosine Similarity, Manhattan Similarity, Euclidean Similarity
    - Webcam Inference

- [Demo Video][link3]


## 2. Environment

- You must need a camera connected on your environment.
- Python version is 3.9.
- Please notice that we assume that you are using 'PyTorch' and device type as 'cpu'.
- Installing all the requirements may take some time. After installation, you can run the codes.
- Specific methods for setting preferences are described in #4.

## 3. Pre-Trained Model Download

- You need to download pretrained model for implementation.
- [Download Link][link2] / Password : b2ec
- The pretrained model use resnet-18 without se.
- Please modify the path for the pretrained model.
- A correct location for the pre-trained model is shown below.

  ```
  ./checkpoints/resnet18_110.pth
  ```

## 4. Installation

- Please download pytorch 1.12.0 depending on your OS. Instruction is in [here](https://pytorch.org/get-started/previous-versions/#v1120).
- [```requirements.txt```][link1] file is required to set up the virtual environment for running the program. This file contains a list of all the libraries needed to run your program and their versions. 

  #### 1. In **venv** Environment,

  ```
  $ python -m venv [your virtual environment name]

  $ source [your virtual environment name]/Scripts/activate
  
  $ pip install torch==1.12.0 #for Mac

  $ pip install -r requirements.txt
  ```
  - Create your own virtual environment.
  - Write the commands to activate the virtual environment and install the necessary libraries.
  - You have a 'requirements.txt' file that lists the required libraries, use the command **pip install -r requirements.txt** to install libraries.

  #### 2. In **Anaconda** Environment,

  ```
  $ conda create -n [your virtual environment name] python=3.9
  
  $ conda activate [your virtual environment name]
  
  $ conda install pytorch==1.12.0 -c pytorch #for Mac

  $ pip install -r requirements.txt
  ```

  - Create your own virtual environment.
  - Activate your Anaconda virtual environment where you want to install the package. If your virtual environment is named 'piheaanenv', you can type **conda activate piheaanenv**.
  - Use the command **pip install -r requirements.txt** to install libraries.


-  If you encounter any conflicts, please check your dependencies carefully and reinstall according to your running environment.


## 5. Usage

#### You need to run [inference_heaan.py][link] file.

1. Face registration

    <img width="500" alt="Screenshot 2023-05-20 at 3 17 07 PM" src="https://github.com/jeewonkimm2/Face-heaan/assets/108987773/074278df-d377-4c17-ada4-47653d4fc559">
  
  - You can press space bar to proceed face registration.

2. Result

    <img width="500" alt="Screenshot 2023-05-21 at 12 20 45 PM" src="https://github.com/jeewonkimm2/Face-heaan/assets/108987773/771b3c62-d1d4-46cc-bccd-572e69e50dbf">

    - Unlock : When registered face is detected.
    - Lock : When registered face is not detected.
    - Too many faces : When many faces are detected.


## 6. Change Measurement Method

```
# 1) cosine similarity measurement
res_ctxt = he.cosin_sim(ctxt1, ctxt2)
result = he.compare('cosine', cos_thres, res_ctxt)

# # 2) euclidean distance measurement
# res_ctxt = he.euclidean_distance(ctxt1, ctxt2)
# result = he.compare('euclidean', euc_thres, res_ctxt)

# # 3) manhattan distance measurement
# res_ctxt = he.manhattan_distance(ctxt1, ctxt2)
# result = he.compare('manhattan', man_thres, res_ctxt)
```

- In [inference_heaan.py][link] file, you'll find codes like the one above. When you download and run the code, it is currently set to the cosine similarity measurement technique.
- If you want to change to the euclidean similarity measure, you can comment out the rest of the block and uncomment the second and third lines of the second block.
- If you want to change to manhattan similarity measurement, you can comment out the rest of the blocks and uncomment the second and third lines of the third block.



## 7. Reference

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4690-4699).


[link]: https://github.com/jeewonkimm2/Face-heaan/blob/main/inference_heaan.py
[link1]: https://github.com/jeewonkimm2/Face-heaan/blob/main/requirements.txt
[link2]: https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA
[link3]: https://youtu.be/k0vf9HcV2Nw
