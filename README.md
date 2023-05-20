# Face-heaan


## 1. Overview
- To implement a face ID using **homomorphic encryption**.

  <img width="500" alt="Screenshot 2023-05-20 at 2 20 35 PM" src="https://github.com/jeewonkimm2/Face-heaan/assets/108987773/553ff032-7d6f-4385-ac0f-022e70f9a9f2">
  
  #### Overall Structure
  
    - Face Identification
    - Face Registration
    - Similarity Measurement : Cosine Similarity, Manhattan Similarity, Euclidean Similarity
    - Webcam Inference



## 2. Environment
- Please notice that we assume that you are using 'PyTorch' and device type as 'cpu'. Python version is 3.9.
- Installing all the requirements may take some time. After installation, you can run the codes.
- Specific methods for setting preferences are described in #4.

## 3. Pre-Trained Model Download

- You need to download pretrained model for implementation.
- [Download Link][link2] / Password : b2ec
- The pretrained model use resnet-18 without se.
- Please modify the path for the pretrained model.
- A correct location for the pre-trained model is shown below.

  ```
  /face-heaan/checkpoints/resnet18_110.pth
  ```

## 4. Installation

- [```requirements.txt```][link1] file is required to set up the virtual environment for running the program. This file contains a list of all the libraries needed to run your program and their versions. 

  #### 1. In **venv** Environment,

  ```
  $ source venv/bin/activate

  $ pip install -r requirements.txt
  ```

  - Write the commands to activate the virtual environment and install the necessary libraries.
  - We've assumed that your environment is named 'venv'. If your virtual environment has a different name, you'll need to modify it to your name.
  - You have a 'requirements.txt' file that lists the required libraries, use the command **pip install -r requirements.txt** to install libraries.

  #### 2. In **Anaconda** Environment,

  ```
  $ conda activate [your virtual environment name]

  $ pip install -r requirements.txt
  ```

  - Activate your Anaconda virtual environment where you want to install the package. If your virtual environment is named 'piheaanenv', you can type **conda activate piheaanenv**.
  - Use the command **pip install -r requirements.txt** to install libraries.


-  If you encounter any conflicts, please check your dependencies carefully and reinstall according to your running environment.


## 5. Usage

#### You need to run [inference_heaan.py][link] file.

1. Face registration

    <img width="500" alt="Screenshot 2023-05-20 at 3 17 07 PM" src="https://github.com/jeewonkimm2/Face-heaan/assets/108987773/074278df-d377-4c17-ada4-47653d4fc559">
  
  - You can press space bar to proceed face registration.

2. Result

    <img width="500" alt="Screenshot 2023-05-20 at 3 14 36 PM" src="https://github.com/jeewonkimm2/Face-heaan/assets/108987773/79f63e39-a2f0-4473-8387-4cd9d3124e1e">

    - Unlock : When registered face is detected.
    
    
    <이미지 추가 필>

    - Lock : When registered face is not detected.


    <이미지 추가 필>
    
    - Too many faces : When many faces are detected.



## 6. Reference

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4690-4699).


[link]: https://github.com/jeewonkimm2/Face-heaan/blob/main/inference_heaan.py
[link1]: https://github.com/jeewonkimm2/Face-heaan/blob/main/requirements.txt
[link2]: https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA
