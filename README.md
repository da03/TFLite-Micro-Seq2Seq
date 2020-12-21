# TFLite-Micro-Seq2Seq
attentional sequence-to-sequence model using TFLite Micro, tested on Arduino Nano 33 BLE


# Training

The code for training a model can be found at this Google Colab: https://colab.research.google.com/drive/1d8Kz6P-7O0OAyFNSAoR0C2V48vldcjfA?usp=sharing

# Deployment to Arduino

The training code will generate two files: `c_src/model.h` and `c_src/model.cpp`. Download these two files and put that in the directory `Arduino_Project/src/`. Then compile the Arduino project in `Arduino_Project`, upload it to MCUs, and run. Note that there's already a pretrained model in `Arduino_Project/src/`. 
