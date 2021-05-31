# Assignment 4  - LATE :(



**ASSIGNMENT  - Create an architecture using below conditions**

- 99.4% validation accuracy
- **Between 12000 to 18000** -  13,640 Parameter
- **19 Epochs**
- **Add GAP** - Added
- **Add Rotation -5 to +5 degrees** -  3



----

#### Concepts Used

- Convolution - (3x3 and Pointwise Convolution)
- Batch Normalization 
- Activation - Relu
- DropOuts (0.069)
- GAP
- Rotation - 3

#### Architecture

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 3,)  #input = 28 Output = 26 RF = 3
        self.bn1 = nn.BatchNorm2d(12)
        self.do1 = nn.Dropout(0.069)    # Drop out of 0.069 :P
                              

        self.conv2 = nn.Conv2d(12, 20, 3) #input = 26 Output = 24 RF = 5
        self.bn2 = nn.BatchNorm2d(20)
        self.do2 = nn.Dropout(0.069)

        self.one1 = nn.Conv2d(20, 10, 1)  #input = 24 Output = 24 RF = 5
        self.pool1 = nn.MaxPool2d(2, 2)   #input = 24 Output = 12 RF = 10

        self.conv3 = nn.Conv2d(10, 16, 3) #input = 12 Output = 10 RF = 12
        self.bn3 = nn.BatchNorm2d(16)
        self.do3 = nn.Dropout(0.069)
        
        self.conv4 = nn.Conv2d(16, 16, 3)  #input = 10 Output = 8 RF = 14
        self.bn4 = nn.BatchNorm2d(16)
        self.do4 = nn.Dropout(0.069)

        # self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(16, 16, 3) #input = 8 Output = 6 RF = 16
        self.bn5 = nn.BatchNorm2d(16)
        self.do5 = nn.Dropout(0.069)
                              
        self.conv6 = nn.Conv2d(16, 16, 3) #input = 6 Output = 4 RF = 18
        self.bn6 = nn.BatchNorm2d(16)
        self.do6 = nn.Dropout(0.069)
        
        self.conv7 = nn.Conv2d(16, 16, 3) #input = 4 Output = 2 RF = 20
        self.bn7 = nn.BatchNorm2d(16)
        self.do7 = nn.Dropout(0.069)

        self.GAP = nn.AvgPool2d(kernel_size=(2,2)) #ADDED GAP 

        self.conv8 = nn.Conv2d(16, 10, 1) #input = 2 Output = 1 RF = 22
```

# Model Summary

```python
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 12, 26, 26]             120
       BatchNorm2d-2           [-1, 12, 26, 26]              24
           Dropout-3           [-1, 12, 26, 26]               0
            Conv2d-4           [-1, 20, 24, 24]           2,180
       BatchNorm2d-5           [-1, 20, 24, 24]              40
           Dropout-6           [-1, 20, 24, 24]               0
            Conv2d-7           [-1, 10, 24, 24]             210
         MaxPool2d-8           [-1, 10, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           1,456
      BatchNorm2d-10           [-1, 16, 10, 10]              32
          Dropout-11           [-1, 16, 10, 10]               0
           Conv2d-12             [-1, 16, 8, 8]           2,320
      BatchNorm2d-13             [-1, 16, 8, 8]              32
          Dropout-14             [-1, 16, 8, 8]               0
           Conv2d-15             [-1, 16, 6, 6]           2,320
      BatchNorm2d-16             [-1, 16, 6, 6]              32
          Dropout-17             [-1, 16, 6, 6]               0
           Conv2d-18             [-1, 16, 4, 4]           2,320
      BatchNorm2d-19             [-1, 16, 4, 4]              32
          Dropout-20             [-1, 16, 4, 4]               0
           Conv2d-21             [-1, 16, 2, 2]           2,320
      BatchNorm2d-22             [-1, 16, 2, 2]              32
          Dropout-23             [-1, 16, 2, 2]               0
        AvgPool2d-24             [-1, 16, 1, 1]               0
           Conv2d-25             [-1, 10, 1, 1]             170
================================================================
Total params: 13,640
Trainable params: 13,640
Non-trainable params: 0
```

### 

#### Model Run

```
0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:49: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.04637498781085014 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.18it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0554, Accuracy: 9820/10000 (98.20%)

loss=0.07610229402780533 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.53it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0327, Accuracy: 9889/10000 (98.89%)

loss=0.06193776801228523 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.18it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0347, Accuracy: 9895/10000 (98.95%)

loss=0.09145304560661316 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.32it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0318, Accuracy: 9909/10000 (99.09%)

loss=0.017810875549912453 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.20it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0277, Accuracy: 9925/10000 (99.25%)

loss=0.022182628512382507 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.22it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0241, Accuracy: 9934/10000 (99.34%)

loss=0.053338613361120224 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.20it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0248, Accuracy: 9923/10000 (99.23%)

loss=0.00915596354752779 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.28it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0248, Accuracy: 9921/10000 (99.21%)

loss=0.012974896468222141 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.28it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0276, Accuracy: 9918/10000 (99.18%)

loss=0.0036802401300519705 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.85it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0218, Accuracy: 9937/10000 (99.37%)

loss=0.06906979531049728 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.80it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0206, Accuracy: 9939/10000 (99.39%)

loss=0.008358734659850597 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.85it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0224, Accuracy: 9936/10000 (99.36%)

loss=0.028453081846237183 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.84it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0198, Accuracy: 9943/10000 (99.43%)

loss=0.0015235616592690349 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.74it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0218, Accuracy: 9938/10000 (99.38%)

loss=0.060001879930496216 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 15.02it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0262, Accuracy: 9915/10000 (99.15%)

loss=0.029139166697859764 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.79it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0208, Accuracy: 9939/10000 (99.39%)

loss=0.0010347756324335933 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.81it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0207, Accuracy: 9932/10000 (99.32%)

loss=0.08070048689842224 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.80it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0220, Accuracy: 9932/10000 (99.32%)

loss=0.008894969709217548 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.80it/s]
Test set: Average loss: 0.0248, Accuracy: 9920/10000 (99.20%)

```



##### Final Accuracy  - 99.51

> **99.4 Achieved at 14th Epoch**