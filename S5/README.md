# 

## **Target**

1. 99.4% **(this must be consistently shown in your last few epochs, and not a one-time achievement)**
2. Less than or equal to 15 Epochs
3. Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
4. Do this in exactly 3 steps



# **<u>Model 1</u>**

## **Target**

Reduce the number of parameters as much we can . But accuracy has taken a hit. 

## 	**Result**

```
Total params: 7,690
Trainable params: 7,690
Non-trainable params: 0
```

Best Training Accuracy : Accuracy = 98.70%

Best Testing Accuracy: Accuracy =98.89%

## Analysis

Model is light now. We can see it is stable . It might get good accuracy at more epochs. But we need the result in less epo chs. Now trying normalizing the input data by adding BN.

# **Model 4**

## **Target**

Adding Batch Normalization  so that model can be generalise and get better accuracy .

## **Result**

```
Total params: 7,840
Trainable params: 7,840
Non-trainable params: 0
```

Best Training Accuracy : Accuracy = 99.57

Best Testing Accuracy: Accuracy = 99.61%

## Analysis

Well we got what we needed. But  In model's training accuracy we got above 99.00 after 5th  epochs. This model is good as we can see very less difference between training and testing accuracy.  But the testing  accuracy seems dropping at 10th epochs and  training  accuracy keeps on increasing. If we use dropout it will be tough for the model to learn and in that way we might be able to learn better.

# **Model 5**

## **Target**

Let's try Droupout in the model and see if the model is increasing it's performance and Stability .

## **Result**

```
Total params: 7,840
Trainable params: 7,840
Non-trainable params: 0
```

Best Training Accuracy : Accuracy = 99.27

Best Testing Accuracy: Accuracy =99.68%

## Analysis

After tuning the Model with batch size 64 and learning rate  0.015 we were able to achieve 99.68 . Dropout gave more stable model. 

