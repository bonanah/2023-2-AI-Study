### 1. AI, ML, DL
- AI > ML > DL


### 2. DL Component 
#### Data
- Dataset :: Classification, Semantic Segmentation, Object Detection, Pose Estimation 
- Model :: Input \to feature \to Ouput 
eg. AlexNet, GoolgLeNet, ResNet, DenseNet, LSTM, AutoEncoder, GAN .etc
- Loss Function : the difference of the ....
this is dependent on the task
1. Regression : MSE (... error )
2. Classification : Cross Entropy
3. Probabilicstic : MLE (maximum likelihood estimation)
- Optimizer & Regularization 





### 3. Neural Network




### 4. 





### 5.






### 6. 




### 7. Convolutional Neural Networks (CNN)

각 레이어의 입출력 데이터의 형상 유지
이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식
복수의 필터로 이미지의 특징 추출 및 학습
추출한 이미지의 특징을 모으고 강화하는 Pooling 레이어
필터를 공유 파라미터로 사용하기 때문에, 일반 인공 신경망과 비교하여 학습 파라미터가 매우 적음

- space information of image can be store in the matrix (Fully Connected Multi Layered Neural Network ignores the relation between adjacent pixels)
- vectorizing images \to 3dimensional matrix space 
- has channels, and these are used as parameters (reduce the number of parameters)
- Convolution Computation -> Activation Function -> Pooling *  n times => Fully Connected Layer (last step)

- Pooling Compuation (kernerl size, stride)
1. Max Poolling : take the highest value from the area covered by the kernel
2. Average Pooling : calculate the average value from the area covered by the kernel 

### 8. 1 x 1 Convolution 
- this can change the dimension of the depth \to make neural networks deeper 
- check the number of filters of next layer

1. number of Channels
2. Efficiency
3. Non-linearity (using ReLU, f = max(0, x))

### 9. Modern CNN
#### AlexNet 
- two networks
- filter size = 11 x 11
- convolutional layer 5 + dense layer 3
(dense layer : gathering features that we extracted in previous layer into one layer and represent the tensor we want)
- data aug. , drop-out

#### VGGNet
- convolutional filter size = 3 * 3
- reduce the computation in the same receptive field (receptive field number of parameters)
- 


#### GoogLeNet
- convolutional filter size = 1 * 1
- complex structure (about 20 layers)


#### ResNet
- Human Error < ResNet Error (3.57%)
- more than 20 layers cause the degradation problem & Gradient Vanishing & Explosion
- ResNet \to Residual Learning (= Skip Connection = Shortcut)
$$ F(x) := H(x) - x, H(x) = F(x) + x $$
