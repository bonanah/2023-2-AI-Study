### 1. AI, ML, DL
- AI > ML > DL
ai : 인간의 능력을 인공적으로 구현한 것
ml : 데이터로부터 규칙을 학습하는 ai의 하위분야
dl : neural network 를 기반으로 하는 ml의 하위 분야

### 2. DL Component 
#### Data
- Dataset :: Classification, Semantic Segmentation, Object Detection, Pose Estimation 
- Model :: Input \to feature \to Ouput 
eg. AlexNet, GoolgLeNet, ResNet, DenseNet, LSTM, AutoEncoder, GAN .etc
+ LSTM : 모델 안에 gate 있음 (4개) - input, out put, forget, update


- Loss Function : the difference of the ....
this is dependent on the task
1. Regression :  (... error )
2. Classification : Cross Entropy ()
3. Probabilicstic : MLE (maximum likelihood estimation)
- Optimizer & Regularization 

### 3. Neural Network
Function Approximators ::
Input layer -> Hidden Layer * N -> Output layer

### 4. Nonlinear Function
#### Activation Function :::: Nonlinear Function
- activation function :: hidden layer 와 ouput layer의 neuron에서 output을 결정하는 함수 
- If activation function is a linear function, it can be replaced by 1 activation function even if multiple layers are stacked 
(능력 올리려면 hidden layer 추가해야하는데 추가해도 선형임 )

Step Function : perceptron
sigmoid function : binary classification 
(input -> forward propagation)
hyper tangent function : sigmoid function이랑 비슷, x = 0에서의 기울기가 sigmoid 보다 큼 (기울기 소실이 훨씬 적음)
ReLU : hidden layer에서 주로 쓰임 (음수면 0), 단순 임계값, 연산 속도 빠름 
Leaky ReLU : hyperparameter 도 들어감, 입력값이 음수일 대 아주 작은 수로 반환 
Softmax Function : multiclassification, 주로 output layer에서 사용함 


### 5. Multi-Layer Perceptron 
hidden layer 가 여러 개 

### 6. Generalizaion 
Generalization gap = |Test Error - Training Error|
iteration이 증가할수록  generalization gap 줄어들었다가 다시 늘어나게 되는 경향 ---- overfitting 될 수도 

- Under-fitting
- Optimal Balance
- Over-fitting (training data set 에 너무 잘 맞아떨어져서 실제 전체 data set에서의 정확성 문제될 수 있음 )

Cross Validatoin (교차 검증)
- hold out about 10~30%
- k-fold cross validation 
train data 를 k 개로 분리 후 일정 데이터들을 뽑아 학습 잘 되어가고 있는지 확인 (valid data)
cross imbalance 비슷하게 설정

Ensemble
- 여러 개의 분류 모델을 조합 (성능 향상)
1. Bagging : subset 나누어서 학습 - 각각의 voting , averaging 구함(parallel)
2. Boosting : 학습이 끝나지 않은 데이터들을 모아 새로운 간단한 모델로 재학습 (sequential)

Regularization --- 학습 방해
1. Early Stopping
2. parameter norm pernalty
3. data augmentation : 데이터를 다양한 형태로 변형하여 학습 시킴
+) MNIST , Mixup, Cutmix
4. Noise Robustness (Gaussian nosie, laplace noise)
5. Drop out : 몇 개의 신경망의 노드를 지나지 않고 학습, training 에서만 적용하는 방법, test 에서는 모든 노드가 참여함 
6. Label Smoothing : over fitting 방해하기 위해서 조절함 :: label의 값을 0, 1 로 분류하는 것이 아니고 (hard labeling X) alpha 값 설정하여 soft 하게 labeling 함 

### 7. Convolutional Neural Networks (CNN)

각 레이어의 입출력 데이터의 형상 유지
이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식
복수의 필터로 이미지의 특징 추출 및 학습
추출한 이미지의 특징을 모으고 강화하는 Pooling 레이어
필터를 공유 파라미터로 사용하기 때문에, 일반 인공 신경망과 비교하여 학습 파라미터가 매우 적음

- space information of image can be store in the matrix (Fully Connected Multi Layered Neural Network ignores the relation between adjacent pixels)
(FC 는 모든 hidden layer와 output layer를 연결함, keras에서는 dense...)
- vectorizing images \to 3dimensional matrix space 
- has channels, and these are used as parameters (reduce the number of parameters)
- Convolution Computation -> Activation Function -> Pooling *  n times => Fully Connected Layer (last step)

- Pooling Compuation (kernerl size, stride)
1. Max Poolling : take the highest value from the area covered by the kernel
2. Average Pooling : calculate the average value from the area covered by the kernel 


+) RNN : time series
문제점 : catastrophic forgetting, vanishing gradient 

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

#### GoogLeNet
- convolutional filter size = 1 * 1
- complex structure (about 20 layers)

#### ResNet
- Human Error < ResNet Error (3.57%)
- more than 20 layers cause the degradation problem & Gradient Vanishing & Explosion
- ResNet \to Residual Learning (= Skip Connection = Shortcut)
$$ F(x) := H(x) - x, H(x) = F(x) + x $$
