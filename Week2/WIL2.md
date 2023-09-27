# Intro CV & PyTorch

## Semantic Segmentation (CNN vs FCN) 
### convolutional & fully connected layer
Segmentation : 입력 이미지를 픽셀 수준에서 분석하여 각 픽셀에 클래스 레이블을 할당함 
- Semantic Segmentation : 큰 범위에서 객체 분류 
- Instance Segmentation : instance 마다 label 붙여서 분류

Fully Connected Layer의 경우 공간적 정보를 무시한 벡터로 표시 
Convolutional Layer의 경우 output의 형태가 공간적 정보를 보존하여 upsampling 할 때에 input과 같은 크기의 출력 생산 가능함 

Upsampling : 보강법 --- 디테일하게 표현 
- Deconvolution 
- interpolation 

## Object Detection 
object detection : bounding box로 객체의 위치를 찾는 task

### R-CNN
하나의 이미지마다 convolution 해야함 -- region proposal 만큼 수행해야하기 때문에 시간이 오래 걸림 
1. Selective Search --- input images
2. Region size process --- extract region proposals (about 2,000)
3. Compute CNN features : AlexNet 이라는 CNN 사용해서 
4. Linear SVM으로 classify (classify regions) using feature 

### SPPNet 
R-CNN에서 계산을 여러 번 해야하는 것을 한 번으로 줄이는 알고리즘 
영역 크기에 해당하는 피쳐값인 sub tensor만 가져와 사용해서 속도 향상

### Fast R-CNN
bounding box 에 해당하는 공간 벡터를 뜯어와서 spatial pyramid pooling 과정이 느렸음 
1. Selective Search 
2. Compute CNN Feature
3. ROI (regions of interest) Pooling : 패치들을 고정된 feature 값으로 만들기 위한 작업 
4. Neural Network 를 통해 bbox regressor & softmax 수행하면서 해당하는 클래스 라벨을 찾음 
하나의 테트워크 안에서 돌아감 

### Faster R-CNN
Fast R-CNN의 개념에 Region Proposal Network (RPN) 을 도입함 
#### Region Proposal Network (RPN)
- selective search 알고리즘을 대신함, bounding box region 또한 학습 가능한 network, 이미지 안에 특정 영역이 bounding box로 의미가 있을지 없을지 판단해주는 네트워크 
- 비슷한 부분을 합치는 것이 아닌, 해당 영역 안에 물체가 잇는지 없는지 계산 

### YOLO : You Only Look Once
bounding box를 따로 뽑는 region proposal 단계가 없음 (selective search, RPN .etc)

#### YOLO 의 Loss
1. bounding box regression loss
2. confidence loss
3. classificatoin loss

#### 방법
1. 가로, 세로를 동일한 그리드 영역으로 나누기
2. 각 그리드 영역에 대해서 어디에 사물이 존재하는지 bounding box 와 box에 대한 신뢰도 점수를 예측함 (신뢰도 높을수록 굵게 박스를 그려줌)
3. 어떤 사물인지에 대한 classification 작업을 동시에 진행함
4. 굵은 박스들만 남김 (NMX Algorithm : Non-Maximum Suppression)

### Introduction to PyTorch
PyTorch가 사용되는 비중이 높아지는 편임 

#### Tensorflow 와 비교
- Torch 기반
- Meta
- Dynamic Graph
- 실행 순서 : 동작 중에 그래프 정의 / 조작 (Define and Run)

### PyTorch Basics
#### PyTorch Operatoins
1. Tensor : 다차원의 array 표현하는 개념, list 나 ndarray를 이용해 생성 가능
2. numpy like operations : eg) slicing, indexing, flatten, ones_like, numpy, shape, dtype
3. Tensor handling : view (shallow copy), reshape (deep copy)
4. Tensor handling : squeeze & unsqueeze / tensor의 dimension 설정, mm, matmul, broadcasting, nn.functional / softmax & one-hot encoding (0~1을 어떻게 냐누냐에 따라)
5. AutoGrad - backward


### AutoGrad & Optimizer
#### nn.module 
반복된 블럭들로 여러 개의 layer 쌓아나가면서 output 추출함
output에서 다시 역방향으로 돌아오는 backpropagatoin 을 통해서 블럭의 parameter 업데이트 
딥러닝을 구성하는 layer의 base class
input, output, forward, backward 정의
학습의 대상이 되는 parameter(tensor) 정의 
Augogradient의 학습 대상이 되는 tensor : required_grad = True로 지정
backward : 레이어에 있는 parameter들의 미분을 수행. loss를 현재 가중치에 대해 미분 수행한 편미분값을 가중치에서 빼줌 -> parameter 업데이트 


### PyTorch Dataset
#### Dataset class
데이터 입력 형태 정의함, data augmentation 이나 ToTensor는 Transform 이라는 다는 곳에서 정의함 
Data Cleaning (수작업) -> Transform 정의 (전처리, 텐서 처리) -> Dataset Class 
#### Data Loader class
Data의 Batch 생성하는 클래스, batch 처리가 메인 업무
- attribute : batch_size, shuffle, num_workers, drop_last

model.state_dict() :파라미터, checkpoint를 저장하는 경우에 사용 

### Transfer Learing
남이 이미 학습 시킨 모델들을 pretrained model
마지막 레이어만 자신의 데이터로 재학습 
