# AI-lecture-final-project

Deep learning framewark를 이용하는 것이 아닌 Python numpy 모듈을 이용하여 모델을 구현

Arem data를 이용해 적절한 activation, optimizer등 여러 방법을 적용시켜 정확도를 높이는 프로젝트.

구현 환경
--------

python3 

windows 10

Data
----

AReM data

![arem](https://user-images.githubusercontent.com/60774392/99532090-1adff080-29e7-11eb-844a-6ed63dfce874.PNG)

여섯가지 동작을 수행했을 때의 센서값의 평균과 분산을 나타낸다.



class별 갯수

   + walking  : 4291
    
   + standing : 4301
    
   + sitting  : 4311
    
   + lying    : 4342
    
   + cycling  : 4408
    
   + bending  : 3402

Link: [Activity Recognition system based on Multisensor data fusion (AReM) Data Set][datasetlink]

[datasetlink]: https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+(AReM)

Model
------
model import

    from model import *
  
1. Parameter

        Model( <Learning rate> , <Number of Hidden Layer> , <Hidden_size> , <Activation> , <Dropout rate> ) <Scaled> , <Scale Info> )

    Number of Layer : Network의 Hidden Layer 수

    Hidden_size : Hidden Layer의 노드수 (Python list 형태)

    Activation : Activation function (default : Sigmoid)

    Dropout rate : Dropout rate

    Scaled : Scaling 유무 (type : Boolean ,default : False)

    Scale Info : Scale시 사용한 mean, std 값
    
    

2. train

    + Without Scaling
    
        train.py 163'th line
    
            net = Model(lr, <Number of Hidden Layer> , <Hidden_size> , <Activation> , <Dropout rate> )
            
        Batch 단위로 scaling 수행
    

    + With Scaling
    
       train.py 114~117 line (standard scaling)
    
            data_mean = x_train.mean(axis=0) 
            data_std = x_train.std(axis=0)
            x_train = (x_train-data_mean)/(data_std)
            x_test = (x_test-data_mean)/(data_std)
            
       train.py 163'th line
    
            net = Model(lr, <Number of Hidden Layer> , <Hidden_size> , <Activation> , <Dropout rate> , scaled = True, scaler_info = [data_mean,data_std] )
         
       전체 data를 scaling 후 사용한 mean, std 를 이용하여 test data에 동일하게 적용



Final
------

[인공지능 과제 리포트.pdf][link]

[link]: https://github.com/hsl323/AI-lecture-final-project/blob/main/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%20%EA%B3%BC%EC%A0%9C%20%EB%A6%AC%ED%8F%AC%ED%8A%B8.pdf
