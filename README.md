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

Link: [Activity Recognition system based on Multisensor data fusion (AReM) Data Set][datasetlink]

[datasetlink]: https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+(AReM)

Model Use
----------
model import

  from model import *
  
Parameter

  net = Model( <Learning rate> , <Hidden Number of Layer> , <Hidden_size> , <Activation> , <Dropout rate> , <Scaled> , <Scale Info> )
 
Number of Layer : Network의 Hidden Layer 수
Hidden_size : Hidden Layer의 노드수 (Python list 형태)
