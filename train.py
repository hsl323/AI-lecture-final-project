# coding: utf-8
# 2020/인공지능/final/B511157/이현수
import sys, os
import argparse
import time
sys.path.append(os.pardir)

import numpy as np
from AReM import *
from model import *
from matplotlib import pyplot as plt

class Trainer:
    """
    ex) 200개의 훈련데이터셋, 배치사이즈=5, 에폭=1000 일 경우 :
    40개의 배치(배치당 5개 데이터)를 에폭 갯수 만큼 업데이트 하는것.=
    (200 / 5) * 1000 = 40,000번 업데이트.

    ----------
    network : 네트워크
    x_train : 트레인 데이터
    t_train : 트레인 데이터에 대한 라벨
    x_test : 발리데이션 데이터
    t_test : 발리데이션 데이터에 대한 라벨
    epochs : 에폭 수
    mini_batch_size : 미니배치 사이즈
    learning_rate : 학습률
    verbose : 출력여부

    ----------
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 learning_rate=0.01, verbose=True):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = int(epochs)
        self.batch_size = int(mini_batch_size)
        self.lr = learning_rate
        self.network.lr = self.lr #network의 lr을 설정해줍니다.
        self.verbose = verbose
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = int(max(self.train_size / self.batch_size, 1))
        self.max_iter = int(self.epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []


    def train_step(self):
        # 렌덤 트레인 배치 생성
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 네트워크 업데이트
        self.network.update(x_batch, t_batch)
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            train_acc, _ = self.accuracy(self.x_train, self.t_train)
            test_acc, _ = self.accuracy(self.x_test, self.t_test)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print(
                "=== epoch:", str(round(self.current_epoch, 3)), ", iteration:", str(round(self.current_iter, 3)),
                ", train acc:" + str(round(train_acc, 3)), ", test acc:" + str(round(test_acc, 3)), ", train loss:" + str(round(loss, 3)) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc, inference_time = self.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc) + ", inference_time:" + str(inference_time))

    def accuracy(self, x, t):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0
        start_time = time.time()

        for i in range(int(x.shape[0] / self.batch_size)):
            tx = x[i * self.batch_size:(i + 1) * self.batch_size]
            tt = t[i * self.batch_size:(i + 1) * self.batch_size]

            y = self.network.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        inference_time = (time.time() - start_time) / x.shape[0]

        return acc / x.shape[0], inference_time



    # 데이터셋 탑재
(x_train, t_train), (x_test, t_test) = load_AReM(one_hot_label=False)


data_mean = x_train.mean(axis=0) #standard scaling 부분으로 값을 mean,std를 저장해 둡니다.
data_std = x_train.std(axis=0)
x_train = (x_train-data_mean)/(data_std)
x_test = (x_test-data_mean)/(data_std)

    #아래 부분은 mean_max scaling으로 standard scaling이 더 정확도가 높아 위의 식을 사용하였습니다.

#x_train = (x_train-x_train.max(axis=0))/(x_train.max(axis=0)-x_train.min(axis=0))
#x_test = (x_test-x_test.max(axis=0))/(x_test.max(axis=0)-x_test.min(axis=0))

    # 아래는 하이퍼 파라미터 탐색을 위해 사용한 코드입니다.
    
"""
hidden_layer_num = 3
hidden_size = [50,50,50]
result = []
for idx in range(0,1): #하이퍼 파라미터 탐색을 위한 for문으로 탐색을 진행할때는 range를 (0,20)으로 진행하였습니다.
    #lr = 10 ** np.random.uniform(-5,-2) #하이퍼 파라미터 탐색에 사용하였습니다.
    lr = 0.0017019024512130883 # 하이퍼 파라미터 탐색으로 뽑힌 정확도가 높은 lr값입니다.

    #모델을 초기화 하며 lr, hidden_layer정보 등을 넘겨줍니다.
    network = Model(lr,hidden_layer_num,hidden_size,Sigmoid,scaled = True,
                    scaler_info = [data_mean,data_std])

        # 트레이너 초기화
        
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                          epochs = 280, mini_batch_size=100,
                          learning_rate=lr, verbose=True)

        # 트레이너를 사용해 모델 학습
    trainer.train()
    tacc,_ = trainer.accuracy(x_test,t_test)
    result.append([tacc,lr])
    
"""
print(x_test.shape[0])

if __name__ == '__main__':
    #변경한 default 값은 위의 hyperparameter 탐색 주석처리 부분에서 구한 결과입니다.
    parser = argparse.ArgumentParser(description="train.py --help 로 설명을 보시면 됩니다."
                                                 "사용예)python train.py --sf=myparam --epochs=10")
    parser.add_argument("--sf", required=False, default="params.pkl", help="save_file_name")
    parser.add_argument("--epochs", required=False, default=250, help="epochs : default=250")
    parser.add_argument("--mini_batch_size", required=False, default=100, help="mini_batch_size : default=100")
    parser.add_argument("--learning_rate", required=False, default=0.0017019024512130883, help="learning_rate : default=0.0017019024512130883")
    args = parser.parse_args()
    
    # 모델 초기화
    network = Model(scaled = True, scaler_info = [data_mean,data_std])

    # 트레이너 초기화
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=args.epochs, mini_batch_size=args.mini_batch_size,
                      learning_rate=args.learning_rate, verbose=True)

    # 트레이너를 사용해 모델 학습
    trainer.train()
    
    # 파라미터 보관
network.save_params("params.pkl")
print("Network params Saved ")
network.load_params("params.pkl")

