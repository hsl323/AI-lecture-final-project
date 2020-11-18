# coding: utf-8
# 2020/인공지능/final/B511157/이현수
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#activation 계층으로 아래 activation 중 최종적으로 가장 정확도가 높았던 sigmoid를 사용하였습니다.

class Sigmoid: #sigmoid activation 계층입니다.
    def __init__(self):
        self.out =None
    def forward(self,x):#forward로 input x에 sigmoid 연산을 진행합니다.
        mask = -x > np.log(np.finfo(type(x[0][0])).max) #sigmoid 연산 적용시 overflow가 발생하는것을 막기 위해 적용하였습니다.
                                                        #x의 값중 np.log(np.finfo(type(x[0][0])).max) 보다 큰 값을 mask로 저장합니다.
        tmp = x.copy()
        tmp[mask] = 0 # overflow를 막기위해 만든 mask 를 0으로 만들어 제거후 tmp 변수에 넣어줍니다.
        self.out = 1/(1+np.exp(-tmp)) #이후 sigmoid 연산을 진행합니다.
        return self.out
    def backward(self,dout): #sigmoid의 backward연산으로 dout * sigmoid*(1-sigmoid) 연산에 해당합니다.
        return dout*self.out*(1-self.out)
    
class LRelu: #Leaky Relu activation 계층입니다.
    def __init__(self):
        self.mask = None

    def forward(self, x): #forward 연산으로 x가 0보다 큰값은 그대로 0보다 작은값은 0.01 을 곱하여 내보냅니다.
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] *= 0.01
        return out

    def backward(self, dout): #backward 연산으로 forward시 저장해두었던 mask 를 이용 mask 에 해당하는 값에 0.01을 곱해줍니다.
        dout[self.mask] *= 0.01 #mask 에 해당하는 값은 forward 시 0.01을 곱해주었던 x들입니다.
        dx = dout
        return dx
    
class Relu: #Relu activation 계층입니다.
    def __init__(self):
        self.mask = None

    def forward(self, x): #x가 0보다 큰 값은 그대로 보내주며 작은값은 0으로 만듭니다. 0보다 작은 값을 0으로 만든다는것을 제외하고는 Leaky Relu와 동일합니다.
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):#backward 시에도 mask에 해당하는 값들을 0 으로 만들어줍니다.
        dout[self.mask] = 0
        dx = dout
        return dx

#모델의 layer들입니다.

class Scaler: #scaler 계층으로 들어온 input data들을 falg를 확인 scaling이 되어 있지 않다면 scaling을 진행합니다.
    def __init__(self,mean,std,scale_flag = False):
        self.scale_flag = scale_flag #input data가 scaling이 되어있는지 아닌지 확인용입니다.
        self.mean = mean #input data가 scaling되어 있었다면 scaling할때 진행한 mean과 std를 저장합니다.
        self.std = std #이는 이후에 테스트할 데이터에도 똑같은 범위의 scaling을 진행하기 위함입니다.
    def forward(self,x):
        if not self.scale_flag: #scaling이 되어있지 않고
            if self.mean.any() ==None: # 미리 받은 mean값이 없다면
                self.mean = x.mean(axis=0) #들어온 데이터에 대해 scaling을 진행합니다.
                self.std = x.std(axis=0)
            x = (x-self.mean)/self.std #standard scaling을 진행하였습니다.
        return x

    def backward(self,dout): #scaling으로 backward시에 할일은 없지만 layer형식을 맞춰주기위해 추가했습니다.
        pass
    
class Affine: #affine 계층으로 dot 연산과 + 연산을 진행합니다.
    def __init__(self, W, b):
        self.W = W #w와 b를 받아 저장해둡니다.
        self.b = b
        self.x = None
        self.dW,self.db = None,None

    def forward(self, x):
        self.x = x
        out = np.dot(x,self.W) + self.b #forward 연산으로 np.dot과 +연산을 진행합니다.

        return out

    def backward(self, dout): #backward연산으로 들어온 dout과 w를 dot연산을 이용하여 dx를 계산합니다.
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout) #dw 연산은 x와 dout의 np.dot 연산을 진행합니다.
        self.db = np.sum(dout,axis=0) # np.dot 연산의 .T는 배열의 차원을 맞춰주기 위함입니다.

        return dx

class SoftmaxWithLoss: #softmax with loss 계층입니다.
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x) #들어온 데이터에 대해 softmax와 cross_entropy_error를 진행합니다.
        self.loss = cross_entropy_error(self.y,t) 

        return self.loss

    def backward(self, dout=1): 
        batch_size = self.y.shape[0]
        if self.t.ndim == 1:
            encoding = np.eye(6)[self.t] #입력받은 t가 onehot encoding이 안되어있을때를 위함으로
            self.t = encoding         #t에 onehot encoding을 해줍니다.
        dx = (self.y- self.t) / batch_size #dx 는 softmax with loss의 backward식인 (y-t)/batch_size를 구현하였습니다.

        return dx

class Dropout: #dropout 계층입니다. 정확도가 떨어지는 문제가 있어 최종적으로는 사용하지 않았습니다.
    def __init__(self,dropout_rate = 0.05):
        self.train_flg = False #drop out을 적용시킬지 결정하는 flag로 train시에만 true가 됩니다.
        self.dropout_rate = dropout_rate #droup out을 시킬 확률입니다.
        self.mask = None
    def forward(self,x):
        if self.train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate #np.random.rand로 나온 결과값이 droupout_rate보다 높은 부분만 사용합니다.
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate) #flag가 true가 아닐시 1-rate만 곱하여 내보냅니다.
    def backward(self, dout):
        return dout * self.mask #forward시 사용하지 않았던 노드들에는 dout을 전달하지 않습니다.
    
class SGD: #stocastic gradiant descent입니다.
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] #grad에 lr을 곱하여 진행합니다.


class CustomOptimizer: #rmsprop 입니다.
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
        self.saved = None
        self.r = 0.7
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            self.saved = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val) #이전의 기울기 정보를 저장할 dictionary 입니다.
                
        for key in params.keys():
            if key == 'mean' or key == 'std':
                continue
            self.h[key] = self.r*self.h[key] + (1-self.r)*(grads[key]**2) #이전 data들의 기울기 정보를 저장하며 이전값들에 대해 비중을 적게두기위해
            params[key] -= (self.lr * grads[key])/np.sqrt(self.h[key]+1e-7)#decay (self.r)을 이용합니다.


class MOpt: #momentum optimizer입니다.
    def __init__(self,lr = 0.01,momentum = 0.9):
        self.lr = lr
        self.v = None
        self.momentum = momentum
        self.saved = {}
    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            if key == 'mean' or key == 'std':
                continue
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] #momentum optimizer로 v의 값에 속도를 저장하여 가속도를 조절합니다.
            params[key] += self.v[key] #momentum 정도는 일반적으로 많이 사용하는 0.9로 하였습니다.

class Adam: #adam optimizer입니다. momentum 과 rmsprop를 합쳐논 개념입니다.
    def  __init__(self,lr = 0.01):
        self.lr = lr
        self.v = None
        self.h = None
        self.momentum = 0.9 #momentum 과 decay 입니다.
        self.decay = 0.999
        self.iter = 1
        self.saved = {}
    def update(self,params,grads): #adam 구현은 논문의 psudo 코드를 보고 구현하였으며 보고서에 자세히 설명하겠습니다.
        if self.v is None:
            self.v = {} #속도 v와 이전 기울기값 h를 저장하기 위한 변수입니다.
            self.h = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        self.iter += 1 
        for key in params.keys():
            if key == 'mean' or key == 'std': #params들 중 mean과 std는 건너 뜁니다.
                continue
            self.v[key] = (self.momentum*self.v[key] + (1-self.momentum) * grads[key])
            tmpv = self.v[key]/(1-self.momentum**self.iter)
            self.h[key] = (self.decay*self.h[key] + (1-self.decay)* (grads[key]**2))
            tmph = self.h[key]/(1-self.decay**self.iter)
            params[key] -= (self.lr*tmpv)/(np.sqrt(tmph) + 1e-7) #분모가 0이 되는것을 막기위한 1e-7입니다.

class Model:
    """
    네트워크 모델 입니다.

    """
    #model은 init으로 lr과 layer_num, hidden_size, Activation, droupout_rate,scale정보,scaler시 사용한 정보
    #를 받으며 이는 train시 여러 값들을 변화시키며 돌릴때의 편리성을 위해 추가해주었습니다.
    #모두 default 값이 있어 넣어주지 않아도 정상 작동합니다.
    def __init__(self, lr=0.0017019024512130883,layer_num = 3,hidden_size = [50,50,50],Activation = Sigmoid,
                 dropout_rate = 0.2,scaled=False,scaler_info = []):
        """
        클래스 초기화
        """
        
        self.params = {} #params를 저장할 dictionary 입니다.
        #self.saved_cost= [] cost의 추이를보기위해 추가했었던 변수입니다.
        self.layer_num = layer_num #hidden layer의 개수를 저장하는 변수입니다.
        self.input_size = 6
        self.output_size = 6
        self.lr = lr
        self.hidden_size = hidden_size #hidden layer의 각각의 node수를 저장합니다.
        self.hidden_size.append(self.output_size)
        #layer 를 초기화할때 맞춰주기 위해 마지막에 output layer의 node수를 붙여주었습니다.
        self.Activation = Activation
        #입력받은 Activation 함수를 이용하며 default 는 sigmoid 입니다. 
        self.scaled = scaled
        #데이터의 scaling 유무를 확인하는 변수입니다.
        self.scaler_info = scaler_info #scaling 되어있다면 scaling시 사용한 mean과 std 정보를 받습니다.
        #self.dropout_rate = dropout_rate drop out 시 사용한 변수입니다.
        
        if self.Activation == Relu or self.Activation == LRelu: #activation에 따라 w초기값을 정해주기 위함입니다.
            self.initializer = 4 #relu의 경우 (4/(n_in+n_out))
        else:
            self.initializer = 2 #sigmoid의 경우(2/(n_in+n_out))입니다.
        #위의 두 식은 uniformed 된 초기화로 기본 공식과는 조금 다릅니다.
        #기본 식은 he 초기화는 2/(n_in) 을 사용하며 xavier는 1/(n_in) 입니다.

        self.__init_weight()
        self.__init_layer()
        
        self.optimizer = Adam(self.lr) #optimzer 로 adam을 사용합니다.
        
    def __init_layer(self):
        """
        레이어를 생성하시면 됩니다.
        """
        self.layers = OrderedDict() #ordered dict 는 순서가 있는 dictionary로 forward 와 backward시 for문으로 편하게 돌리기 위함입니다.
        self.layers['scaler'] = Scaler(self.params['mean'],self.params['std'],self.scaled) #scaling layer를 scaling 정보를 이용해 초기화 해줍니다.
        for idx in range(self.layer_num): #각 affine layer와 activation layer를 params로 초기화 해줍니다.
                self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'],self.params[f'b{idx}'])
                self.layers[f'Activation{idx}'] = self.Activation()
                #self.layers[f'Dropout{idx}'] = Dropout(self.dropout_rate)
                #dropout 은 정확도가 낮아져 사용하지 않고 다른방법을 사용하였습니다. epoch를 줄이는 방식으로 하였으며 길어질 것 같아 보고서에 작성하겠습니다.
        #마지막 layer는 activation이 아닌 softmax이므로 affine을 포문 밖에서 한번더 해주었습니다.
        self.layers[f'Affine{self.layer_num}'] = Affine(self.params[f'W{self.layer_num}'],self.params[f'b{self.layer_num}'])
        self.last_layer = SoftmaxWithLoss() #softmax with loss layer를 초기화 해줍니다.
        
        return

    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        
        size = self.input_size
        
        if len(self.scaler_info)==2: #미리 받은 scaling 정보가 있다면 params 에 기록합니다.
            self.params['mean'] = self.scaler_info[0]
            self.params['std'] = self.scaler_info[1]
        else:
            self.params['mean'], self.params['std'] = None, None
        for idx in range(self.layer_num+1): #w 와 b를 초기화 해주며 이때 위에서 설정했던 initializer에 따라 he 또는 xavier 초기화를 진행합니다.
            self.params[f'W{idx}'] = np.random.randn(size,self.hidden_size[idx]) * np.sqrt(self.initializer/(size+self.hidden_size[idx]))
            self.params[f'b{idx}'] = np.zeros(self.hidden_size[idx],dtype =np.float64)
            size = self.hidden_size[idx] #for문으로 반복해서 초기화해주기 위해 size 변수를 이용 이전 노드의 노드수를 저장하게 하였습니다.

        return

    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        #for idx in range(self.layer_num):
            #self.layers[f'Dropout{idx}'].train_flg = True
        
        grads = self.gradient(x, t) #gradient를 계산하는 함수를 호출합니다.
        
        self.optimizer.update(self.params, grads) #계산한 gradient로 optimizer를 이용해 w,b를 업데이트 합니다.
        
        #for idx in range(self.layer_num):
            #self.layers[f'Dropout{idx}'].train_flg = False
            
    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        
        for layer in self.layers.values():
            x = layer.forward(x) #forward를 진행하여 결과값 x를 받아옵니다.
        return x #위의 계산으로 나온 x를 반환합니다.

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x) #predict로 y값을 계산합니다.
        return self.last_layer.forward(y, t) #softmax with loss 의 return 값이 loss값임을 이용합니다.

    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward
        #loss 계산을 위해 forward를 진행합니다.        
        self.loss(x,t)
        # backward
        dout = self.last_layer.backward(1)
        #softmax with loss에 저장된 t,y값을 이용해 dout을 계산합니다.
        layers = list(self.layers.values())
        layers.reverse() #backward는 역순으로 진행하기때문에 list를 reverse 해줍니다.
        for layer in layers:
            dout = layer.backward(dout) #layer 차례차례 backward를 계산합니다.

        
        # 결과 저장
        grads = {}
        for idx in range(self.layer_num+1):
            grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW # backward이후 affine 계층이 가지고 있는 dw,db 값으로
            grads[f'b{idx}'] = self.layers[f'Affine{idx}'].db # w와 b의 gradient 를 받아옵니다.
            
        return grads #위에서 받은 gradient 값을 반환합니다.

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        self.__init_layer() #불러온 params를 이용하여 layer를 초기화 해줍니다.
        pass
