'''
========================
A BASIC NEURAL NETWORK
========================
'''

'''
------------------------
ENGLISH
------------------------
A NEURON DIAGRAM
x -> [sigma(wx+b)] -> y_pred

SYMBOLS
[] : Neuron
x : Input of the neuron
sigma : activation funtion, using sigmoid function sigma(x) = (1/1+exp(-x))
w : weight in the neuron
b : bias in the neuron
y_pred : Input of the neuron

A NEURAL LAYER DIAGRAM
xi -> [sigma[Wji * xi + bj]] -> y_predj

SYMBOLS
[] : Neural layer
xi : Input vector with i*1 dimension
sigma : activation funtion, using sigmoid function sigma(x) = (1/1+exp(-x))
Wji : Weight matrix in this layer with j*i dimension
bj : bias vector in this layer with j*1 dimension
y_predj : Output vector of this layer with j*1 dimension

CLASSES
Input_layer : Input vector layer class
            : Contains Input vector, xi
            : Cannot be trained
            : Has 'forward()' function to feed output to next layer

Output_layer : Output vector layer class
             : Contains Output vector, y_predj
             : Contains Loss, (yj - y_predj)**2
             : Contains Gradient of Loss respects to y_predj, 2*(yj - y_predj)*-1
             : Cannot be trained
             : Has 'forward()' function to create Output vector, y_predj
             : Has 'loss_func()' function to calculate Loss and Gradient of Loss

Sigmoid_layer : Activation function Layer class
              : Contains input from previous layer, zj
              : Contains output from this layer, sigma(z) = 1/(1+np.exp(-(z)))
              : Contains Local Gradient of this layer, local_grad
              : Cannot be trained
              : Has 'forward()' function to feed output to next layer
              : Has 'local_gradient()' Local Gradient of this layer

FC_layer : Fully connected Neural Layer class
         : Contains input from previous layer, ai
         : Contains Weight matrix, Wji
         : Contains bias vector, bj
         : Contains output from this layer, zj = [Wji * ai + bj]
         : Contains Gradient of upper layer respects to Wji, w_grad
         : Contains Gradient of upper layer respects to bj, b_grad
         : Contains Local Gradient of this layer, local_grad
         : Can be trained (using chain rule)
         : Has 'forward()' function to feed output to next layer
         : Has 'local_gradient()' Local Gradient of this layer

NeuralNet : Neural Network class
          : Contains an Input_layer
          : Contains FC_layers
          : Contains Sigmoid_layers
          : Contains an Output_layer
          : Contains Loss_store list to store loss while training
          : Has 'forward()' function to
            - feed input xi to Input_layer
            - feed output from Input_layer to next FC_layer
            - feed output from FC_layer to next Sigmoid_layers
            - feed output from Sigmoid_layers to Output_layer
            - return Output vector, y_predj
          : Has 'backpropagation()' function to
            - use forward() to calculate y_predj from xi
            - use Output_layer to calculate Loss and Gradient of Loss
            - use Gradient of Loss to calculate Local Gradient of previous layer
            - use Local Gradient to calculate Local Gradient of next previous layer until every layers has Local Gradients
          : Has 'train()' function to
            - use backpropagation() for a number of epoch
------------------------
日本語
------------------------
ニューロン図
x -> [sigma(wx+b)] -> y_pred

記号
[] : ニューロン
x : ニューロンの入力
sigma : アクティベーション機能、シグモイド関数の使用 sigma(x) = (1/1+exp(-x))
w : ニューロンの重量
b : ニューロンのバイアス
y_pred : ニューロンの出力

ニューラルネットワーク層図
xi -> [sigma[Wji * xi + bj]] -> y_predj

記号
[] : ニューラルネットワーク層
xi : i * 1次元の入力ベクトル
sigma : シグモイド関数を使用したアクティブ化関数 sigma(x) = (1/1+exp(-x))
Wji : このレイヤーのj * i次元の重み行列
bj : j * 1次元のこのレイヤーのバイアスベクトル
y_predj : j * 1次元のこのレイヤーの出力ベクトル

CLASSES
Input_layer : 入力ベクターレイヤークラス
            : 入力ベクトルを含む, xi
            : 訓練することはできない
            : 出力を次のレイヤーに送る 'forward（）'関数がある

Output_layer : 出力ベクターレイヤークラス
             : 出力ベクトルが含む, y_predj
             : 損失が含む, (yj - y_predj)**2
             : y_predjに対する損失の勾配の考慮が含む, 2*(yj - y_predj)*-1
             : 訓練することはできない
             : 出力ベクトルを作成するための「forward（）」関数がある, y_predj
             : 損失と損失の勾配を計算するための「loss_func（）」関数がある

Sigmoid_layer : アクティベーション機能レイヤークラス
              : 前のレイヤーからの入力が含む, zj
              : このレイヤーからの出力が含む, sigma(z) = 1/(1+np.exp(-(z)))
              : このレイヤーのローカルグラデーションが含む, local_grad
              : 訓練することはできない
              : 出力を次のレイヤーに送る 'forward（）'関数がある
              : このレイヤーの「local_gradient（）」ローカルグラデーション

FC_layer : 完全に接続されたニューラルネットワーク層クラス
         : 前のレイヤーからの入力が含む, ai
         : 重量マトリックスが含む, Wji
         : バイアスベクトルを含む, bj
         : このレイヤーからの出力が含む, zj = [Wji * ai + bj]
         : Wjiに対する上位層の勾配を含む, w_grad
         : bjに対する上位層の勾配を含む, b_grad
         : このレイヤーのローカルグラデーションが含む, local_grad
         : トレーニング可能（連鎖法則を使用）
         : 出力を次のレイヤーに送る 'forward（）'関数がある
         : このレイヤーの「local_gradient（）」ローカルグラデーション

NeuralNet : ニューラルネットワーククラス
          : Input_layerが含む
          : FC_layersを含む
          : Sigmoid_layersを含む
          : Output_layerが含む
          : トレーニング中に損失を保存するLoss_storeリストが含む
          : 「forward（）」関数が
            - 入力xiをInput_layerにフィードする
            - Input_layerから次のFC_layerに出力をフィードする
            - FC_layerから次のSigmoid_layersへの出力のフィードする
            - Sigmoid_layersからOutput_layerへの出力のフィードする
            - 出力ベクトル、y_predj
          : 「backpropagation（）」関数が
            - forward（）を使用してxiからy_predjを計算する
            - Output_layerを使用して損失と損失の勾配を計算する
            - 損失の勾配を使用して前の層の局所勾配を計算する
            - ローカルグラデーションを使用して、すべてのレイヤーがローカルグラデーションになるまで、次の前のレイヤーのローカルグラデーションを計算する
          : 「train（）」関数が
            - 多くのエポックにbackpropagation（）を使用します

'''
import numpy as np

class Input_layer:
    def __init__(self,dim):
        # dimension of input(1,dim)
        # 入力の次元（1、dim）
        self.dim = dim
        # Input vector
        # 入力ベクトル
        self.x = np.zeros((self.dim,1))
    def forward(self,x):
        self.x = x
        return self.x

class Output_layer:
    def __init__(self,dim):
        # dimension of output(1,dim)
        # 出力の次元（1、dim）
        self.dim = dim
        # output y_predj
        # y_predjを出力
        self.y_pred = np.zeros((self.dim,1))
        # Loss
        # 損失
        self.loss = np.zeros((self.dim,1))
        # Gradient of Loss
        # 損失の勾配
        self.loss_grad = np.zeros((self.dim,1))
    def forward(self,y_pred):
        self.y_pred = y_pred
        return self.y_pred
    def loss_func(self,y):
        self.loss = (np.power(y-self.y_pred,2))
        self.loss_grad = -2*(y-self.y_pred)
        return self.loss,self.loss_grad

class FC_layer:
    def __init__(self,dim_in,dim_out):
        np.random.seed(99)
        # dimension of input(1,dim)
        # 入力の次元（1、dim）
        self.dim_in = dim_in
        # dimension of output(1,dim)
        # 出力の次元（1、dim）
        self.dim_out = dim_out
        # input from previous layer, ai(1,dim)
        # 前の層からの入力、ai（1、dim）
        self.x_vector = None
        # Weight matrix, Wji
        # 重み行列、Wji
        self.w_martix = np.random.rand(self.dim_out,self.dim_in)-0.5
        # bias vector, bj
        # バイアスベクトル、bj
        self.b_vector = np.random.rand(self.dim_out,1)-0.5
        # output from this layer, zj = [Wji * ai + bj]
        # このレイヤーからの出力、zj = [Wji * ai + bj]
        self.y_vector = None
        # learning rate
        # 学習率
        self.lr = 0.15
    def forward(self,x):
        self.x_vector = x
        self.y_vector = np.dot(self.w_martix,x) + self.b_vector
        return self.y_vector
    def local_gradient(self,y_grad):
        # Gradient of upper layer respects to Wji
        # Wjiに対する上位層の勾配
        w_grad = np.dot(y_grad,self.x_vector.T)
        # Gradient of upper layer respects to bj
        # bjに対する上位層の勾配
        b_grad = y_grad
        # Local Gradient of this layer
        # このレイヤーのローカルグラデーション
        local_grad = np.dot(self.w_martix.T,y_grad)
        # Update Wji and bj
        # Wjiとbjを更新
        self.w_martix -= self.lr*w_grad
        self.b_vector -= self.lr*b_grad
        return local_grad

class Sigmoid_layer:
    def __init__(self,dim):
        # dimension of input(1,dim)
        # 入力の次元（1、dim）
        self.dim = dim
        # output from this layer, sigma(z)
        # このレイヤーからの出力、sigma（z）
        self.y_vector = np.zeros((self.dim,1))
        # Local Gradient of this layer
        # このレイヤーのローカルグラデーション
        self.local_grad = np.zeros((self.dim,1))
    def forward(self,x):
        for i in range(len(x)):
            self.y_vector[i] = 1/(1+np.exp(-(x[i])))
        return self.y_vector
    def local_gradient(self,y_grad):
        for i in range(len(y_grad)):
            self.local_grad[i] = (self.y_vector[i]*(1-self.y_vector[i]))*y_grad[i]
        return self.local_grad

class NeuralNet():
    def __init__(self,input_dim,output_dim):
        self.input = Input_layer(input_dim)
        self.hidden1 = FC_layer(self.input.dim,5)
        self.sigmoid1 = Sigmoid_layer(self.hidden1.dim_out)
        self.hidden2 = FC_layer(self.sigmoid1.dim,5)
        self.sigmoid2 = Sigmoid_layer(self.hidden2.dim_out)
        self.hidden3 = FC_layer(self.sigmoid2.dim,output_dim)
        self.sigmoid3 = Sigmoid_layer(self.hidden3.dim_out)
        self.output = Output_layer(output_dim)
        self.loss_store = []

    def forward(self,x):
        x_1 = self.input.forward(x)
        x_2 = self.hidden1.forward(x_1)
        x_3 = self.sigmoid1.forward(x_2)
        x_4 = self.hidden2.forward(x_3)
        x_5 = self.sigmoid2.forward(x_4)
        x_6 = self.hidden3.forward(x_5)
        x_7 = self.sigmoid3.forward(x_6)
        out = self.output.forward(x_7)
        return out

    def backpropagation(self,x,y):
        for i in range(len(x)):
            self.forward(x[i])
            loss,loss_grad = self.output.loss_func(y[i])
            s3_grad = self.sigmoid3.local_gradient(loss_grad)
            h3_grad = self.hidden3.local_gradient(s3_grad)

            s2_grad = self.sigmoid2.local_gradient(h3_grad)
            h2_grad = self.hidden2.local_gradient(s2_grad)

            s1_grad = self.sigmoid1.local_gradient(h2_grad)
            h1_grad = self.hidden1.local_gradient(s1_grad)
            self.loss_store.append(loss)

    def train(self,x,y,epoch):
        for i in range(epoch):
            self.backpropagation(x,y)
            print('Epoch {} loss {}'.format(i+1,sum(self.loss_store)/len(self.loss_store)))
            self.loss_store = []

# train on funtion y = sigmoid(5*(x**2) - 4 + x + x**3) : A Neural Network can solve a sigmoid of non-linear function
# 関数の学習y = sigmoid（5 *（x ** 2）-4 + x + x ** 3）：ニューラルネットワークは非線形関数のシグモイドを解くことができます
x_train = []
y_train = []
# generate train data
# トレーニングデータを生成する
for i in range(20000):
    x_train.append(np.random.randn(1,1))
    y= 5*(x_train[-1]**2) - 4 + x_train[-1] + x_train[-1]**3
    y_train.append(1/(1+np.exp(-y)))
# Create NeuralNetwork
# ニューラルネットワークの作成
My_NN = NeuralNet(1,1)
# Train NeuralNetwork
# ニューラルネットワークの学習
epoch = 10
My_NN.train(x_train,y_train,epoch)
# Check result
# 結果を確認
index = np.random.randint(0,len(x_train))
x = x_train[index]
y = y_train[index]
y_pred = My_NN.forward(x)
print('real    {}'.format(y))
print('predict {}'.format(y_pred))
print('error   {}'.format(abs(y-y_pred)/y))











