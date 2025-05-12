from sklearn.datasets import make_moons
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import asyncio
import os
import csv
import timeit


cp.set_printoptions(
    linewidth=100,
    precision=3,
    suppress=True,
    formatter={'float': '{: 8.3f}'.format}
)
np.set_printoptions(
    linewidth=100,
    precision=3,
    suppress=True,
    formatter={'float': '{: 8.3f}'.format}
)
'''
sigmoid - 1
relu - 2

SE - Mean Squared Error

'''

async def aenumerate(asequence, start=0):
    """Asynchronously enumerate an async iterator from a given start value"""
    n = start
    async for elem in asequence:
        yield n, elem
        n += 1

async def async_zip(aiter1, aiter2):
    it1 = aiter1.__aiter__()
    it2 = aiter2.__aiter__()
    while True:
        try:
            x = await it1.__anext__()
            y = await it2.__anext__()
            yield x, y
        except StopAsyncIteration:
            break

class NeuralNetwork:

    def __init__(self,layers,learningRate,activations,lossFunc,clipSize = 2.0,unitTest = False):
        self.layers = layers
        self.weights = [cp.random.standard_normal(size=(y,x)) for (x,y) in zip(self.layers[1:],self.layers[:-1])] if not unitTest else [cp.array([[0.2, 0.4, 0.1, -0.5],[-0.3, 0.1, 0.2, 0.3]]),cp.array([[0.5, -0.6],[0.1, 0.2],[-0.3, 0.4],[0.2, 0.1]])]
        self.biases = [cp.zeros(x,dtype=cp.float32) for x in self.layers[1:]] if not unitTest else [cp.array([0,0,0,0],dtype=cp.float32),cp.array([0,0],dtype=cp.float32)]
        self.learningRate = learningRate
        self.activations = activations
        self.trainingData = ValueError("No training data loaded")
        self.testData = ValueError("No test data loaded")
        self.clipSize = clipSize
        self.specialCase = (activations[-1] == 1 and lossFunc == 2) or (activations[-1] == 3 and lossFunc == 3)
        self.lossFunc = lossFunc
        self.actvationMap = {
            1:self.sigmoid,
            2:self.reLU,
            3:self.softmax
        }
        self.activationsDerivativeMap = {
            1:self.der_sigmoid,
            2:self.der_reLU,
            3:self.der_softmax
        }
        self.lossMap = {
            1:self.MSE,
            2:self.BCE,
            3:self.CCE
        }
        self.lossDerivativeMap = {
            1:self.der_MSE,
            2:self.der_BCE,
            3:self.der_CCE
        }
        self.lossDerivativeMapSpecial = {
            2:self.der_BCE_Sigmoid,
            3:self.der_CCE_Softmax
        }

    # @staticmethod
    # def sigmoid(x):
    #     return 1/(1+cp.exp(-x))
    
    @staticmethod
    def sigmoid(x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    @staticmethod
    def der_sigmoid(x):
        temp = 1 / (1 + cp.exp(-x))
        eps = 1e-8
        temp = cp.clip(temp,eps,1-eps)
        return temp * (1 - temp)

    @staticmethod
    def reLU(x):
        return cp.maximum(x,0)

    @staticmethod
    def der_reLU(x):
        return cp.where(x > 0, 1, 0)

    @staticmethod
    def leaky_reLU(x):
        return cp.where(x > 0, x, x*0.01)

    @staticmethod
    def der_leaky_reLU(x):
        return cp.where(x > 0, 1, 0.01)
		
    @staticmethod
    def softmax(x):
        exp = x - cp.max(x,axis=1,keepdims=True)
        s = cp.exp(exp)
        return s / cp.sum(s,axis=1,keepdims=True)

    @staticmethod
    def der_softmax(x):
        if x.ndim == 1:
            temp = x - cp.max(x)
            ex = cp.exp(temp)
            s = ex / cp.sum(ex)
            return cp.diag(s) - cp.outer(s,s)
        elif x.ndim == 2:
            temp = x - cp.max(x, axis = 1, keepdims=True)
            ex = cp.exp(temp)
            s = ex / cp.sum(ex, axis = 1, keepdims=True)
            s_expanded = s[:, :, None]
            s_transposed = s[:, None, :]
            outer = s_expanded * s_transposed

            diag = cp.zeros_like(outer)
            indices = cp.arange(s.shape[1])
            diag[:, indices, indices] = s
            return diag - outer
        
    #redundant
    @staticmethod
    def SE(output,expected):
        return cp.square(output-expected)
    
    #redundant
    @staticmethod
    def der_SE(output,expected):
        return 2*(output-expected)

    @staticmethod
    def MSE(output,expected):
        return cp.mean(cp.square(output - expected))
    
    @staticmethod
    def der_MSE(output,expected):
        return 2 * (output - expected) / cp.size(output)
    
    @staticmethod
    def BCE(output,expected):
        eps = 1e-8
        output = cp.clip(output,eps,1-eps)
        return -cp.mean(expected * cp.log(output) + (1 - expected)*cp.log(1 - output))

    @staticmethod
    def der_BCE(output,expected):
        eps = 1e-8
        output = cp.clip(output,eps,1-eps)
        return (expected - output) / (output * (1 - output))

    @staticmethod
    def der_BCE_Sigmoid(output,expected):
        return output - expected

    @staticmethod
    def CCE(output,expected):
        eps = 1e-8
        return -cp.sum(expected * cp.log(output + eps))

    @staticmethod
    def der_CCE(output,expected):
        eps = 1e-8
        cp.where(cp.abs(output) < eps, output + 2*eps,output)
        return -expected / output

    @staticmethod
    def der_CCE_Softmax(output,expected):
        return output - expected

    def loadTrainingData(self,fileName):
        allData = cp.loadtxt(fileName,delimiter=",")
        labels = allData[:,[0]]
        data = allData[:,1:]
        print(labels,data)

    def forward(self,data):
        print("before: ",data)
        allData = []
        for (w,b,activation) in zip(self.weights,self.biases,self.activations):
            data = self.actvationMap[activation](np.dot(data,w) + b)
            print(type(data))
            allData.append(data)
        print ("after: ",allData)
        return allData

    async def forward_gpu(self,data):
        data = cp.asanyarray(data)
        activations = [data]
        preactivations = [data]
        for (w,b,activation) in zip(self.weights,self.biases,self.activations):
            preactivated = cp.dot(data,w) + b
            preactivations.append(preactivated)
            data = self.actvationMap[activation](preactivated)
            activations.append(data)
        await asyncio.sleep(0)
        return preactivations,activations

    async def backpropagate(self,preactivated_neurons,activated_neurons,label):
        loss = self.lossMap[self.lossFunc](activated_neurons[-1],label)
        # print("loss: ",loss,activated_neurons[-1],label)
        if self.specialCase:
            derLoss = self.lossDerivativeMapSpecial[self.lossFunc](activated_neurons[-1],label)
        else:
            derLoss = self.lossDerivativeMap[self.lossFunc](activated_neurons[-1],label)
        layerIndex = -1
        # for i in range(len(activated_neurons)):
        #     print("pre:",preactivated_neurons[i],"post:",activated_neurons[i])
        # print("\nLoss:",derLoss)
        derWeights = []
        derBiases = []
        for i in range(len(preactivated_neurons) - 1,0,-1):
            if i != len(preactivated_neurons) - 1 or self.specialCase:    
                derivative = self.activationsDerivativeMap[self.activations[layerIndex]](preactivated_neurons[i])
                if derivative.ndim == derLoss.ndim:
                    derLoss *= derivative
                elif derivative.ndim == derLoss.ndim + 1:
                    derLoss = cp.einsum('bij,bj->bi', derivative, derLoss)
            shape = derLoss.shape[0]
            derWeights.append(cp.matmul(cp.transpose(activated_neurons[i - 1]),derLoss) / shape)
            derBiases.append(cp.sum(derLoss,axis=0)/ derLoss.shape[0])
            # print("derLoss: ",derLoss,"\nactivation: ",activated_neurons[i-1],"\nderWeights: ",derWeight,"\nderBiases:",derBiases)
            derLoss = cp.matmul(derLoss,cp.transpose(self.weights[layerIndex]))
            layerIndex -= 1
        await asyncio.sleep(0)
        return loss,derWeights[::-1],derBiases[::-1]
    
    # async def shuffle_arrs(self,data,lables):
    #     permute = cp.random.permutation(len(data))
    #     return data[permute],lables[permute]

    async def data_generator(self,data,labels,batchSize,stochastic):
        size = data.shape[0]
        permute = cp.random.permutation(size)
        data = data[permute]
        labels = labels[permute]
        if not stochastic:
            size = int(cp.ceil(size / batchSize))
        for i in range(0,size,batchSize):
            print (i,i+batchSize,size)

    async def train_gpu(self,data,labels,epochs,batchSize,stochastic=False):        
        data = cp.asanyarray(data)
        labels = cp.asanyarray(labels)

        costs = []

        startingCost = 0

        avgLoadTime = 0

        for epoch in range(epochs):
            t1 = timeit.default_timer()
            
            permute = cp.random.permutation(len(data))
            data = data[permute]
            labels = labels[permute]


            size = data.shape[0]
            # print(size)
            noBatches = int(cp.ceil(size / batchSize))

            if stochastic:
                batches = cp.array_split(data,size)
                label_batches = cp.array_split(labels,size)
                noBatches = size
            else:
                batches = cp.array_split(data,noBatches)
                label_batches = cp.array_split(labels,noBatches)
        

            # print("batches: ",batches[0],label_batches[0],len(batches),len(label_batches),noBatches,size)
            
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(noBatches)]

            totalCost = 0

            tasks = []

            for i,(batch,label) in enumerate(zip(batches,label_batches)):
                # print(i,batch,label)

                with streams[i % noBatches]:
                    preactivatedNeurons,activationNeurons = await self.forward_gpu(batch)
                    cost,dW,dB = await self.backpropagate(preactivatedNeurons,activationNeurons,label)
                for dw in dW:
                    # print("Grad norm:", cp.linalg.norm(dw))
                    gradNorm = cp.linalg.norm(dw)
                    if gradNorm > self.clipSize:
                        dw *= self.clipSize
                        dw /= gradNorm
                totalCost += cost
                for k in range(len(self.weights)):
                    # print("changes: ",dW[k ],dB[k])
                    self.weights[k] -= self.learningRate * dW[k]
                    self.biases[k] -= self.learningRate * dB[k]  
            if epoch == 0:
                startingCost = totalCost/noBatches
            # os.system("cls")
            # print(f"###########################################"+"".join(['#' for i in range(int(cp.ceil(cp.log10(epoch + 1)) + cp.ceil(cp.log10(epochs))))])+f"\n\t\tEPOCH {epoch + 1} / {epochs}\n###########################################"+"".join(['#' for i in range(int(cp.ceil(cp.log10(epoch + 1)) + cp.ceil(cp.log10(epochs))))]))
            # print(f"Starting cost: {startingCost}")
            # print(f"Average cost: {totalCost/noBatches}")
            costs.append(totalCost/noBatches)
            # print("Weights: ")
            # for i in self.weights:
            #     print(i)
            # print("Biases")
            # for i in self.biases:
            #     print(i)
            t2 = timeit.default_timer()
            avgLoadTime += (t2 - t1)

            print(f"Time per epoch: {t2-t1}")

        x = [i for i in range(epochs)]
        y = costs

        print(f"Average epoch time: {avgLoadTime/epochs}")

        return x,y

    def train(self,data,batchSize,epochs):
        self.train_gpu(data,)

layers = np.array([2,8,4,1])
activations = np.array([2,2,1])
lossFunc = 2

'''
self.actvationMap = {
            1:self.sigmoid,
            2:self.reLU,
            3:self.softmax
    }
self.lossMap = {
            1:self.MSE,
            2:self.BCE,
            3:self.CCE
    }
'''


data = cp.array([[1.21,3.66],[14.91,0.12]])
label = cp.array([[1,0],[0,1]])


thedata = []
labels = []

with open("train.csv",newline="\n") as theFile:
    spamreader = csv.reader(theFile)
    for row in spamreader:
        if row[0] == "0":
            labels.append(cp.array([0]))
        elif row[0] == "1":
            labels.append(cp.array([1]))
        dataa = cp.array([float(i) for i in row[1:]])
        thedata.append(dataa)


thedata = cp.array(thedata)
labels = cp.array(labels)

streamm = cp.cuda.Stream(non_blocking=True)
# network.forward_gpu(data,streamm)

network = NeuralNetwork(layers,0.05,activations,lossFunc,unitTest=False)

dataaa, labless = make_moons(n_samples=500, noise=0.1, random_state=42)
labless = labless.reshape(-1, 1)

dataaa = cp.array(dataaa.astype(cp.float32))
labless = cp.array(labless.astype(cp.float32))

print(dataaa[0],labless[0])

# (x1,y1) = asyncio.run(network.train_gpu(dataaa,labless,1,batchSize=256,stochastic=True))
# x1 = np.asanyarray(x1)
# y1 = np.asanyarray([i.get() for i in y1])
# plt.plot(x1,y1)
# plt.show()

network.loadTrainingData("train.csv")