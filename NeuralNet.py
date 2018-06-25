import numpy as np

def sigmoid(x):
    return (1.0/(1+ np.exp(-x)))

def sigmoid_derivative(x):
    return (x * (1.0 - x))

class Neuron:
    eta = 0.3
    alpha = 0.5
    
    def __init__(self, neuronNum, numOutputs):
        global neuronCount
        self.neuronNum = neuronNum
        self.numOutputs = numOutputs
        
        self.weight = np.random.rand(1,numOutputs)
        self.deltaWeight = np.random.rand(1,numOutputs)
        self.output = 0.0
        self.gradient = 0.0
        self.dow = 0.0
        
        
    def setOutputVal(self, inputVal):
        self.output = inputVal
        print("Input from ", self.neuronNum, "th neuron - ", self.output)

    def feedForward(self, layerNum, prevLayer):
        global loopCount
        sum = 0.0

        for i in (range(prevLayer.numNeurons)):
            sum = sum+ prevLayer.neurons[i].weight[0,self.neuronNum] * prevLayer.neurons[i].output

        self.output = sigmoid(sum)
        #print("Layer ",layerNum, "neuron ", self.neuronNum, "output ", self.output)

    def calcOutputGradients(self, target):
        delta = 0.0
        delta = target - self.output
        self.gradient = delta * sigmoid_derivative(self.output)
        #print("Neuron ", self.neuronNum, "gradient ", self.gradient)

    def calcHiddenGradients(self, nextLayerNeurons):
        sum = 0.0 

        for output in range(self.numOutputs):
            sum = sum + self.weight[0,output] * nextLayerNeurons[output].gradient

        self.dow = sum


        self.gradient = sum * sigmoid_derivative(self.output)
        #print(self.gradient)

    def updateWeights(self,nextLayerNeurons):
        #print("Update weights")
        for neuronNum in range(self.numOutputs):
            #print(neuronNum)
            oldDeltaWeight = self.deltaWeight[0,neuronNum]
            newDeltaWeight = self.eta * self.output * nextLayerNeurons[neuronNum].gradient + self.alpha * oldDeltaWeight

            #print("Index ", neuronNum, "Old weight ", self.weight[0,neuronNum])

            self.weight[0,neuronNum] += newDeltaWeight
            self.deltaWeight[0,neuronNum] = oldDeltaWeight

            #print("Index ", neuronNum, "New weight ", self.weight[0,neuronNum])
            
            #print(output)
        
        #print(self.numOutputs)
       
        

class Layer:
    def __init__(self, myLayerNum, numNeurons, topology):
        self.layerNum = myLayerNum
        self.numNeurons = numNeurons
        self.neurons = []

        for neuronNum in range(self.numNeurons):
            if self.layerNum == (len(topology) - 1):
                numOutputs = 0
            else:
                numOutputs = topology[myLayerNum+1]
            self.neurons.append(Neuron(neuronNum, numOutputs))
            #print("no of output - ", numOutputs)
            
        
   
class Network:
    def __init__(self, topology):
        global totalCount
        self.numLayers = len(topology)
        self.layers = []
        self.error = 0.0
        self.input = []
        self.weights = []
        self.activations = []
        self.deltas = []
        self.eta = 0.2
        print("Topology - ",topology)
        print(" ")

        # Create a list of weights for the network and initialize them
        for layerNum in range(self.numLayers-1):
            weight = np.random.random((topology[layerNum]+1,topology[layerNum+1]))
            self.weights.append(weight)

        #print(self.weights)
        we = np.array([[0.5, 0.5, 0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]])
        self.weights[0] = we
        we = np.array([[0.5],[0.123],[0.475],[0.5]])
        self.weights[1] = we
        #print("Weights - ",self.weights)
        #print(" ")

            # Add bias to all the layers
            #bias = np.atleast_2d(np.ones(1))
            #self.activations.append(bias)


    def printNetworkDetails(self):
        print("Total layers - ",self.numLayers)
        for layerNum in range(self.numLayers):
            print("No of neurons in ",layerNum, "th layer - ",self.layers[layerNum].numNeurons)
        

    def train(self, input, output, iterations=1000000):

        for loopCount in range(iterations):
            i = loopCount % input.shape[0]

            #print("Input - ",input[i])
            #print(" ")
            
            b = np.ones((1,1))
            zer = np.zeros((1,1))
            self.activations = [(np.concatenate((b,input[i].reshape((np.size(input[i],0),1)))))]

            for layerNum in range(self.numLayers-1):
                
                z = np.dot(self.activations[layerNum].T,self.weights[layerNum])
                activation = sigmoid(z.T)
                #print(activation.shape)
                self.activations.append(np.concatenate((zer,activation)))
                #print(self.activations[-1].shape)
                
            self.activations[-1] = (np.delete(self.activations[-1],0)).reshape((1,output[i].size))
            #print((output[i].size))
            #print(self.activations[-1].shape)
            #self.activations[-1].reshape((1,output[i].size))
            #print("Activations - ",self.activations)
            #print(" ")
            

            error = output[i] - self.activations[-1]
            #print("Error - ",error)
            #print(" ")
            self.deltas = [(error * sigmoid_derivative(self.activations[-1])).reshape((1,output[i].size))]
            #print(self.deltas)
            #print(" ")
            #self.deltas[-1].reshape((1,output[i].size))
            
            
            lastLayer = False
            
            for layerNum in range(self.numLayers-2,0,-1):
                #print(layerNum)
                #print((self.deltas[-1]))
                #print(self.weights[layerNum].T.shape)
                delta = self.deltas[-1]
                #print(delta)
                if(lastLayer):
                    delta = np.delete(delta,0)
                    
                #print(delta)
                delta = delta.dot(self.weights[layerNum].T)
                delta = delta * (sigmoid_derivative(self.activations[layerNum])).T
                #print(delta.shape)
                self.deltas.append(delta)
                lastLayer = True
                #print(self.deltas)
            #print(self.deltas)

            self.deltas.reverse()
            #print("deltas - ",self.deltas)
            #print(" ")

            for layerNum in range(len(self.weights)):
                #print(layerNum)
                layer = np.atleast_2d(self.activations[layerNum])
                #print(layer.shape)
                delta = np.atleast_2d(self.deltas[layerNum])
                #print(delta)
                if(layerNum != (len(self.weights)-1)):
                    delta =  np.atleast_2d(np.delete(delta,0))
                #print(delta.shape)
                #bp = layer.dot(delta)
                #print(delta.shape)

                bp = layer.dot(delta)
                #print("bp - ",bp)

                self.weights[layerNum] += self.eta * bp

            #print("Updated weights - ", self.weights)

            if loopCount % 100000 == 0:
                print('epochs:', loopCount)

    def test(self, input):

        activation = (input)
        #print(activation.shape)

        for layerNum in range(self.numLayers-1):
            #print(layerNum)
            

            activation = np.concatenate((np.ones(1),np.array(activation)))
            #activation = np.atleast_2d(activation)
            print("Activation - ",activation)
            activation = activation.dot(self.weights[layerNum])
            activation = sigmoid(activation)
            #activation = np.atleast_2d(activation)
            
            

        
        #activation = np.delete(activation,0)
        print("Activation - ",activation)

        #a = np.concatenate((np.ones(1).T, np.array(x)))

                

                
        
    def retPrevLayer(self, layerNum):
        prevLayer = self.layers[layerNum - 1]
        return prevLayer

    def printWeights(self):
        for layerNum in range(self.numLayers):
            for neuronNum in range(self.layers[layerNum].numNeurons):
                print("Layer ", layerNum, "Neuron ", neuronNum, "weight ", self.layers[layerNum].neurons[neuronNum].weight)

    def calcError(self, target):
        err = 0.0
        delta = 0.0
        lastLayer = self.layers[-1]
        actualOutput = []
        for neuronNum in range(lastLayer.numNeurons):
            delta = target[neuronNum] - lastLayer.neurons[neuronNum].output
            actualOutput.append(lastLayer.neurons[neuronNum].output)
            err = err + delta * delta

        err = err**(.5)
        print("output - ",actualOutput)
        print("Error - ", err)

        recentAvgErr = 0.0
        smoothingFactor = .70     
        recentAvgErr = (recentAvgErr * smoothingFactor + err) / (smoothingFactor + 1.0)
        
    def backProp(self, target):
        global loopCount
        lastLayer = self.layers[-1]
        for neuronNum in range(lastLayer.numNeurons):
            lastLayer.neurons[neuronNum].calcOutputGradients(target[neuronNum])

        
        for layerNum in range(self.numLayers-2,-1,-1):
            #print("Layer num ", layerNum)
            currentLayerNeurons = self.layers[layerNum].neurons
            nextLayerNeurons = self.layers[layerNum+1].neurons
            
            for neuronNum in range(self.layers[layerNum].numNeurons):
                currentLayerNeurons[neuronNum].calcHiddenGradients(nextLayerNeurons)

        loopCount = 0

        for layerNum in range(self.numLayers-2,-1,-1):
            for neuronNum in range(self.layers[layerNum].numNeurons):
                #print("Layer num - ", layerNum)
                #print("Neuron num - ", neuronNum)
                nextLayerNeurons = self.layers[layerNum+1].neurons
                self.layers[layerNum].neurons[neuronNum].updateWeights(nextLayerNeurons)
                
                
            #self.layers[layerNum].calcHiddenGradients()

    def printGradientsWeights(self):

        for layerNum in range(self.numLayers):
            
            for neuronNum in range(self.layers[layerNum].numNeurons):
                print("Layer ", layerNum)
                print("Neuron ", neuronNum)
                print("Weights ",self.layers[layerNum].neurons[neuronNum].weight)
                print("Gradients ",self.layers[layerNum].neurons[neuronNum].gradient)
                print("Output ",self.layers[layerNum].neurons[neuronNum].output)
                print("DOW ",self.layers[layerNum].neurons[neuronNum].dow)

                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                
                
            
        
def main():
    global totalCount
    global loopCount
    topology = []
    topology.append(2)
    topology.append(3)
    topology.append(1)
    

    net = Network(topology)
    
    #net.printNetworkDetails()

    input = np.array([[0,0],[0,1],[1,0],[1,1]])
    target = np.array([[0],[1],[1],[0]])

    net.train(input, target)
        #net.calcError(target[i])
        #net.backProp(target[i])

    for t in input:
        a = net.test(t)
        

    
if __name__ == '__main__':
    main()
