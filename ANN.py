import numpy 
import scipy.special
import matplotlib.pyplot as plt
import itertools


# Neural network class definition
class NeuralNetwork:
# Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

        # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs




########################################################################
########################################################################
########################################################################
input_nodes = 9 # was 9
hidden_nodes = 40 #22 #40 seemed good
output_nodes =  1#was 1
learning_rate = 0.5 #was 0.5 #0.3 seemed good

N = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)



####
#Opens up the spreadsheet with the MGAM data
#data = open("mgam.csv", 'r')

data = open("mgam.csv", 'r')

date = []
opennum = []
high = []
low = []
close = []
volume = []


sdate = []
sopen = []
shigh = []
slow = []
sclose = []
svolume = []

#limits the data to only 250 long
makeittwohundred = 250 #was 200
for x in range (0,makeittwohundred):
    mydata = data.readline()
    test = mydata.split(',')
    sdate.append(test[0])
    sopen.append(test[1])
    shigh.append(test[2])
    slow.append(test[3])
    sclose.append(test[4])
    svolume.append(test[5])

#shuffles the data , as the top value is the column header
date=(sdate[1:])
opennum=(sopen[1:])
high=(shigh[1:])
low=(slow[1:])
close=(sclose[1:])
volume=(svolume[1:])

#print(date)
#print(open)
#print(high)
#print(low)
print(close)
#print(volume)



#show current stock trend just to check sanity
xscale=(range(1,99))
yscaletest= (close[1:99])
plt.figure()
plt.gca().invert_xaxis()
plt.plot(xscale, yscaletest , color="red" , label= "actual")
plt.show()



   #sweeps throught the data, inputting it in blocks of 9 
   #at a time and using the 10th bit as test data. Range reversed 
   #in order to start in the past rather than the present
I = 220 #was 160
for x in reversed(range(1,I)):
    inputs = (numpy.asfarray(close[(x+1):(x+10)]) / 500 * 0.99) + 0.01
    targets = (numpy.asfarray(close[x]) /500 *0.99) + 0.01
    N.train(inputs, targets)


newclose=[float(i) for i in close]

errors=[]
outs =[]

#inputs the most recent 30 days and stores the outputs to then plot,
#error is calculated and stored
for x in reversed(range(1,30)):
    inputs = (numpy.asfarray(close[x+1:x+10]) / 500 * 0.99) + 0.01
    outputs = N.query(inputs)
    outputs = ((outputs - 0.01) /0.99) *500
    print(outputs)
    merged_list = list(itertools.chain(*outputs))
    outs.append(merged_list[0])
    print(newclose[x])
    print(merged_list[0])
    error =  newclose[x] - merged_list[0]
    errors.append(error)
outs.reverse()
  

#Code below is all "stand alone" and such ive put in a block for the moment
confidencelevel = []
for x in range(1,8):
    confidence = ((errors[x]/newclose[x])*100 )
    confidencelevel.append(confidence)
a=(len(confidencelevel))
b=(sum(confidencelevel))
n = []
for x in range(0,7):
    nprev = abs(confidencelevel[x])
    n.append(nprev)
m = (max(n))


#confidence level is a function of the average error 
#accross a week, a value beneath 98% is 
#bad(unless the stock really is volatile)
print(confidencelevel)
inputs = (numpy.asfarray(close[1:10]) / 500 * 0.99) + 0.01
outputs = N.query(inputs)
outputs = ((outputs - 0.01) /0.99) *500
print('tomorrows stock will be :')
print(outputs)
print('Week long confidence level = ')
absoluteconfidence = abs(b/a)
print((100-absoluteconfidence), '%')


#risk level is the maximum percentage error in one week, the higher it is the worse
print('Week long risk level = ')
print(m , '%')

#plotting two weeks of predictions versus actual data
xscale=(range(0,14))
yscaletest= (newclose[0:14])
print(newclose[0:14])
plt.figure()
plt.gca().invert_xaxis()
plt.plot(xscale, yscaletest , color="red" , label= "actual")
yscaletest= (outs[0:15])
print(outs[0:15])
xscale=(range(-1,14))
plt.plot(xscale, yscaletest , color ="green")
plt.show()