#
# A simple simulated annealing example to deonstrate a regression function
#
import numpy as np
import math
import random
import matplotlib.pyplot as plt

#imports the spreadsheet containing the stock data for MGAM
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


#limits the data to a list 250 long
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
#removes the top row , this contains the headers and so are not the format required
date=(sdate[1:])
opennum=(sopen[1:])
high=(shigh[1:])
close=(sclose[1:])
volume=(svolume[1:])
np.flipud(close)


#
# Calculate the acceptance probability
#

def acceptance_probability(old_cost, new_cost, T):
   if (old_cost - new_cost) >0:
       a = 1 
   else:
       a = math.exp((old_cost-new_cost)/T)
   return a

#
# Calculates a 5 order equation/trendline
#
def calcLine(parameters,m):
   parameters = np.array(parameters) 
   line = np.full(m,0)
   for i in range(0,m):
      line[i] =  parameters[5]*(float(i)**5) + parameters[4]*(float(i)**4) + parameters[3]*(float(i)**3) + parameters[2]*(float(i)**2)  + parameters[1]*(float(i)) + parameters[0]
   return line

#
#
# Simulated Anealling Core
#
#
def anneal(parameters,target,alpha,iterations,var,n,m):

    # This is the import change, now we have to calculate a solution based on parameters each time
    solution = calcLine(parameters,m)
    old_cost = cost(solution, target)
    cost_values = list()
    cost_values.append(old_cost)
    T = 1.0
    T_min = 0.0001
    while T > T_min:
        i = 1
        while i <= iterations:
            #print("Iteration : " + str(i) + "Cost : " + str(old_cost))
            #print("Iteration : " + str(i) + "Target : " + str(target) + "Solution: " + str(solution))
        #    print("Iteration : " + str(i))
          
            print("Temp : " + str(T))
        # Again, the change is that we need to establish the neighbour parameters, not the solution
            new_parameters = neighbour(parameters,n,var)
            new_solution = calcLine(new_parameters,m)
            new_cost = cost(new_solution,target)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random.random():
                parameters = new_parameters
                old_cost = new_cost
            i += 1
            cost_values.append(old_cost)
        T = T*alpha
    # Note we return the parameters NOT the solution
    return parameters, old_cost, cost_values

#
# Still using RMS error
#
def cost(solution,target):
   delta = np.subtract(target,solution)
   delta2 = np.square(delta)
   ave = np.average(delta2)
   dcost = math.sqrt(ave)
   return dcost

# 
# Note the neighbour is a linear rray not multiple dimensions
#
def neighbour(parameters,n,d):
   delta = np.random.random(n)
   scale = np.full(n,2*d)
   offset = np.full(n,1.0-d)
   var = np.multiply(delta,scale)
   m = np.add(var,offset)
   new_solution = np.multiply(parameters,m)
   return new_solution



# Number of parameters (m and c in y=mx+c)   
n = 6
# Number of data points

#is about 90
#number of numbers the code attempts to find the trendline for
m = 90


initial_parameters = [312, 3.0779 , -0.228, 0.0061 , -0.00007 , 0.0000003   ]
#The intial weightings/coeffiecents on the trendline, in reverse order
#initial_parameters = [320, 3 , -0.2, 0.1 , -0.1 , 0.3   ]
# Calculate the line based on the default parameters
initial = calcLine(initial_parameters,m)

# The target values are a random array

#make target 
#close[0] is reserved as it is the "future"
target = np.array((close[1: 91]), dtype='float16')

#plt.plot(calcLine(initial_parameters,m))
#plt.show()

# Simulated Annealing Parameters
alpha=0.8
iterations = 2000
var = 0.00001

# Run the Optimization
final_parameters, cost, cost_values = anneal(initial_parameters,target,alpha,iterations,var,n,m)

# Obtain the final line
final_solution = calcLine(final_parameters,m)

# print the values
#print(target)
#print(final_parameters)
#print(cost)

# Plot the values
plt.subplot(131)
plt.title("Error Function in Simulated Annealing")
plt.plot(cost_values)
plt.grid(True)

plt.subplot(132)
plt.title("Initial")
plt.plot(target,marker='.',linestyle='None')
plt.plot(initial)

plt.subplot(133)
plt.title("Final")
plt.plot(target,marker='.',linestyle='None')
plt.plot(final_solution)

plt.show()
print("Tomorrows stock will be:")
print(final_parameters[0])
