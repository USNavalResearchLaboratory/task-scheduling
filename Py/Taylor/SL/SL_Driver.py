import os
os.system('cls')


##### Formatting #####

print("\n")


##### Import Statements #####

from OneDim_CNN import OneDim_CNN_function


##### User-Specified Inputs (Environment) #####

number_of_sets = 10000

number_of_tasks = 3

number_of_features = 3

maximum_weight_of_tasks = 10


##### User-Specified Inputs (One-Dimensional Convolutional Neural Network) #####

number_of_filters = 18

dropout = 0.15

number_of_neurons = 18

epochs = 5

batch_size = 10

split_percentage = 0.1


##### One-Dimensional Convolutional Neural Network #####

OneDim_CNN_function(number_of_sets,number_of_tasks,number_of_features,maximum_weight_of_tasks,number_of_filters,dropout,number_of_neurons,epochs,batch_size,split_percentage)


##### Formatting #####

print("\n")

