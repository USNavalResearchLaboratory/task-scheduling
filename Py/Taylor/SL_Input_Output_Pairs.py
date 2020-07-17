##### Import Statements #####

import numpy
import pandas


##### Data (Fix "N" and Generate "MC" Sets of Tasks) #####

def import_dataset(filename):

	dataset = pandas.read_csv(filename,sep=' ')
	dataset_strings = dataset.values
	dataset_numbers = dataset_strings.astype('float32')

	del dataset
	del dataset_strings

	return dataset_numbers

inputs_before = import_dataset(r'E:Inputs_Before.txt')

print("\nInputs (Before) = \n\n{}".format(inputs_before))
print("\nShape of Inputs (Before) = {}".format(numpy.shape(inputs_before)))


##### SL Input-Output Pairs #####

number_of_sets = 10000 ########## SPECIFY
number_of_tasks = 3 ############# SPECIFY
number_of_features = 3 ########## SPECIFY

X = numpy.zeros([(number_of_features)+number_of_tasks,(number_of_sets*number_of_tasks*number_of_tasks)])
Y = numpy.zeros([1,number_of_sets*number_of_tasks])

for i in range(len(inputs_before)):

    features = numpy.zeros([number_of_features,number_of_tasks])

    release_times = inputs_before[i,0:1*number_of_tasks]
    features[0,:] = release_times

    weights = inputs_before[i,number_of_tasks:2*number_of_tasks]
    features[1,:] = weights

    lengths = inputs_before[i,2*number_of_tasks:3*number_of_tasks]
    features[2,:] = lengths

    # print("\nF = \n\n{}".format(gen_features))

    EST = inputs_before[i,3*number_of_tasks:4*number_of_tasks]

    decisions_1 = numpy.zeros([number_of_tasks,number_of_tasks])

    # print("\nD(1) = \n\n{}".format(decisions_1))

    decisions_2 = numpy.zeros([number_of_tasks,number_of_tasks])
    decisions_2[0,EST[0].astype('int')] = 1

    # print("\nD(2) = \n\n{}".format(decisions_2))

    decisions_3 = numpy.zeros([number_of_tasks,number_of_tasks])
    decisions_3[0,EST[0].astype('int')] = 1
    decisions_3[1,EST[1].astype('int')] = 1
    
    # print("\nD(3) = \n\n{}".format(decisions_3))

    X_1 = numpy.concatenate((features,decisions_1),axis=0)
    X_2 = numpy.concatenate((features,decisions_2),axis=0)
    X_3 = numpy.concatenate((features,decisions_3),axis=0)

    X_incomplete = numpy.concatenate((X_1,X_2),axis=1)
    X_complete = numpy.concatenate((X_incomplete,X_3),axis=1)
    
    # print("\nX = \n\n{}".format(X_complete))

    X[:,(number_of_tasks*number_of_tasks*i):(number_of_tasks*number_of_tasks*(i+1))] = X_complete
    Y[:,(number_of_tasks*i):(number_of_tasks*(i+1))] = EST

print("\nX = \n\n{}".format(X))
print("\nShape of X = {}".format(numpy.shape(X)))
print("\nY = \n\n{}".format(Y))
print("\nShape of Y = {}\n".format(numpy.shape(Y)))


##### Export Statements #####

disposable_row_inputs = numpy.zeros([1,(number_of_sets*number_of_tasks*number_of_tasks)])
disposable_row_outputs = numpy.zeros([1,(number_of_sets*number_of_tasks)])

inputs_after = numpy.concatenate((disposable_row_inputs,X),axis=0)
outputs_after = numpy.concatenate((disposable_row_outputs,Y),axis=0)

numpy.savetxt('SL_Inputs.txt',inputs_after,fmt='%.3e')
numpy.savetxt('SL_Outputs.txt',outputs_after,fmt='%.3e')


##### Documentation #####

# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
# https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
# https://numpy.org/doc/1.18/reference/generated/numpy.concatenate.html
# https://numpy.org/devdocs/reference/generated/numpy.savetxt.html

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

