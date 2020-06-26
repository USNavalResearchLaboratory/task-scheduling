##### Import Statements #####

import keras
import numpy
import pandas


##### Data (SL Input-Output Pairs) #####

def import_data(filename):

	data = pandas.read_csv(filename,sep=' ')
	data_strings = data.values
	data_numbers = data_strings.astype('float32')

	del data
	del data_strings

	return data_numbers

SL_Inputs_before = import_data(r'E:SL_Inputs.txt')
SL_Outputs_before = import_data(r'E:SL_Outputs.txt')

print("\nSL Inputs (Before) = \n\n{}".format(SL_Inputs_before))
print("\nShape of SL Inputs (Before) = {}".format(numpy.shape(SL_Inputs_before)))
print("\nSL Outputs (Before) = \n\n{}".format(SL_Outputs_before))
print("\nShape of SL Outputs (Before) = {}".format(numpy.shape(SL_Outputs_before)))


##### Data (Formatted SL Input-Output Pairs) #####

number_of_sets = 10000 ########################### SPECIFY
number_of_tasks = 3 ############################## SPECIFY
number_of_features = 3 ########################### SPECIFY

SL_Inputs_after = numpy.zeros([number_of_sets*number_of_tasks,(number_of_features*number_of_tasks+number_of_tasks*number_of_tasks)])

for j in range(number_of_sets):
	
	for i in range(number_of_tasks):

		matrix = SL_Inputs_before[0:(number_of_features+number_of_tasks),(number_of_tasks*number_of_tasks)*j+(number_of_tasks)*i:(number_of_tasks*number_of_tasks)*j+(number_of_tasks)*(i+1)]
		vector = numpy.ndarray.flatten(matrix)
	
		SL_Inputs_after[number_of_tasks*j+i,:] = vector

print("\nSL Inputs (After) = \n\n{}".format(SL_Inputs_after))
print("\nShape of SL Inputs (After) = {}".format(numpy.shape(SL_Inputs_after)))

SL_Inputs_reshaped = numpy.reshape(SL_Inputs_after,(SL_Inputs_after.shape[0],1,SL_Inputs_after.shape[1]))
print("\nReshaped SL Inputs (After) = {}".format(numpy.shape(SL_Inputs_reshaped)))

SL_Outputs_after = keras.utils.to_categorical(numpy.transpose(SL_Outputs_before))

print("\nSL Outputs (After) = \n\n{}".format(SL_Outputs_after))
print("\nShape of SL Outputs (After) = {}".format(numpy.shape(SL_Outputs_after)))


##### Training and Testing Split #####

split_percentage = 0.1 ############### SPECIFY

split_index = int(len(SL_Inputs_reshaped)*(1-split_percentage))

SL_Inputs_training = SL_Inputs_reshaped[0:split_index,:]
SL_Inputs_testing = SL_Inputs_reshaped[split_index:len(SL_Inputs_reshaped),:]

print("\nShape of SL Inputs (Training) = {}".format(numpy.shape(SL_Inputs_training)))
print("\nShape of SL Inputs (Testing) = {}".format(numpy.shape(SL_Inputs_testing)))

SL_Outputs_training = SL_Outputs_after[0:split_index,:]
SL_Outputs_testing = SL_Outputs_after[split_index:len(SL_Outputs_after),:]

print("\nShape of SL Outputs (Training) = {}".format(numpy.shape(SL_Outputs_training)))
print("\nShape of SL Outputs (Testing) = {}".format(numpy.shape(SL_Outputs_testing)))


##### Convolutional Neural Network #####

print("\n\n##### KERAS AND TENSORFLOW (START) #####\n")

CNN = keras.Sequential()
CNN.add(keras.layers.Conv1D(filters=18,kernel_size=1,activation='relu',input_shape=(1,18)))
CNN.add(keras.layers.Conv1D(filters=18,kernel_size=1,activation='relu'))
CNN.add(keras.layers.Dropout(0.15))
CNN.add(keras.layers.MaxPooling1D(pool_size=1))
CNN.add(keras.layers.Flatten())
CNN.add(keras.layers.Dense(18,activation='relu'))
CNN.add(keras.layers.Dense(3,activation='softmax'))
CNN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

CNN.fit(SL_Inputs_training,SL_Outputs_training,epochs=5,batch_size=10,verbose=1)

_, accuracy = CNN.evaluate(SL_Inputs_testing,SL_Outputs_testing,batch_size=10,verbose=1)
print("\nAccuracy (keras.evaluate) = {:.0%}".format(accuracy))

print("\n##### KERAS AND TENSORFLOW (END) #####\n")


##### Results of Convolutional Neural Network #####

CNN_actual = SL_Outputs_before[0,split_index:]
print("\nCNN Actual = \n\n{}".format(CNN_actual))
print("\nShape of CNN Actual = {}".format(numpy.shape(CNN_actual)))

CNN_prediction = CNN.predict_classes(SL_Inputs_testing)
print("\nCNN Prediction = \n\n{}".format(CNN_prediction))
print("\nShape of CNN Prediction = {}\n".format(numpy.shape(CNN_prediction)))

CNN_results_incomplete = numpy.transpose([CNN_actual,CNN_prediction])

disposable_row = numpy.zeros([1,2])
CNN_results_complete = numpy.concatenate((disposable_row,CNN_results_incomplete),axis=0)

numpy.savetxt('CNN_Results.txt',CNN_results_complete,fmt='%i')


##### Documentation #####

# https://keras.io/api/utils/python_utils/#to_categorical-function
# https://keras.io/api/models/sequential/
# https://keras.io/api/layers/convolution_layers/convolution1d/

# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
# https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
# https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
# https://numpy.org/doc/1.18/reference/generated/numpy.concatenate.html
# https://numpy.org/devdocs/reference/generated/numpy.savetxt.html

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

