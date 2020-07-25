def OneDim_CNN_function(number_of_sets,number_of_tasks,number_of_features,maximum_weight_of_tasks,number_of_filters,dropout,number_of_neurons,epochs,batch_size,split_percentage):

	##### Import Statements #####

	import keras
	import numpy
	import pandas
	import warnings

	from input_output_pairs import input_output_pairs_function


	##### Data (SL Input-Output Pairs) #####

	SL_Inputs_before, SL_Outputs_before = input_output_pairs_function(number_of_sets,number_of_tasks,number_of_features,maximum_weight_of_tasks)
	print("\n\nExecuting 'OneDim_CNN.py' ...")

	# print("\nSL Inputs (Before) = \n\n{}".format(SL_Inputs_before))
	print("\n\tShape of SL Inputs (Before) = {}".format(numpy.shape(SL_Inputs_before)))
	# print("\nSL Outputs (Before) = \n\n{}".format(SL_Outputs_before))
	print("\n\tShape of SL Outputs (Before) = {}".format(numpy.shape(SL_Outputs_before)))


	##### Data (Formatted SL Input-Output Pairs) #####

	SL_Inputs_after = numpy.zeros([number_of_sets*number_of_tasks,(number_of_features*number_of_tasks+number_of_tasks*number_of_tasks)])

	for j in range(number_of_sets):

		for i in range(number_of_tasks):

			matrix = SL_Inputs_before[0:(number_of_features+number_of_tasks),(number_of_tasks*number_of_tasks)*j+(number_of_tasks)*i:(number_of_tasks*number_of_tasks)*j+(number_of_tasks)*(i+1)]
			vector = numpy.ndarray.flatten(matrix)

			SL_Inputs_after[number_of_tasks*j+i,:] = vector

	# print("\nSL Inputs (After) = \n\n{}".format(SL_Inputs_after))
	print("\n\tShape of SL Inputs (After) = {}".format(numpy.shape(SL_Inputs_after)))

	SL_Inputs_reshaped = numpy.reshape(SL_Inputs_after,(SL_Inputs_after.shape[0],1,SL_Inputs_after.shape[1]))
	# print("\nShape of Reshaped SL Inputs (After) = {}".format(numpy.shape(SL_Inputs_reshaped)))

	SL_Outputs_after = keras.utils.to_categorical(numpy.transpose(SL_Outputs_before))

	# print("\nSL Outputs (After) = \n\n{}".format(SL_Outputs_after))
	print("\n\tShape of SL Outputs (After) = {}".format(numpy.shape(SL_Outputs_after)))


	##### Training and Testing Split #####

	split_index = int(len(SL_Inputs_reshaped)*(1-split_percentage))

	SL_Inputs_training = SL_Inputs_reshaped[0:split_index,:]
	SL_Inputs_testing = SL_Inputs_reshaped[split_index:len(SL_Inputs_reshaped),:]

	print("\n\tShape of SL Inputs (Training) = {}".format(numpy.shape(SL_Inputs_training)))
	print("\n\tShape of SL Inputs (Testing) = {}".format(numpy.shape(SL_Inputs_testing)))

	SL_Outputs_training = SL_Outputs_after[0:split_index,:]
	SL_Outputs_testing = SL_Outputs_after[split_index:len(SL_Outputs_after),:]

	print("\n\tShape of SL Outputs (Training) = {}".format(numpy.shape(SL_Outputs_training)))
	print("\n\tShape of SL Outputs (Testing) = {}".format(numpy.shape(SL_Outputs_testing)))


	##### One-Dimensional Convolutional Neural Network #####

	features = number_of_features*number_of_tasks+number_of_tasks*number_of_tasks

	print("\n\n##### KERAS WITH TENSORFLOW BACKEND (START) #####\n")

	CNN = keras.Sequential()
	CNN.add(keras.layers.Conv1D(filters=number_of_filters,kernel_size=1,activation='relu',input_shape=(1,features)))
	CNN.add(keras.layers.Conv1D(filters=number_of_filters,kernel_size=1,activation='relu'))
	CNN.add(keras.layers.Dropout(dropout))
	CNN.add(keras.layers.MaxPooling1D(pool_size=1))
	CNN.add(keras.layers.Flatten())
	CNN.add(keras.layers.Dense(number_of_neurons,activation='relu'))
	CNN.add(keras.layers.Dense(number_of_tasks,activation='softmax'))
	CNN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	CNN.fit(SL_Inputs_training,SL_Outputs_training,epochs=epochs,batch_size=batch_size,verbose=1)

	_, accuracy = CNN.evaluate(SL_Inputs_testing,SL_Outputs_testing,batch_size=batch_size,verbose=2)

	print("\n##### KERAS WITH TENSORFLOW BACKEND (END) #####\n")


	##### Results of Convolutional Neural Network #####

	print("\n\tCNN Accuracy (Keras) = {:.0%}".format(accuracy))

	CNN_ideal = SL_Outputs_before[0,split_index:]
	#print("\nCNN Results (Ideal) = \n\n{}".format(CNN_ideal))
	print("\n\tShape of CNN Results (Ideal) = {}".format(numpy.shape(CNN_ideal)))

	CNN_prediction = CNN.predict_classes(SL_Inputs_testing)
	# print("\nCNN Results (Prediction) = \n\n{}".format(CNN_prediction))
	print("\n\tShape of CNN Results (Prediction) = {}".format(numpy.shape(CNN_prediction)))

	CNN_results = numpy.transpose([CNN_ideal,CNN_prediction])

	# numpy.savetxt('CNN_Results.txt',CNN_results,fmt='%i')


	##### Analysis of Results of Convolutional Neural Network #####

	check = numpy.zeros([len(CNN_results),1])
	counter = 0

	for i in range(len(CNN_results)):

		ideal = CNN_results[i,0]
		prediction = CNN_results[i,1]

		if ideal == prediction:
			check[i,0] = 1
			counter += 1
		else:
			check[i,0] = 0

	# print("\nCheck = \n\n{}".format(check))
	print("\n\tShape of Check = {}".format(numpy.shape(check)))

	print("\n\tCount = {}".format(counter))

	accuracy_check = counter/len(check)

	print("\n\tCNN Accuracy (Check) = {:.0%}".format(accuracy_check))


	##### Documentation #####

	# https://keras.io/api/utils/python_utils/#to_categorical-function
	# https://keras.io/api/models/sequential/
	# https://keras.io/api/layers/convolution_layers/convolution1d/

	# https://numpy.org/doc/stable/reference/generated/numpy.shape.html
    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
	# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
	# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
	# https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    # https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html

	# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

	# Example => https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/