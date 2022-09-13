function train_anfis(members, input_data, test_data, data_size, epochs)
	
	%create the cross-validation split of the input data
	cv = cvpartition(input_data(:,7),"holdout",0.2,"Stratify",false)
	
	%create the training and validation indexes
	idxTrain = training(cv)
	idxValidation = test(cv)

	%create the training and validation tables
	stellar_train = input_data(idxTrain,:)
	stellar_validation = input_data(idxValidation,:)

	%set the anfis options / hyper-parameters
	opt = anfisOptions("InitialFIS",members,"EpochNumber",epochs,"StepSizeIncreaseRate",1.2,"ValidationData",stellar_validation)
	
	%training the adaptive neuro-fuzzyinference system
	[fis, trainError, stepSize, chkFIS,checkError] = anfis(stellar_train,opt)

	%export training details by epoch to csv
	training_details = [trainError, stepSize, checkError]
	training_details_file_name = strcat("anfis_training_details_",num2str(members),"_members_",num2str(data_size),".csv")
	csvwrite(training_details_file_name, training_details)

	%predict against the training data and export to .csv
	training_results = round(evalfis(fis,input_data(:,1:6)))
	training_file_name = strcat("anfis_training_",num2str(members),"_members_",num2str(data_size),".csv")
	csvwrite(training_file_name,training_results)

	%predict against the test data and export to .csv
	test_results = round(evalfis(fis,test_data(:,1:6)))
	testing_file_name = strcat("anfis_testing_",num2str(members),"_members_",num2str(data_size),".csv")
	csvwrite(testing_file_name,test_results)

	%clear the fis in memory to prevent new epochs being appended
	clear fis
end