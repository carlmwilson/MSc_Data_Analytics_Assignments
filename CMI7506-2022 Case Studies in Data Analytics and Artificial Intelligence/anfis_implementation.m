%load training datasets

load stellar_training_78.csv
load stellar_training_156.csv
load stellar_training_781.csv
load stellar_training_1561.csv
load stellar_training_7805.csv
load stellar_training_15610.csv
load stellar_training_78052.csv

%load testing dataset
load stellar_testing_n.csv

%complete experiments
train_anfis(2, stellar_training_78, stellar_testing_n, 78, 30)
train_anfis(2, stellar_training_156, stellar_testing_n, 156, 35)
train_anfis(2, stellar_training_781, stellar_testing_n, 781, 31)
train_anfis(2, stellar_training_1561, stellar_testing_n, 1561, 39)
train_anfis(2, stellar_training_7805, stellar_testing_n, 7805, 19)
train_anfis(2, stellar_training_15610, stellar_testing_n, 15610, 21)
train_anfis(2, stellar_training_78052, stellar_testing_n, 78052, 24)

train_anfis(3, stellar_training_78, stellar_testing_n, 78, 29)
train_anfis(3, stellar_training_156, stellar_testing_n, 156, 13)
train_anfis(3, stellar_training_781, stellar_testing_n, 781, 15)
train_anfis(3, stellar_training_1561, stellar_testing_n, 1561, 17)
train_anfis(3, stellar_training_7805, stellar_testing_n, 7805, 21)






