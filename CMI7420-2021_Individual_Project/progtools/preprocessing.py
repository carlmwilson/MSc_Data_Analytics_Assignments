import h5py
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import pad_sequences

import shap

class RawFlightData():
    """Class for raw flight data to be read in from hp5y file and queried as dataframes
    """
    def __init__(self, filename: str):

        with h5py.File(filename, 'r') as hdf:
        # Development set
            self.W_dev = np.array(hdf.get('W_dev'))             # W
            self.X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s
            self.Y_dev = np.array(hdf.get('Y_dev'))             # RUL  
            self.A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

            # Test set
            self.W_test = np.array(hdf.get('W_test'))           # W
            self.X_s_test = np.array(hdf.get('X_s_test'))       # X_s
            self.Y_test = np.array(hdf.get('Y_test'))           # RUL  
            self.A_test = np.array(hdf.get('A_test'))           # Auxiliary

            # Varnams
            self.W_var = np.array(hdf.get('W_var'))
            self.X_s_var = np.array(hdf.get('X_s_var'))
            self.A_var = np.array(hdf.get('A_var'))

            # from np.array to list dtype U4/U5
            self.W_var = list(np.array(self.W_var, dtype='U20'))
            self.X_s_var = list(np.array(self.X_s_var, dtype='U20'))
            self.A_var = list(np.array(self.A_var, dtype='U20'))

        self.W = np.concatenate((self.W_dev, self.W_test), axis=0)            
        self.X_s = np.concatenate((self.X_s_dev, self.X_s_test), axis=0)
        self.Y = np.concatenate((self.Y_dev, self.Y_test), axis=0) 
        self.A = np.concatenate((self.A_dev, self.A_test), axis=0)

    def dev_flight_data(self):
        """creates dataframe with development flight input data

        Returns:
            DataFrame: development flight data
        """
        return pd.DataFrame(data = self.W_dev, columns = self.W_var)

    def test_flight_data(self):
        """creates dataframe with test flight input data

        Returns:
            DataFrame: test flight data
        """
        return pd.DataFrame(data = self.W_test, columns = self.W_var)

    def all_flight_data(self):
        """creates dataframe with all flight input data

        Returns:
            DataFrame: all flight data
        """
        return pd.concat([pd.DataFrame(data = self.W_dev, columns = self.W_var),
                          pd.DataFrame(data = self.W_test, columns = self.W_var)])

    def dev_sensor_data(self):
        """creates dataframe with development physical sensor data

        Returns:
            DataFrame: development physical sensor data
        """
        return pd.DataFrame(data = self.X_s_dev, columns = self.X_s_var)

    def test_sensor_data(self):
        """creates dataframe with test physical sensor data

        Returns:
            DataFrame: test physical sensor data
        """
        return pd.DataFrame(data = self.X_s_test, columns = self.X_s_var)

    def all_sensor_data(self):
        """creates dataframe with all physical sensor data

        Returns:
            DataFrame: all physical sensor data
        """
        return pd.concat([pd.DataFrame(data = self.X_s_dev, columns = self.X_s_var),
                          pd.DataFrame(data = self.X_s_test, columns = self.X_s_var)])

    def dev_RUL_data(self):
        """creates dataframe with development RUL data

        Returns:
            DataFrame: development RUL data
        """
        return pd.DataFrame(data = self.Y_dev, columns = ["RUL"])

    def test_RUL_data(self):
        """creates dataframe with test RUL data

        Returns:
            DataFrame: test RUL data
        """
        return pd.DataFrame(data = self.Y_test, columns = ["RUL"])

    def all_RUL_data(self):
        """creates dataframe with  all RUL data

        Returns:
            DataFrame: all RUL data
        """
        return pd.concat([pd.DataFrame(data = self.Y_dev, columns = ["RUL"]),
                          pd.DataFrame(data = self.Y_test, columns = ["RUL"])])

    def dev_aux_data(self):
        """creates dataframe with development auxiliary data

        Returns:
            DataFrame: development auxiliary data
        """
        return pd.DataFrame(data = self.A_dev, columns = self.A_var)

    def test_aux_data(self):
        """creates dataframe with test auxiliary data

        Returns:
            DataFrame: test auxiliary data
        """
        return pd.DataFrame(data = self.A_test, columns = self.A_var)

    def all_aux_data(self):
        """creates dataframe with all auxiliary data

        Returns:
            DataFrame: all auxiliary data
        """
        return pd.concat([pd.DataFrame(data = self.A_dev, columns = self.A_var),
                          pd.DataFrame(data = self.A_test, columns = self.A_var)])


class DataStructure():
    def __init__(self, X_in, y_in, index_in):
        self.X_in = X_in
        self.y_in = y_in
        self.index_in = index_in

    def create_X(self, max_len: int):
        """creates an array of shape compatible with 1D CNN

        Args:
            max_len (int): maximum number of points per cycle

        Returns:
            numpy array: numpy array of shape (cycles, max_len, predictors)
        """
        X_out = []

        for i in tqdm(self.index_in.unit.unique()):
            for j in self.index_in[self.index_in["unit"]==i]["cycle"].unique():
                #predictor data
                working_df=self.X_in[(self.index_in["unit"]==i)&(self.index_in["cycle"]==j)]

                padded_array = pad_sequences([working_df],
                                             padding="post",
                                             value=0,
                                             dtype="float32",
                                             maxlen=max_len)
                X_out.append(padded_array)

        #reshape to remove redundent axis
        X_out = np.reshape(X_out,(np.shape(X_out)[0],
                                  np.shape(X_out)[2],
                                  np.shape(X_out)[3]))
        return X_out

    def create_y(self, piece_wise:bool=True):
        """crate the target data for training

        Args:
            piece_wise (bool, optional): optional piece-wise or linear target. Defaults to True.

        Returns:
            numpy array: array of shape (cycles, RUL)
        """
        #create a dictionary to store RUL data
        y_out = []

        #loop through each unit
        for i in tqdm(self.index_in.unit.unique()):
            #loop through each cycle for a specific unit
            max_cycle = self.y_in[(self.index_in["unit"]==i) & (self.index_in["hs"]==0)]["RUL"].max()

            for j in self.index_in[self.index_in["unit"]==i]["cycle"].unique():
                #reduce the n measurement points for each cycle down to 1 RUL point
                rul=self.y_in[(self.index_in["unit"]==i)&(self.index_in["cycle"]==j)].mean().squeeze()

                if piece_wise:
                    if rul > max_cycle:
                        rul = max_cycle

                #reduce the n health state datapoints for each cycle down to 1 point
                hs=self.index_in[(self.index_in["unit"]==i)&(self.index_in["cycle"]==j)]["hs"].mean()

                y_out.append([i, j, rul, hs])
        return pd.DataFrame(y_out, columns=["unit", "cycle", "RUL", "hs"])

class CreateShapValues():
    """generates SHAP data for export
    """
    def __init__(self, model_in, sample_in, sample_size, features):
        self.model_in = model_in
        self.sample_in = sample_in
        self.sample_size = sample_size
        self.features = features
        self.shap_index_ = np.random.choice(sample_in.shape[0],sample_size,replace=False)
        #create an explainer using DeepExplainer
        self.explainer_ = shap.DeepExplainer(self.model_in,
                                            self.sample_in[np.random.choice(self.sample_in.shape[0],
                                                                            self.sample_size,
                                                                            replace=False)])
        self.expected_value_ = self.explainer_.expected_value[0]

    def create_shap_values(self):
        """creates the shap values for the model

        Returns:
            pd.DataFrame: dataframe of SHAP values
        """
        

        #create a list to store shap numbers
        shap_results=[]

        #loop through random sample
        for i in tqdm(self.shap_index_):
            #calculate shap values one cycle at a time due to OOM issues
            shap_values = self.explainer_.shap_values(self.sample_in[i-1:i],check_additivity=False)

            #reshape shap values from 3d to 2d
            shap_values_2d = shap_values[0].reshape(-1,len(self.features))

            #append shap values to results list
            shap_results.append(shap_values_2d)

        shap_results =  np.asarray(shap_results)
        shap_results = shap_results.reshape(2000*self.sample_size, len(self.features))
        df_shap_results = pd.DataFrame(data=shap_results,
                                       columns=self.features).add_suffix("_SHAP")
        return df_shap_results


    def create_predictor_values(self, scaler_in):
        """creates the corresponding predictor values for the shap values

        Args:
            scaler_in (MinMaxScaler): model scaler for reverse transformation of scaled inputs

        Returns:
            pd.DataFrame: dataframe of corresponding predictor values
        """
        #create a list to store predictors
        predictor_values = []

        #loop through random sample
        for i in tqdm(self.shap_index_):

            #reshape predictor values from 3d to 2d
            sample_in_2d = self.sample_in[i-1:i].reshape(-1,len(self.features))

            #append predictors to results list            
            predictor_values.append(sample_in_2d)

        predictor_values = np.asarray(predictor_values)
        predictor_values = predictor_values.reshape(2000*self.sample_size, len(self.features))
        predictor_values = scaler_in.inverse_transform(predictor_values)
        df_predictor_values = pd.DataFrame(data=predictor_values,
                                           columns=self.features)
        return df_predictor_values