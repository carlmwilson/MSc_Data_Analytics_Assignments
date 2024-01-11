import math
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.metrics import r2_score, mean_squared_error


import matplotlib.pyplot as plt

def result_stats(df_in, cycles=10):
    """
    function to determine Mann-Whitney between n cycles at start of units life and n cycles at end of units life
    df_in: dataframe of sensor deltas between measurement and prediction
    cycles: the number of cycles to assess at beginning and end of units life
    returns: dataframe of sensors and units with U scores and P-values
    """
    
    #create a dataframe to store results
    out=[]
    
    #loop through each sensor
    for i in df_in.drop(columns=["unit","cycle","hs"]).columns.unique():
        #create a working dataframe of sensor, unit, cycle and health state
        df_var = df_in[[i,"unit","cycle","hs"]]

        #loop through each unit
        for u in df_var["unit"].unique():
           
            #create a working dataframe selecting the specified unit
            df_unit = df_var[df_var["unit"]==u]
            
            #determine the maximum cycle for the selected unit during health state 0
            hs0_max = df_unit[df_unit["hs"]==0]["cycle"].max()
            
            #minimum cycles for health state 0 is the max minus the number of cycles for analysis
            hs0_min = hs0_max-cycles
            
            #maximum number of cycles for health state 1 is the number of cycles for analysos
            hs1_max = cycles
            
            #minimum number of cycles for health state 1 is always zero
            hs1_min = 0

            #create the unhealthy population
            unhealthy = df_unit[(df_unit["cycle"]>=hs0_min)&(df_unit["cycle"]<=hs0_max)][i]
            
            #create the healthy population
            healthy = df_unit[(df_unit["cycle"]>=hs1_min)&(df_unit["cycle"]<=hs1_max)][i]

            #determine the Mann-Whitney U stat and p-value
            mwu, p = mannwhitneyu(healthy, unhealthy)
    
            #append results to dataframe to be returned
            out.append([i, u, round(mwu,5), round(p,5)])
            
    return pd.DataFrame(out, columns=["sensor", "unit", "U", "p_value"])

def rul_scoring(y_true, y_pred):
    """
    function to calcualte key scores on a scatter plot
    y_true: ground truth to be assessed against - vector
    y_pred: predicts to be scored against ground truth - vector
    returns: scatter plot with RMSE, Nasa score and Sc score function
    """
    r2=r2_score(y_true, y_pred)
    rmse=mean_squared_error(y_true, y_pred,squared=False)
    nasa_score=NASA_score(y_true,y_pred)
    Sc = (0.5*rmse)+(NASA_score(y_true,y_pred))
    nl="\n"
    text=f"R²: {r2:.3f}{nl}RMSE: {rmse:.3f}{nl}NASA Score: {nasa_score:.3f}{nl}Sc: {Sc:.3f}"
    props = dict(boxstyle="round",facecolor="white", alpha=1)
    plt.figure(figsize=(8,8))
    plt.grid(b=True,which="major")
    plt.scatter(x=y_true,y=y_pred)
    plt.xlabel("RUL")
    plt.xlim(0,80)
    plt.ylabel("Prediction")
    plt.ylim(0,80)
    plt.text(2,78,text,fontsize=14,bbox=props,verticalalignment="top")
    plt.show()
    
def NASA_score(y_true, y_pred):
    """
    function to calculate the NASA score for a pair of vectors
    y_true: ground truth to be assessed against - 1d vector
    y_pred: predicts to be scored against ground truth - 1d vector
    returns: NASA score 0d scaler
    """
    df_y_true = pd.DataFrame(data=y_true,columns=["y_true"])
    df_y_pred = pd.DataFrame(data=y_pred,columns=["y_pred"])
    df_score = df_y_true.join(df_y_pred)
    
    df_score["delta"]=df_score["y_true"]-df_score["y_pred"]
    df_score["abs_delta"]=abs(df_score["delta"])
    for i in df_score.index:
        if df_score.loc[i,"delta"]<=0:
            df_score.loc[i,"penalty"]=math.exp(1/13*df_score.loc[i,"abs_delta"])
        else:
            df_score.loc[i,"penalty"]=math.exp(1/10*df_score.loc[i,"abs_delta"])
    score = 1/len(df_score)*df_score["penalty"].sum()
    return score