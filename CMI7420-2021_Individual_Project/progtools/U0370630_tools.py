import os
import h5py
import math
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


import keras_tuner
from keras_tuner.engine.hyperparameters import HyperParameter
from keras_tuner import BayesianOptimization

def result_stats(df_in, cycles=10):
    """
    function to determine Mann-Whitney between n cycles at start of units life and n cycles at end of units life
    df_in: dataframe of sensor deltas between measurement and prediction
    cycles: the number of cycles to assess at beginning and end of units life
    returns: dataframe of sensors and units with U scores and P-values
    """
    
    #create a dataframe to store results
    df_out=pd.DataFrame()
    
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
            mwu, p = stats.mannwhitneyu(healthy, unhealthy)
    
            #append results to dataframe to be returned
            df_out = df_out.append({
                "sensor":i,
                "unit":u,
                "U":round(mwu,5),
                "p_value":round(p,5),
            },ignore_index=True)
            
    return df_out

def grid_search(build_model, directory_name, project_name, max_trials=400):
    """
    function for grid-searching the hyper-parameter space
    directory_name: directory for storing grid search histories
    project_name: folder name for grid search history storage
    search_res: the resulting tuner object
    """
    search_res = BayesianOptimization(build_model,
                                      objective=keras_tuner.Objective("val_mse",
                                                                direction = "min"),
                                      max_trials=max_trials,
                                      seed=42,
                                      directory=directory_name,
                                      project_name=project_name,
                                      overwrite=False)
    return search_res


def scoring(y_true, y_pred):
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


def unit_plots(df_in,title="Prediction Plot"):
    """
    function to create plots of predictions and targets against cycle number per unit
    df_in: dataframe with predictions and targets
    title: plot title
    returns: figure
    """
    #set the colour scheme for the chart
    pio.templates.default="simple_white"

    #loop through each unit
    for i in df_in["unit"].unique():
        #create a working dataframe of selected unit
        df_working = df_in[df_in["unit"]==i]

        #calcuate the RMSE for hs0, hs1 and overall [Hsa]
        hs0_RMSE = ((1/len(df_working[df_working["hs"]==0]))*(df_working[df_working["hs"]==0]["delta_sq"].sum()))**0.5
        hs1_RMSE = ((1/len(df_working[df_working["hs"]==1]))*(df_working[df_working["hs"]==1]["delta_sq"].sum()))**0.5
        hsa_RMSE = (1/len(df_working)*(df_working["delta_sq"].sum()))**0.5

        #create the figure with a title of RSME's
        res_fig = px.scatter(df_working,
                          x="cycle",
                          y=["pred","RUL"],
                         title=f"{title} - Unit: {i:.0f} - Unhealthy RMSE: {hs0_RMSE:.2f} - Healthy RMSE: {hs1_RMSE:.2f} - Total: {hsa_RMSE:.2f}")
        res_fig.update_xaxes(range=[0,100],
                            #autorange="reversed",
                             title="Cycle [-]"
                            )
        res_fig.update_yaxes(range=[0,100],
                            title="Remaining Useful Life [-]")
        res_fig.show()


def shap_interaction_plot(df_in, x, y, color):
    """
    function to create an interaction plot for 3 axes:
    df_in: dataframe with at least 3 columns of data
    x: column name for x-axis
    y: column name for y-axes
    color: column name for colour axis 
    """
    plt.figure(figsize=(8,6))
    #plt.grid(b=True,which="major")
    sc = plt.scatter(df_in[[x]].squeeze(),
                     df_in[[y]].squeeze(),
                     c=df_in[[color]].squeeze(),
                    cmap=plt.cm.get_cmap("jet"))
    plt.colorbar(sc,label=color)
    plt.xlabel(x,fontsize=14)
    plt.ylabel(y,fontsize=14)
    plt.show()


def time_plot(df_in, y1, y2):
    """
    function to create a lineplot with two y-axes; x is always fixed at 2000 due to datashape
    df_in: dataframe with at least two columns
    y1: column name for y axis 1
    y2: column name for y axis 2
    """
    
    #set the colour scheme for the chart
    pio.templates.default="simple_white"

    #create the baseplot with a second y acxis
    fig = make_subplots(specs=[[{"secondary_y":True}]])

    #add the first trace for y axis 1
    fig.add_trace(go.Scatter(x=np.arange(0,2000,1),
                  y=df_in[y1],
                 name=y1),
                 secondary_y=False)

    #add the second trace for y axis 2
    fig.add_trace(go.Scatter(x=np.arange(0,2000,1),
                  y=df_in[y2],
                 name=y2),
                 secondary_y=True)

    #configure the layout and position for legend
    fig.update_layout(height=600,
                     width=600,
                     legend=dict(yanchor="top",
                                y=0.99,
                                xanchor="right",
                                x=0.85))
    #adjust x axes label
    fig.update_xaxes(title="time [10s]", mirror=True, showline=True)
    #adjust yaxes 1 label
    fig.update_yaxes(title=y1,secondary_y=False)
    #adjust yaxes 2 label
    fig.update_yaxes(title=y2,secondary_y=True)
    fig.show()