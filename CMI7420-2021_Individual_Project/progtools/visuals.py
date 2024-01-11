import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_df_color_per_unit(data, variables, labels, size=7, labelsize=17, option='Time'):
    """multi-purpose plotting function for plotting various parameters of specified engine units.
    +++Creit to M. Chao; original code taken from Jupyter notebooks provided with dataset.+++

    Args:
        data (DataFrame): data to be plotted
        variables (list): specific channels to be plotted
        labels (list): y-axis labels
        size (int, optional): height of plot in inches. Defaults to 7.
        labelsize (int, optional): chart label sizes. Defaults to 17.
        option (str, optional): x-axis label. Defaults to 'Time'.
    """
    plt.clf()        
    input_dim = len(variables)
    cols = min(np.floor(input_dim**0.5).astype(int),4)
    rows = (np.ceil(input_dim / cols)).astype(int)
    gs   = gridspec.GridSpec(rows, cols)
    leg  = []
    fig  = plt.figure(figsize=(size,max(size,rows*2)))
    color_dic_unit = {'Unit 1': 'C0', 'Unit 2': 'C1', 'Unit 3': 'C2', 'Unit 4': 'C3', 'Unit 5': 'C4', 'Unit 6': 'C5',
                      'Unit 7': 'C6', 'Unit 8': 'C7', 'Unit 9': 'C8', 'Unit 10': 'C9', 'Unit 11': 'C10',
                      'Unit 12': 'C11', 'Unit 13': 'C12', 'Unit 14': 'C13', 'Unit 15': 'C14', 'Unit 16': 'C15',
                      'Unit 17': 'C16', 'Unit 18': 'C17', 'Unit 19': 'C18', 'Unit 20': 'C19'} 

    unit_sel  = np.unique(data['unit'])
    for n in range(input_dim):
        ax = fig.add_subplot(gs[n])
        for j in unit_sel:
            data_unit = data.loc[data['unit'] == j]
            if option=='cycle':
                time_s = data.loc[data['unit'] == j, 'cycle']
                label_x = 'Time [cycle]'
            else:
                time_s = np.arange(len(data_unit))
                label_x = 'Time [s]'
            ax.plot(time_s, data_unit[variables[n]], '-o', color=color_dic_unit['Unit ' + str(int(j))],
                    alpha=0.7, markersize=5)
            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)
            leg.append('Unit '+str(int(j)))
        plt.ylabel(labels[n], fontsize=labelsize)    
        plt.xlabel(label_x, fontsize=labelsize)
        ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        if n==0:
            ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend(leg, loc='best', fontsize=labelsize-2) #lower left
    plt.tight_layout() 
    plt.show()
    plt.close()

def matrix_plot(data,plot_range=[-1,1]):
    """plot correlation tables as colour plot

    Args:
        data (DataFrame): correlation matrix of pandas type
        plot_range (list, optional): range for colour axis. Defaults to [-1,1].
    """

    #create the figure
    fig = plt.figure(figsize=(20,12))
    #create a 1x1 with first subplot
    ax=fig.add_subplot(111)
    #create a colro axis with ranges
    cax=ax.matshow(data,cmap="PRGn",vmin=plot_range[0],vmax=plot_range[1])
    #include a colour bar
    fig.colorbar(cax,fraction=0.015,pad=0.02)

    #create the x ticks according column axis
    x_ticks=np.arange(0,len(data.columns),1)
    #create the y ticks according the index axis
    y_ticks=np.arange(0,len(data.index),1)

    #set the number of x ticks and label them according to columsn
    ax.set_xticks(x_ticks)
    plt.xticks(rotation=90)
    ax.set_xticklabels(data.columns)

    #set the number of y ticks and label them according to index
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(data.index)

    #include a grid to seperate colours    
    plt.grid(b=True,which="minor",color="black")

    plt.show()
    
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
                          y=["yhat","RUL"],
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
    plt.grid(b=True,which="major")
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