"""
Student number: 0589870
Student name: Dmitrii Shumilin
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from fault_detector import FaultDetector
import os
"""
To run this insert next command to command line in direction with this file "streamlit run app.py"
"""


def main():
    """
    Main function for streamlit app
    :return:
    """

    st.title('Fault detection algorithm')
    files = os.listdir('data')
    st.sidebar.title('Choosing file')

    normal_dataset_name = st.sidebar.selectbox('Choose dataset with process normal behavior',
                                               files,
                                               key='normal_dataset_name')  # Choosing normal file and fault below
    fault_dataset_name = st.sidebar.selectbox('Choose dataset with fault', files, key='fault_dataset_name')
    type_pca = st.sidebar.radio('Preferable type of PCA analysis', ['PCA', 'DPCA'], key='type_pca')
    isExtended = st.sidebar.checkbox('Enable additional settings')  # Box for choosing methods
    
    if isExtended:  # If checkbox result is True
        L = st.sidebar.slider('Lags number', min_value=1, max_value=50, step=1, key='L')
        a = st.sidebar.slider('Target decomposition dimensions', min_value=1, max_value=52, step=1, key='a')
        isRolling = st.sidebar.checkbox('Rolling window on dataset')
        if isRolling:  # If checkbox for rolling window is True
            window_data = st.sidebar.number_input('Window size', min_value=1, max_value=100, step=1,
                                                  key='window_data_size')
        else:
            window_data = 10
    else:
        L = None
        isRolling = False
        window_data = 10

    norm, fault = load_data(normal_dataset_name, fault_dataset_name)
    
    detector = FaultDetector(data_norm=norm,
                             data_fault=fault,
                             type_PCA=type_pca,
                             L=L,
                             rolling=isRolling,
                             window=window_data)
    if isExtended:
        detector.a = a
    
    print_plots(fault)
    st.text("Note: Time-series rows can be deleted and restored by clicking on their labels on the plot")

    st.subheader('Mean and variance for data variables')
    print_means_std(detector)

    st.markdown("<font size=4><strong>Resulting $T^2$ and $Q$ statistics</strong></font>", unsafe_allow_html=True)
    st.text(f'Current dimensionality a is {detector.a}')
    if type_pca == 'DPCA':
        st.text(f'Current chosen number of lag L is {detector.L}')

    print_statistics(detector)
    st.text("Note: All plots can be interacted with")


def load_data(normal_dataset_name, fault_dataset_name):
    """
    Upload chosen datafiles
    :param normal_dataset_name:  file name for normal data
    :param fault_dataset_name:  file name for data with fault
    :return: norm, fault - pandas DataFrame
    """
    norm = pd.read_table(f"data/{normal_dataset_name}", sep='\s+', header=None)  # Read data file
    fault = pd.read_table(f"data/{fault_dataset_name}", sep='\s+', header=None)  # Read data file
    return norm, fault


def print_plots(data):
    """
    Plot original time series data
    :param data: pd.DataFrame, dataset to plot
    :return: None
    """
    fig = px.line(data, x=data.index, y=data.columns, title='Original time series with fault',
                  template='simple_white')  # Create original time series plot
    fig.update_layout(height=500, width=800, showlegend=True, xaxis_title="Time point (each 3 min)",
                      yaxis_title="Parameter", legend_title="Variables")  # Set decoration for plot
    st.plotly_chart(figure_or_data=fig, use_container_width=False)  # Send it to streamlit
    return


def print_means_std(detector):
    """
    Print plots with mean and std values for scaled fault dataset
    :param detector: class FaultDetector
    :return: None
    """
    # Mean and std for normalized fault data
    means = pd.DataFrame(detector.data1.mean(axis=0))
    std_div = pd.DataFrame(detector.data1.std(axis=0))
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Mean values for normalized fault data",
                                        "Std values for normalized fault data"),
                        x_title='Variable number',
                        y_title='Value')  # Create the box for future plots
    
    fig.add_trace(go.Bar(x=means.index, y=means[0], name='Mean values', ), row=1, col=1)
    fig.add_trace(go.Bar(x=std_div.index, y=std_div[0], name='Std values'), row=1, col=2)
    fig.update_layout(height=400, width=800, showlegend=True, template='simple_white')
    st.plotly_chart(figure_or_data=fig, use_container_width=False)
    return


def print_statistics(detector):
    """
    Plot statistics via methods hotelling_statistic and q_statistic
    :param detector: class FaultDetector
    :return: None
    """
    t_stat = detector.hotelling_statistic()  # Calling methods from class
    q_stat = detector.q_statistic()  # Calling method from class

    isRollingWindow = st.checkbox('Enable rolling window')
    
    if isRollingWindow:
        window = st.number_input('Window size', min_value=1, max_value=100, step=1, key='window_size')
        rol_avg_t = pd.Series(t_stat).rolling(window=window).mean()
        rol_avg_q = pd.Series(q_stat).rolling(window=window).mean()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("T2 statistic for fault data", "Q statistic for fault data"),
                        x_title="Time point (each 3 min)",
                        y_title='Value')
    
    fig.add_trace(go.Scatter(x=np.arange(len(t_stat)), y=t_stat, mode='lines', name='T2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(q_stat)), y=q_stat, mode='lines', name='Q'), row=1, col=2)
    if isRollingWindow:
        fig.add_trace(go.Scatter(x=np.arange(len(rol_avg_t)), y=rol_avg_t, mode='lines', name='MA of T2'), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(rol_avg_q)), y=rol_avg_q, mode='lines', name='MA of Q'), row=1, col=2)
    
    fig.update_layout(height=400, width=800, showlegend=True, template='simple_white')
    st.plotly_chart(figure_or_data=fig, use_container_width=False)


if __name__ == '__main__':
    main()
