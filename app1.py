# -*- coding: utf-8 -*-
"""
A new file.
"""
import numpy as np
import streamlit as st
from plotly import graph_objects as go

def fig01():
    # data
    # A = np.array([
    #     [1, 2],[1, 0],[0, 1]
    # ])
    # b = np.array([8, 4, 3])
    # c = np.array([2, 3])
    # xmin, xmax = -0.5, 8.5
    # ymin, ymax = -0.5, 4.5
    P = np.array([
        [4, 0],[4, 2],[2, 3],[0, 3],[0, 0],[4, 0]
    ])
    T1 = np.array([
        [0, 4],[8, 0]
    ])
    T2 = np.array([
        [4, 0],[4, 4]
    ])
    T3 = np.array([
        [0, 3],[8, 3]
    ])
    # figure
    fig = go.Figure()
    fig.add_trace
    fig.add_trace(go.Scatter(
        x=T1[:, 0],
        y=T1[:, 1],
        mode='lines',
        line_color='blue',
        line_width=1,
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=T2[:, 0],
        y=T2[:, 1],
        mode='lines',
        line_color='blue',
        line_width=1,
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=T3[:, 0],
        y=T3[:, 1],
        mode='lines',
        line_color='blue',
        line_width=1,
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=P[:, 0],
        y=P[:, 1],
        fill='toself',
        fillcolor='cyan',
        mode='lines+markers',
        line_color='blue',
        line_width=1,
        marker_color='red',
        showlegend=False,
    ))
    fig.add_annotation(
        x=6,
        y=1,
        text="x1 + 2x2 = 8",
        showarrow=False,
        xshift=50
    )
    fig.add_annotation(
        x=4,
        y=2.5,
        text="x1 = 4",
        showarrow=False,
        xshift=20
    )
    fig.add_annotation(
        x=3,
        y=3,
        text="x2 = 3",
        showarrow=False,
        yshift=10
    )
    return fig

def app():
    st.header("题型回顾")
    st.write("问：怎么解这个题？")
    st.latex(
        r'''\begin{array}{l}
        \max 2{x_1} + 3{x_2}\\
        s.t.\left\{ \begin{array}{l}
        {x_1} + 2{x_2} \le 8\\
        0 \le {x_1} \le 4\\
        0 \le {x_2} \le 3
        \end{array} \right.
        \end{array}'''
    )
    label = st.radio('', ['隐藏', '显示'], key='radio1_1')
    if label=='显示':
        st.write("答：图解法。")
        fig = fig01()
        obj = st.slider('移动滑块以改变目标值', min_value=0., max_value=14., step=0.1)
        st.write('目标值 = {}'.format(obj))
        fig.add_trace(go.Scatter(
            x=[obj/2, 0],
            y=[0, obj/3],
            mode='lines',
            line_color='black',
            line_width=1,
            showlegend=False
        ))
        st.plotly_chart(fig)
    pass