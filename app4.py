# -*- coding: utf-8 -*-
"""
A new file.
"""
import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
# local
from app1 import fig01
from app3 import Model, pivot

def run_simplex(A, b, c):
    # layout
    div_fig = st.empty()
    div_text = st.empty()
    # compute
    mdl = Model(A, b, c)
    while True:
        if np.all(mdl._c<=0):
            break
        pivot(mdl)
        pass
    # display
    m, n = mdl._A.shape
    columns = [f'x{i+1}' for i in range(n)] + ['bnd']
    index = ['obj'] + [f'con{i+1}' for i in range(m)]
    df = pd.DataFrame(np.zeros([m+1, n+1]), columns=columns, index=index)
    options = [f'{l}{i}' for i in range(mdl.num-1) for l in ['mark', 'pivot']]
    label = st.select_slider('label', options=options)
    l = options.index(label)
    i = int((l+1)/2)
    df.values[1:, :-1] = mdl.log['weight'][i]
    df.values[0, :-1] = mdl.log['value'][i]
    df.values[1:, -1] = mdl.log['capacity'][i]
    df.values[0, -1] = mdl.log['objective'][i]
    sol = np.zeros(n)
    sol[mdl.log['basis'][i]] = df.values[1:, -1]
    obj = abs(df.values[0, -1])
    div_text.write(f'sol: {sol[:2]}, obj: {obj}')
    fig = fig01()
    fig.add_trace(go.Scatter(
            x=[obj/2, 0],
            y=[0, obj/3],
            mode='lines',
            line_color='black',
            line_width=1,
            showlegend=False
        ))
    div_fig.plotly_chart(fig)
    pass

def app():
    st.header("小结")
    st.write("S1.与第2部分的枚举法相比，单纯形法计算量更小。其中枚举法要遍历126个端点，\
             单纯形法要转轴5次。")
    st.write("S2.单纯形法的转轴操作对应一次端点移动。")
    label = st.radio('', ['隐藏', '显示'], key='radio4_1')
    if label=='显示':
        st.write("问题：")
        st.latex(
            r'''\begin{array}{l}
            \max 2{x_1} + 3{x_2}\\
            s.t.\left\{ \begin{array}{l}
            {x_1} + 2{x_2} \le 8\\
            0 \le {x_1} \le 4\\
            0 \le {x_2} \le 3\\
            \end{array} \right.
            \end{array}'''
        )
        st.write("图示：")
        A = np.array([
            [1, 2], [1, 0], [0, 1]
        ])
        b = np.array([8, 4, 3])
        c = np.array([2, 3])
        run_simplex(A, b, c)
    pass