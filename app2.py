# -*- coding: utf-8 -*-
"""
A new file.
"""
import numpy as np
import streamlit as st
from itertools import combinations

def compute_intersects(A, b):
    m, n = A.shape
    Aexp = np.vstack([A, -np.eye(n)])
    bexp = np.hstack([b, [0] * n])
    inters = []
    for comb in combinations(range(m+n), n):
        try:
            inter = np.linalg.solve(Aexp[list(comb)], bexp[list(comb)])
            if np.all(np.matmul(Aexp, inter) - bexp <= 0):
                inters.append(inter)
        except:
            pass
        pass
    return np.array(inters)

def app():
    st.header("增加维度")
    st.write("问：怎么解这个题？")
    st.latex(
        r'''\begin{array}{l}
        \max 2{x_1} + 3{x_2} + {x_3} + {x_4}\\
        s.t.\left\{ \begin{array}{l}
        {x_1} + 2{x_2} + {x_3} + 2{x_4}\le 8\\
        0 \le {x_1} \le 4\\
        0 \le {x_2} \le 3\\
        0 \le {x_3} \le 2\\
        0 \le {x_4} \le 3\\
        \end{array} \right.
        \end{array}'''
    )
    label = st.radio('', ['隐藏', '显示'], key='radio2_1')
    if label=='显示':
        st.write("答：高维空间中图解法不再适用，需要新解法。")
        st.write("问：需要怎样的解法？")
        label2 = st.radio('', ['隐藏', '显示'], key='radio2_2')
        if label2=='显示':
            st.write("答：线性规划的最优解在可行域的端点上，\
                     求出所有端点再找最优解。")
            st.write("问：怎么计算可行域的端点？")
            label3 = st.radio('', ['隐藏', '显示'], key='radio2_3')
            if label3=='显示':
                st.write("答：$n$个线性无关$n$元一次方程组有唯一解，即$n$个线性无关$n$维超平面交于唯一点。")
                A = np.array([
                    [1, 2, 1, 2],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                b = np.array([8, 4, 3, 2, 3])
                c = np.array([2, 3, 1, 1])
                inters = compute_intersects(A, b)
                objs = [np.inner(inter, c) for inter in inters]
                max_idx = np.argmax(objs)
                max_inter = inters[max_idx]
                max_obj = objs[max_idx]
                st.write('现在我们可以求解高维线性规划问题了。方程组有4元9约束\
                         ，共$C_9^4 = 126$组情形，排除共线和不可行还剩余{0}\
                         组情形，即{0}个端点。遍历后求出最优解为{1}，最优目标值为\
                         {2}。'.format(len(inters), max_inter, max_obj))
    pass