# -*- coding: utf-8 -*-
"""
A new file.
"""
import numpy as np
import pandas as pd
import streamlit as st

class Model:
    def __init__(self, A, b, c):
        m, n = A.shape
        self._A = np.hstack([A, np.eye(m)]).astype(float)
        self._b = b.copy().astype(float)
        self._c = np.hstack([c, np.zeros(m)]).astype(float)
        self._z = [0.0]
        self._basis = list(range(n, n+m))
        self.log = {
            'basis': [self._basis.copy()],
            'capacity': [self._b.copy()],
            'objective': [self._z[0]],
            'pivot_col': [],
            'pivot_row': [],
            'value': [self._c.copy()],
            'weight': [self._A.copy()],
        }
        self.num = 1
    pass

def gauss(A, b, c, z, row_idx, col_idx):
    b[row_idx] = b[row_idx] / A[row_idx, col_idx]
    A[row_idx] = A[row_idx] / A[row_idx, col_idx]
    res = list(range(len(A)))
    res.remove(row_idx)
    for j in res:
        if A[j, col_idx]:
            b[j] -= b[row_idx] * A[j, col_idx]
            A[j] -= A[row_idx] * A[j, col_idx]
        pass
    z[0] -= b[row_idx] * c[col_idx]
    c[:] -= A[row_idx] * c[col_idx]
    pass

def pivot(self):
    max_var_idx = np.argmax(self._c)
    tmp = np.array(list(map(lambda x: max(0, x)+1e-9, self._A[:, max_var_idx])))
    min_con_idx = np.argmin(self._b / tmp)
    self._basis[min_con_idx] = max_var_idx
    gauss(self._A, self._b, self._c, self._z, min_con_idx, max_var_idx)
    self.log['basis'].append(self._basis.copy())
    self.log['capacity'].append(self._b.copy())
    self.log['objective'].append(self._z[0])
    self.log['pivot_col'].append(max_var_idx)
    self.log['pivot_row'].append(min_con_idx)
    self.log['value'].append(self._c.copy())
    self.log['weight'].append(self._A.copy())
    self.num += 1
    pass

def run_simplex(A, b, c):
    # layout
    st.subheader('视图')
    div_df = st.empty()
    st.subheader('结果')
    div_win = st.empty()
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
    j = int(l/2)
    col_idx = mdl.log['pivot_col'][j]
    row_idx = mdl.log['pivot_row'][j]
    if l % 2:
        div_df.dataframe(df.style.apply(
            lambda x: ['background: lightgreen'
                       if x.name == f'con{row_idx+1}'
                       else '' for _ in x],
            axis=1
        ))

    else:
        div_df.dataframe(df.style.apply(
            lambda x: ['background: lightgreen'
                       if x.name == f'x{col_idx+1}'
                       else '' for _ in x],
            axis=0
        ))
    val = mdl.log['value'][0]
    sol = np.zeros(n)
    sol[mdl.log['basis'][i]] = df.values[:m, n]
    obj = abs(df.values[0, -1])
    div_win.text(f'val: {val}\nsol: {sol}\nobj: {obj}')
    pass

def app():
    st.header("单纯形法")
    st.write("要求：使用单纯形法解此问题。")
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
    label = st.radio('', ['隐藏', '显示'], key='radio3_1')
    if label=='显示':
        my_expander1 = st.beta_expander(label='步骤1')
        with my_expander1:
            st.write("步骤1.模型标准化")
            st.latex(
                r'''\begin{array}{l}
                \max 2{x_1} + 3{x_2} + {x_3} + {x_4} = z\\
                s.t.\left\{ \begin{array}{l}
                {x_1} + 2{x_2} + {x_3} + 2{x_4} + {x_5} = 8\\
                {x_1} + {x_6} = 4\\
                {x_2} + {x_7} = 3\\
                {x_3} + {x_8} = 2\\
                {x_4} + {x_9} = 3\\
                {x_i} \ge 0\\
                \end{array} \right.
                \end{array}'''
            )
            st.write("矩阵形式：")
            data = np.array([
                [2, 3, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 2, 1, 2, 1, 0, 0, 0, 0, 8],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 4],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 3],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 2],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 3]
            ])
            columns = [f'x{i}' for i in range(1, 10)] + ['int']
            index = ['obj'] + [f'con{i}' for i in range(1, 6)]
            df = pd.DataFrame(data, columns=columns, index=index)
            st.dataframe(df)
            st.write("标准型的约束方程组含有9个变元，5个约束，其解空间维度是4。换个更直观的表现方式：")
            st.latex(
                r'''\begin{array}{l}
                \max 2{x_1} + 3{x_2} + {x_3} + {x_4} = z\\
                s.t.\left\{ \begin{array}{l}
                {x_5} = 8 - {x_1} - 2{x_2} - {x_3} - 2{x_4}\\
                {x_6} = 4 - {x_1}\\
                {x_7} = 3 - {x_2}\\
                {x_8} = 2 - {x_3}\\
                {x_9} = 3 - {x_4}\\
                {x_i} \ge 0\\
                \end{array} \right.
                \end{array}'''
            )
            st.write("$x_5, \dots, x_9$由$x_1, \dots, x_4$唯一决定。其中，\
                     $x_5, \dots, x_9$称为基变量，$x_1, \dots, x_4$称为非基变量。\
                     考虑到$x_i \geq 0$，解空间是$\{x_1, \dots, x_4\}$张成的某个4维多面体。\
                     问题的最优解即该多面体上的某个顶点。所以，该如何找到这个顶点呢？")
        my_expander2 = st.beta_expander(label='步骤2')
        with my_expander2:
            st.write("步骤2.改变基变量")
            st.write("交换基变量和非基变量的两个变元，例如$x_1$和$x_5$，看看会发生什么？")
            data2 = data.copy()
            A = data2[1:, :-1]
            b = data2[1:, -1]
            c = data2[0, :-1]
            z = [data2[0, -1]]
            gauss(A, b, c, z, 0, 0)
            data2[0, -1] = z[0]
            columns = [f'x{i}' for i in range(1, 10)] + ['int']
            index = ['obj'] + [f'con{i}' for i in range(1, 6)]
            df2 = pd.DataFrame(data2, columns=columns, index=index)
            st.dataframe(df2)
            st.write("注意右上角的目标函数省略了$z$，即改变基变量后的目标函数为：")
            st.latex(r'''z - 16 = -x_2 - x_3 - 3x_4 - 2x_5''')
            st.write("令基变量为0，目标函数值是$z = 16$。不过非基变量是$[x_1, x_6, x_7,\
                     x_8, x_9] = [8, -4, 3, 2, 3]$，违背了$x_i \ge 0$的条件。\
                     不过这给了我们一个改变基变量的思路：一是保证非基变量的截距非负，\
                     二是使目标函数的系数均为负。如果存在这样的变换，那么令基变量为0即得到目标函数的最大值。")
        my_expander3 = st.beta_expander(label='步骤3')
        with my_expander3:
            st.write("步骤3.转轴操作")
            st.dataframe(df)
            st.write("注意非基变量与基变量不能随意交换，例如$x_1$可以和$x_5, x_6$交换，\
                     但不能和$x_7, x_8, x_9$交换。因为前者可以互相表示，后者不可以。\
                     我们的目标是使目标函数的系数均为负，且非基变量的截距非负。\
                     不妨选择目标函数中最大系数的对应变元作为入基变量，此处为$x_2$。\
                     约束平面$con_1$、$con_3$与$x_2$相交，前者截距是$8/2 = 4$，要求$x_2 \le 4$；\
                     后者截距是$3/1 = 3$，要求$x_2 \le 3$。两者取交集，因此$x_2$应与$con_3$转轴。\
                     即非基变量$x_2$入基，$con_3$与对应$x_2$的基变量$x_7$出基。\
                     计算所有可行的交换，结果如下表所示：")
            swaps = dict([
                ['x1-x5', [0, 0]],
                ['x1-x6', [1, 0]],
                ['x2-x5', [0, 1]],
                ['x2-x7', [2, 1]],
                ['x3-x5', [0, 2]],
                ['x3-x8', [3, 2]],
                ['x4-x5', [0, 3]],
                ['x4-x9', [4, 3]]
            ])
            index, result = [], []
            for key, value in swaps.items():
                data3 = data.copy()
                A = data3[1:, :-1]
                b = data3[1:, -1]
                c = data3[0, :-1]
                z = [data3[0, -1]]
                gauss(A, b, c, z, *value)
                if np.all(b>=0):
                    index.append(key)
                    result.append(list(b)+list(c)+z)
                pass
            columns = [f'con{i}' for i in range(1, 6)] + [f'x{i}' for i in range(1, 10)] + ['int']
            df3 = pd.DataFrame(result, columns=columns, index=index)
            st.dataframe(df3)
            st.write("结果表明$x_2-x_7$交换是最优的，这验证了我们的推理。")
        my_expander4 = st.beta_expander(label='步骤4')
        with my_expander4:
            st.write("4.完整流程")
            A = data[1:, :-1]
            b = data[1:, -1]
            c = data[0, :-1]
            run_simplex(A, b, c)
    pass