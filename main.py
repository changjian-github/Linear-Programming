# -*- coding: utf-8 -*-
"""
A new file.
"""
import app0, app1, app2, app3, app4
import streamlit as st

def main():
    pages = {
        '0.首页': app0,
        '1.题型回顾': app1,
        '2.增加维度': app2,
        '3.单纯形法': app3,
        '4.小结': app4
    }
    st.sidebar.title('导航栏')
    selection = st.sidebar.radio('', list(pages.keys()))
    page = pages[selection]
    page.app()
    pass

if __name__ == '__main__':
    main()
    pass