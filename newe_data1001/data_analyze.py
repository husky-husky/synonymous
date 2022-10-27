# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/10/11
import pandas as pd

data = pd.read_excel("vex-seq/vex-seq_input.xlsx")

a = data["Class"].value_counts()
print(1)
