# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/10/1
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

"""
根据psi的值生成标签,170个负例，124个正例
"""
# psi_values = pd.read_csv("delta_PSI_values.tsv", sep="\t")
# vex_seq = pd.read_csv("synonymous_vex-seq.csv")
# vex_seq["Class"] = None
#
# psi_values = psi_values[["variant", "delta_PSI"]]
#
# df = pd.merge(vex_seq, psi_values, how="left", on=["variant"])
# neg = 0
# pos = 0
# for index, row in df.iterrows():
#     psi = float(row["delta_PSI"])
#     if abs(psi) < 5:
#         neg = neg + 1
#         vex_seq.loc[index, "Class"] = 0
#     else:
#         pos = pos + 1
#         vex_seq.loc[index, "Class"] = 1
#
# vex_seq.to_csv("synonymous_vex-seq.csv", index=0)

"""
将标签加入数据中
"""
# label = pd.read_csv("synonymous_vex-seq.csv")
# label = label[["Chr", "Pos", "Ref", "Alt", "Class"]]
#
# features = pd.read_excel("vex-seq_input.xlsx")
#
# df = pd.merge(features, label, how="left", on=["Chr", "Pos", "Ref", "Alt"])
# df.to_excel("vex-seq_input.xlsx")

"""
添加trap和cadd得分(可能用不到这三个特征，但是先加上)
"""
# trap_cadd = pd.read_excel("vex_seq_trap_cadd.xlsx")
# features = pd.read_excel("vex-seq_input.xlsx")
#
# df = pd.merge(features, trap_cadd, how="left", on=["Chr", "Pos", "Ref", "Alt"])
# df.to_excel("vex-seq_input.xlsx", index=0)



