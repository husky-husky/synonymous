# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/10/17
import pandas as pd
from tqdm import tqdm

# data = pd.read_csv("variant_data_info.csv")

"""
判断数据本身有没有重复(没有重复)
"""
# data.drop_duplicates(subset=["Chr", "Pos", "Alt", "Ref"], keep="first", inplace=True)

"""
判断数据与之前的数据是否有重复(无重复)
"""
# train_data = pd.read_excel("../final_data1001_add_vex.xlsx")
# count = 0
# for index, row in data.iterrows():
#     Chr = row["Chr"][3:]
#     a = train_data.loc[(train_data["Chr"] == row["Chr"]) & (train_data["Pos"] == row["Pos"]) &
#                        (train_data["Ref"] == row["Ref"]) & (train_data["Alt"] == row["Alt"])]
#
#     if a.shape[0] > 0:
#         count = count + 1


"""
第三篇论文的数据来源也有mfass，判断数据是否有重合(97条MFASS来源的数据全部重复)
"""
# mfass = pd.read_csv("../mfass/mfass_base_variants_info.csv")
#
# data = data.loc[data["Datasets"] == "MFASS"]
# count = 0
# for index, row in data.iterrows():
#     a = mfass.loc[(mfass["chr"] == row["Chr"]) & (mfass["snp_position_hg37_1based"] == row["Pos"]) &
#                   (mfass["ref_allele"] == row["Ref"]) & (mfass["alt_allele"] == row["Alt"])]
#
#     if a.shape[0] > 0:
#         count = count + 1


"""
生成VCF文件，VariantValidator要求
"""
# mfass = pd.read_csv("variant_data_info.csv")
#
# f = open("third_paper.vcf", "w")
# for index, row in tqdm(mfass.iterrows()):
#     Chr = row["Chr"][3:]
#     text = "chr{}:{}{}>{}".format(Chr, row["Pos"], row["Ref"].upper(),
#                                   row["Alt"].upper())
#     f.write(text + "\n")


"""
读取mane标注结果
"""
# variant_data_info = pd.read_csv("variant_data_info.csv")
#
# mane = pd.read_csv("batch_job_third_paper.txt", sep="\t")
# Chr = []
# Pos = []
# REF = []
# Alt = []
# dataset = []
# for index, row in mane.iterrows():
#     if "=" in row['HGVS_Predicted_Protein']:
#         info = row["Input"]
#         Chr.append(row["GRCh37_CHR"])
#         Pos.append(row["GRCh37_POS"])
#         REF.append(row["GRCh37_REF"])
#         Alt.append(row["GRCh37_ALT"])
#
#         a = "chr{}".format(row["GRCh37_CHR"])
#
#         source = variant_data_info.loc[(variant_data_info["Chr"] == "chr{}".format(row["GRCh37_CHR"])) &
#                                        (variant_data_info["Pos"] == row["GRCh37_POS"]) &
#                                        (variant_data_info["Ref"] == row["GRCh37_REF"]) &
#                                        (variant_data_info["Alt"] == row["GRCh37_ALT"]), "Datasets"]
#         dataset.append(source.values[0])
#
# values = list(zip(dataset, Chr, Pos, REF, Alt))
# df = pd.DataFrame(values, columns=["Dataset", "Chr", "Pos", "Ref", "Alt"])
# df.to_csv("synonymous_from_third_paper.csv", index=0)

"""
再次判断第三篇论文中来源为MFASS的同义变异是否和MFASS（第二篇论文中的数据）重复.结论全部重复
"""
# mfass = pd.read_csv("../mfass/synonymous_from_MFASS.csv")
# third = pd.read_csv("synonymous_from_third_paper.csv")
#
# third = third.loc[third["Dataset"] == "MFASS"]
# count = 0
# for index, row in third.iterrows():
#     a = mfass.loc[(mfass["Chr"] == row["Chr"]) & (mfass["Pos"] == row["Pos"]) &
#                   (mfass["Ref"] == row["Ref"]) & (mfass["Alt"] == row["Alt"])]
#
#     if a.shape[0] == 0:
#         count = count + 1
# print(count)

"""
判断第三篇论文中来源为vex-seq的同义变异是否和vex-seq（第一篇论文中的数据）重复.结论:6个vex_seq数据均未重复
"""
# vex_seq = pd.read_csv("../vex-seq/synonymous_vex-seq.csv")
# third = pd.read_csv("synonymous_from_MLCsplice.csv")
#
# third = third.loc[third["Dataset"] == "Vex-seq"]
# count = 0
# for index, row in third.iterrows():
#     a = vex_seq.loc[(vex_seq["Chr"] == row["Chr"]) & (vex_seq["Pos"] == row["Pos"]) &
#                     (vex_seq["Ref"] == row["Ref"]) & (vex_seq["Alt"] == row["Alt"])]
#
#     if a.shape[0] == 0:
#         count = count + 1
#
# print(count)

"""
重新表明数据来源，加上后缀"_MLC"
"""
MLC_splice = pd.read_csv("synonymous_from_MLCsplice.csv")
MLC_splice["Dataset"] = MLC_splice["Dataset"] + "_MLC"
MLC_splice.to_csv("synonymous_from_MLCsplice.csv", index=0)
print(1)


