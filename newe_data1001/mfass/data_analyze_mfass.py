# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/10/6
import pandas as pd
from tqdm import tqdm

"""
生成vcf格式
"""
# mfass = pd.read_csv("mfass_base_variants_info.csv")
#
# f = open("mfass2.vcf", "w")
# for index, row in tqdm(mfass.iterrows()):
#     Chr = row["chr"][3:]
#     text = "chr{}:{}{}>{}".format(Chr, row["snp_position_hg37_1based"], row["ref_allele"].upper(),
#                                   row["alt_allele"].upper())
#     f.write(text + "\n")

"""
读取mane标注结果
"""
mane = pd.read_csv("batch_job_mfass.txt", sep="\t")
Chr = []
Pos = []
REF = []
Alt = []
for index, row in mane.iterrows():
    if "=" in row['HGVS_Predicted_Protein']:
        info = row["Input"]
        Chr.append(row["GRCh37_CHR"])
        Pos.append(row["GRCh37_POS"])
        REF.append(row["GRCh37_REF"])
        Alt.append(row["GRCh37_ALT"])

    values = list(zip(Chr, Pos, REF, Alt))
    df = pd.DataFrame(values, columns=["Chr", "Pos", "Ref", "Alt"])
    df.to_csv("synonymous_from_MFASS.csv", index=0)


print(1)
