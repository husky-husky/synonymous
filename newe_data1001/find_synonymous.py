# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/10/19
import pandas as pd
from tqdm import tqdm


def find_transcript_id():
    mane = pd.read_csv("MANE.GRCh38.v1.0.select_ensembl_genomic.gff", sep="\t",
                       names=["chr", "source", "type", "start", "end", "1", "2", "3", "info"])
    variants = pd.read_csv("mfass/mfass_transcript.csv")
    variants["NM"] = None

    miss_count = 0
    for index, row in tqdm(variants.iterrows()):
        chr_id = row["chr"]
        pos = row["snp_position_hg37_1based"]

        mane_shuffle = mane.loc[(mane["chr"] == chr_id) & (mane["type"] == "gene") &
                                (mane["start"] <= pos) & (mane["end"] >= pos)]
        if mane_shuffle.shape[0] == 0:
            miss_count = miss_count + 1
            continue

        index_loc = mane_shuffle.index.values[0]  # 找到该variant对应的gene

        if mane.iloc[index_loc + 1]["type"] == "transcript":  # 向下取一行，获取transcript的信息
            transcript = mane.iloc[index_loc + 1]["info"]
            transcript = transcript.split(";")
            for info in transcript:
                info = info.split("=")
                if info[0] == "Dbxref":
                    info = info[-1].split(":")
                    variants.loc[index, "NM"] = info[-1]

    variants.to_csv("mfass/mfass_transcript.csv")


if __name__ == "__main__":
    find_transcript_id()
