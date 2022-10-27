# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/10/27
import pandas as pd


def filter_in_three_papers():
    mfass = pd.read_csv("../mfass/synonymous_from_MFASS.csv")
    mfass["Dataset"] = "MFASS"
    vex_seq = pd.read_csv("../vex-seq/synonymous_vex-seq.csv")
    vex_seq["Dataset"] = "Vex-seq"
    MLC_splice = pd.read_csv("../MLCsplice/synonymous_from_MLCsplice.csv")

    vex_seq = vex_seq[["Dataset", "Chr", "Pos", "Ref", "Alt"]]

    df = pd.concat([mfass, vex_seq, MLC_splice])

    a = df.drop_duplicates(subset=["Chr", "Pos", "Ref", "Alt"], keep="first")

    a.to_csv("three_papers_concat.csv", index=0)


def filter_test_and_train():
    three_data = pd.read_csv("three_papers_concat.csv")
    train_data = pd.read_excel("../../final_data_0924.xlsx")
    test_data = pd.read_excel("../../train_and_test/test-new-7-27.xlsx")

    train_data = train_data[["Chr", "Pos", "Ref", "Alt"]]
    test_data = test_data[["Chr", "Pos", "Ref", "Alt"]]
    count = 0
    three_data["is_exist"] = 0
    for index, row in three_data.iterrows():
        a = train_data.loc[(train_data["Chr"] == row["Chr"]) & (train_data["Pos"] == row["Pos"]) &
                           (train_data["Ref"] == row["Ref"]) & (train_data["Alt"] == row["Alt"])]

        b = test_data.loc[(test_data["Chr"] == row["Chr"]) & (test_data["Pos"] == row["Pos"]) &
                          (test_data["Ref"] == row["Ref"]) & (test_data["Alt"] == row["Alt"])]

        if a.shape[0] > 0 or b.shape[0] > 0:
            count = count + 1
            three_data.loc[index, "is_exist"] = 1

    c = three_data.drop(three_data[three_data["is_exist"] == 1].index)
    c.to_csv("three_data_without_duplicate.csv", index=0)


if __name__ == "__main__":
    filter_test_and_train()
