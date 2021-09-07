import torch
import pickle
import numpy as np
import pandas as pd
import math

import sys
sys.path.insert(0, '../common')
import DataGetter

import AaeEncoder


def getDfWithFeatures(game, targDays, advancedFeatures=False):
    datagetter = DataGetter.DataGetter()
    datagetter.setDataPath("/home/sdeev/Data_science/Code/vadamsky/data")
    (i_revenue_all, i_revenue_not_null, df) = datagetter.getGamesData([game], [], [])

    validInds = pd.Series([False] * len(df))
    for adset in df["Adset"].unique():
        for cohort in df[df["Adset"] == adset]["CohortNum"].unique():
            ind = (df["Adset"] == adset) & (df["CohortNum"] == cohort)
            dft = df[ind]
            if len(dft[dft["DayNum"] == targDays]) > 0:
                validInds = (validInds) | (ind)

    df = df[validInds][["Spend", "Reach", "Impressions", "Clicks", "Installs", "Payers", "DayNum"]]

    # df["Shows"]     = df["Impressions"] / df["Reach"]
    # df["Cost"]      = df["Spend"] / df["Impressions"]
    # df["CostClicks"] = df["Cost"] * df["Clicks"]
    # df["CostInstalls"] = df["Cost"] * df["Installs"]
    # df["Pow2Installs"] = (df["Installs"].values * df["Installs"].values)
    # df["Pow2Clicks"] = (df["Clicks"].values * df["Clicks"].values)

    if advancedFeatures:
        df["effInOfCl"] = df["Installs"] / df["Clicks"]
        df["effInOfRe"] = df["Installs"] / df["Reach"]
        df["effInOfIm"] = df["Installs"] / df["Impressions"]
        df["effInOfSp"] = df["Installs"] / df["Spend"]
        df["effPaOfCl"] = df["Payers"] / df["Clicks"]
        df["effPaOfRe"] = df["Payers"] / df["Reach"]
        df["effPaOfIm"] = df["Payers"] / df["Impressions"]
        df["effPaOfSp"] = df["Payers"] / df["Spend"]
        df["effPaOfIn"] = df["Payers"] / df["Installs"]

    df = df.replace([math.inf, -math.inf], math.nan)
    df = df.fillna(0)

    return df


def load_data(game, featDays=2, targDays=13, train_batch_size=32):
    print('loading data!')
    df = getDfWithFeatures(game, targDays)
    # fill load list
    feat_keys = [c for c in df.columns if c != "DayNum" and c != "Payers"]
    maxs = np.max(np.array(df[feat_keys + ["Payers"]]), axis = 0)
    load_list = []

    for i in range(len(df)):
        if df.iloc[i]['DayNum'] == featDays:
            lst = df.iloc[i][feat_keys].tolist()
            lst.append(df.iloc[i + targDays - featDays]["Payers"])
            i = i + targDays - featDays
            # print(lst)
            load_list.append(lst)

    # normalize list by Impressions
    impressionsInd = 2
    i = 0
    for lst in load_list:
        lst_ = np.array(lst)
        lst_ = lst_ / maxs
        ###lst_ = lst_ / lst_[impressionsInd]
        #impressions = lst[2]
        load_list[i] = lst_.tolist()
        #print(lst_, maxs, load_list[i])
        i = i + 1

    # fill array with train_batch_size
    if train_batch_size>0:
        data = np.zeros(shape=(int(len(load_list) / train_batch_size), train_batch_size, len(feat_keys) + 1), dtype=float)
        dt = np.zeros(shape=(train_batch_size, len(feat_keys) + 1), dtype=float)
        upindex = 0
        index = 0
        for lst in load_list:
            dt[index, :] = lst
            index = index + 1
            if index == train_batch_size:
                index = 0
                #print(dt)
                data[upindex, :, :] = dt
                # if upindex==0:
                #    print(dt)
                dt = np.zeros(shape=(train_batch_size, len(feat_keys) + 1), dtype=float)
                upindex = upindex + 1

        data = torch.from_numpy(data)  # learn_list)
        # X = Variable(torch.from_numpy(learn_list).float(), requires_grad = True)
    else:
        data = torch.from_numpy(np.array(load_list))

    return (data, maxs)


