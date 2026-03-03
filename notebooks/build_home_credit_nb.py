# -*- coding: utf-8 -*-
import json
import textwrap
from pathlib import Path


def to_source(text: str) -> list[str]:
    lines = textwrap.dedent(text).strip("\n").split("\n")
    return [line + "\n" for line in lines]


def main() -> None:
    cells: list[dict] = []

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": to_source(
                """
# Kaggle Home Credit Default Risk — pipeline complet (pandas + LightGBM)

Notebook autonome pour Kaggle : nettoyage ciblé, feature engineering multi-tables, agrégations et LightGBM CV.
"""
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """
from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

DATA_PATH = Path("../input/home-credit-default-risk")
GENERATE_SUBMISSION_FILES = True
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 200)


@contextmanager
def timer(title: str):
    t0 = time.time()
    yield
    print(f"{title} - elapsed: {time.time() - t0:.1f}s")
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """
from typing import Tuple


def safe_div(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Memory {start_mem:.1f} -> {end_mem:.1f} MB")
    return df


def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True) -> Tuple[pd.DataFrame, list[str]]:
    original_columns = list(df.columns)
    df = pd.get_dummies(df, columns=[c for c in df.columns if df[c].dtype == "object"], dummy_na=nan_as_category)
    new_cols = [c for c in df.columns if c not in original_columns]
    return df, new_cols


def label_encoder(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col], _ = pd.factorize(df[col])
    return df


def get_age_label(days_birth: float) -> int:
    age_years = -days_birth / 365
    if age_years < 27:
        return 1
    if age_years < 40:
        return 2
    if age_years < 50:
        return 3
    if age_years < 65:
        return 4
    if age_years < 99:
        return 5
    return 0


APP_DROP_COLS = [
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "EMERGENCYSTATE_MODE",
]
DOC_FLAGS = [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]


def drop_application_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in APP_DROP_COLS if c in df.columns]
    drop_cols += [f"FLAG_DOCUMENT_{i}" for i in DOC_FLAGS if f"FLAG_DOCUMENT_{i}" in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def group_merge(df: pd.DataFrame, group_col: str, agg_map: dict, prefix: str) -> pd.DataFrame:
    g = df.groupby(group_col).agg(agg_map)
    g.columns = [f"{prefix}{group_col}_{col}_{stat}".upper() for col, stat in g.columns]
    g.reset_index(inplace=True)
    return df.merge(g, on=group_col, how="left")
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def get_train_test(data_path: Path = DATA_PATH) -> pd.DataFrame:
    train = pd.read_csv(data_path / "application_train.csv")
    test = pd.read_csv(data_path / "application_test.csv")
    test["TARGET"] = np.nan
    df = pd.concat([train, test], ignore_index=True)

    df = df[df["CODE_GENDER"] != "XNA"]
    df = df[df["AMT_INCOME_TOTAL"] < 20000000]
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    df.loc[df["DAYS_LAST_PHONE_CHANGE"] == 0, "DAYS_LAST_PHONE_CHANGE"] = np.nan

    docs = [c for c in df.columns if "FLAG_DOC" in c or "FLAG_DOCUMENT" in c]
    df["DOCUMENT_COUNT"] = df[docs].sum(axis=1)
    df["NEW_DOC_KURT"] = df[docs].kurtosis(axis=1)

    df["AGE_RANGE"] = df["DAYS_BIRTH"].apply(get_age_label)

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCES_PROD"] = df[ext_cols].prod(axis=1)
    df["EXT_SOURCES_WEIGHTED"] = df["EXT_SOURCE_1"] * 2 + df["EXT_SOURCE_2"] * 1 + df["EXT_SOURCE_3"] * 3
    df["EXT_SOURCES_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCES_MAX"] = df[ext_cols].max(axis=1)
    df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCES_NANMEDIAN"] = np.nanmedian(df[ext_cols], axis=1)
    df["EXT_SOURCES_VAR"] = df[ext_cols].var(axis=1)

    df["CREDIT_TO_ANNUITY_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["CREDIT_TO_GOODS_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
    df["ANNUITY_TO_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TO_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["INCOME_TO_EMPLOYED_RATIO"] = safe_div(df["AMT_INCOME_TOTAL"], df["DAYS_EMPLOYED"])
    df["INCOME_TO_BIRTH_RATIO"] = safe_div(df["AMT_INCOME_TOTAL"], df["DAYS_BIRTH"])
    df["EMPLOYED_TO_BIRTH_RATIO"] = safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["ID_TO_BIRTH_RATIO"] = safe_div(df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"])
    df["CAR_TO_BIRTH_RATIO"] = safe_div(df["OWN_CAR_AGE"], df["DAYS_BIRTH"])
    df["CAR_TO_EMPLOYED_RATIO"] = safe_div(df["OWN_CAR_AGE"], df["DAYS_EMPLOYED"])
    df["PHONE_TO_BIRTH_RATIO"] = safe_div(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"])

    group_cols = ["ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "AGE_RANGE", "CODE_GENDER"]
    group_stats = {
        "EXT_SOURCES_MEAN": ["median", "std"],
        "AMT_INCOME_TOTAL": ["mean", "std"],
        "CREDIT_TO_ANNUITY_RATIO": ["mean", "std"],
        "AMT_CREDIT": ["mean"],
        "AMT_ANNUITY": ["mean", "std"],
    }
    for gcol in group_cols:
        df = group_merge(df, gcol, group_stats, prefix="GRP_")

    df = label_encoder(df)
    df = drop_application_columns(df)
    return df
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def get_bureau_balance(data_path: Path = DATA_PATH) -> pd.DataFrame:
    bb = pd.read_csv(data_path / "bureau_balance.csv")
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=False)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg({"MONTHS_BALANCE": ["min", "max", "mean", "size"]})
    for col in bb_cat:
        bb_agg[col] = bb.groupby("SK_ID_BUREAU")[col].mean()
    bb_agg.columns = [f"BB_{c[0]}_{c[1]}" if isinstance(c, tuple) else f"BB_{c}" for c in bb_agg.columns]
    bb_agg.reset_index(inplace=True)
    return bb_agg


def agg_and_prefix(df: pd.DataFrame, group_key: str, agg_map: dict, prefix: str) -> pd.DataFrame:
    agg = df.groupby(group_key).agg(agg_map)
    agg.columns = [f"{prefix}{col}_{stat}" for col, stat in agg.columns]
    agg.reset_index(inplace=True)
    return agg


def get_bureau(data_path: Path = DATA_PATH) -> pd.DataFrame:
    bureau = pd.read_csv(data_path / "bureau.csv")

    bureau["CREDIT_DURATION"] = -bureau["DAYS_CREDIT"] + bureau["DAYS_CREDIT_ENDDATE"]
    bureau["ENDDATE_DIF"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_ENDDATE_FACT"]
    bureau["DEBT_PERCENTAGE"] = safe_div(bureau["AMT_CREDIT_SUM"], bureau["AMT_CREDIT_SUM_DEBT"])
    bureau["DEBT_CREDIT_DIFF"] = bureau["AMT_CREDIT_SUM"] - bureau["AMT_CREDIT_SUM_DEBT"]
    bureau["CREDIT_TO_ANNUITY_RATIO"] = safe_div(bureau["AMT_CREDIT_SUM"], bureau["AMT_ANNUITY"])

    bb_agg = get_bureau_balance(data_path)
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=True)
    status_cols = [c for c in bureau.columns if c.startswith("STATUS_")]
    bureau["STATUS_12345"] = bureau[[c for c in status_cols if any(s in c for s in ["1", "2", "3", "4", "5"])]].sum(axis=1)

    if "BB_MONTHS_BALANCE_size" in bureau.columns:
        ll = bureau.groupby("BB_MONTHS_BALANCE_size").agg({"AMT_CREDIT_SUM": ["mean"], "STATUS_12345": ["mean"]})
        ll.columns = [f"LL_{c[0]}_{c[1]}" for c in ll.columns]
        ll.reset_index(inplace=True)
        bureau = bureau.merge(ll, on="BB_MONTHS_BALANCE_size", how="left")

    num_agg = {
        "SK_ID_BUREAU": ["count"],
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "CREDIT_DURATION": ["mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_CREDIT_SUM_INVEST": ["mean"],
        "AMT_ANNUITY": ["max", "mean"],
        "DEBT_PERCENTAGE": ["mean"],
        "DEBT_CREDIT_DIFF": ["mean"],
        "ENDDATE_DIF": ["mean"],
        "STATUS_12345": ["mean"],
    }
    for cat in bureau_cat:
        num_agg[cat] = ["mean"]

    bureau_agg = agg_and_prefix(bureau, "SK_ID_CURR", num_agg, prefix="BUREAU_")

    for status, prefix in [("Active", "BUREAU_ACTIVE_"), ("Closed", "BUREAU_CLOSED_")]:
        flag = f"CREDIT_ACTIVE_{status}"
        if flag in bureau.columns:
            subset = bureau[bureau[flag] == 1]
            if not subset.empty:
                agg = agg_and_prefix(subset, "SK_ID_CURR", num_agg, prefix=prefix)
                bureau_agg = bureau_agg.merge(agg, on="SK_ID_CURR", how="left")

    for ctype in ["Consumer credit", "Credit card", "Mortgage", "Car loan", "Microloan"]:
        col = f"CREDIT_TYPE_{ctype}"
        if col in bureau.columns:
            sub = bureau[bureau[col] == 1]
            if not sub.empty:
                agg = agg_and_prefix(sub, "SK_ID_CURR", num_agg, prefix=f"BUREAU_{ctype.replace(' ', '_').upper()}_")
                bureau_agg = bureau_agg.merge(agg, on="SK_ID_CURR", how="left")

    for months in [6, 12]:
        recent = bureau[bureau["DAYS_CREDIT"] >= -30 * months]
        if not recent.empty:
            agg = agg_and_prefix(recent, "SK_ID_CURR", num_agg, prefix=f"BUREAU_LAST{months}M_")
            bureau_agg = bureau_agg.merge(agg, on="SK_ID_CURR", how="left")

    last_overdue = (
        bureau.sort_values("DAYS_CREDIT")
        .groupby("SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"]
        .last()
        .reset_index()
        .rename(columns={"AMT_CREDIT_MAX_OVERDUE": "BUREAU_LAST_LOAN_MAX_OVERDUE"})
    )
    bureau_agg = bureau_agg.merge(last_overdue, on="SK_ID_CURR", how="left")

    debt_credit = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"].sum().reset_index()
    debt_credit["BUREAU_DEBT_OVER_CREDIT"] = safe_div(debt_credit["AMT_CREDIT_SUM_DEBT"], debt_credit["AMT_CREDIT_SUM"])
    bureau_agg = bureau_agg.merge(debt_credit[["SK_ID_CURR", "BUREAU_DEBT_OVER_CREDIT"]], on="SK_ID_CURR", how="left")

    if "CREDIT_ACTIVE_Active" in bureau.columns:
        active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
        dc_active = active.groupby("SK_ID_CURR")["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"].sum().reset_index()
        dc_active["BUREAU_ACTIVE_DEBT_OVER_CREDIT"] = safe_div(dc_active["AMT_CREDIT_SUM_DEBT"], dc_active["AMT_CREDIT_SUM"])
        bureau_agg = bureau_agg.merge(dc_active[["SK_ID_CURR", "BUREAU_ACTIVE_DEBT_OVER_CREDIT"]], on="SK_ID_CURR", how="left")

    return bureau_agg
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def get_previous_applications(data_path: Path = DATA_PATH) -> pd.DataFrame:
    prev = pd.read_csv(data_path / "previous_application.csv")
    pay = pd.read_csv(data_path / "installments_payments.csv")

    date_cols = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    for c in date_cols:
        prev.loc[prev[c] == 365243, c] = np.nan
    prev["DAYS_LAST_DUE_DIFF"] = prev["DAYS_LAST_DUE"] - prev["DAYS_FIRST_DUE"]

    encode_cols = [
        "NAME_CONTRACT_STATUS",
        "NAME_CONTRACT_TYPE",
        "CHANNEL_TYPE",
        "NAME_TYPE_SUITE",
        "NAME_YIELD_GROUP",
        "PRODUCT_COMBINATION",
        "NAME_PRODUCT_TYPE",
        "NAME_CLIENT_TYPE",
    ]
    prev, prev_cat = one_hot_encoder(prev, nan_as_category=True)
    prev_cat = [c for c in prev_cat if any(col in c for col in encode_cols)]

    prev["APPLICATION_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["APPLICATION_CREDIT_RATIO"] = safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])
    prev["CREDIT_TO_ANNUITY_RATIO"] = safe_div(prev["AMT_CREDIT"], prev["AMT_ANNUITY"])
    prev["DOWN_PAYMENT_TO_CREDIT"] = safe_div(prev["AMT_DOWN_PAYMENT"], prev["AMT_CREDIT"])
    prev["SIMPLE_INTERESTS"] = safe_div(prev["AMT_ANNUITY"] * prev["CNT_PAYMENT"], prev["AMT_CREDIT"]) - 1

    approved_flag = prev.get("NAME_CONTRACT_STATUS_Approved")
    active_mask = prev["DAYS_LAST_DUE"].fillna(0) == 365243
    active_df = prev[(approved_flag == 1) & active_mask]
    active_pay = pay[pay["SK_ID_PREV"].isin(active_df["SK_ID_PREV"].unique())]
    if not active_df.empty and not active_pay.empty:
        pay_sum = active_pay.groupby("SK_ID_PREV")["AMT_INSTALMENT", "AMT_PAYMENT"].sum().reset_index()
        active_features = active_df[["SK_ID_PREV", "SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY"]].merge(pay_sum, on="SK_ID_PREV", how="left")
        active_features["INSTALMENT_PAYMENT_DIFF"] = active_features["AMT_PAYMENT"] - active_features["AMT_INSTALMENT"]
        active_features["REMAINING_DEBT"] = active_features["AMT_CREDIT"] - active_features["AMT_PAYMENT"]
        active_features["REPAYMENT_RATIO"] = safe_div(active_features["AMT_PAYMENT"], active_features["AMT_INSTALMENT"])
        total_active = (
            active_features.groupby("SK_ID_CURR")["REPAYMENT_RATIO"].mean().reset_index().rename(columns={"REPAYMENT_RATIO": "PREVIOUS_TOTAL_REPAYMENT_RATIO"})
        )
    else:
        total_active = pd.DataFrame(columns=["SK_ID_CURR", "PREVIOUS_TOTAL_REPAYMENT_RATIO"])

    pay["LATE_PAYMENT"] = (pay["DAYS_ENTRY_PAYMENT"] - pay["DAYS_INSTALMENT"]) > 0

    prev_agg = {
        "AMT_ANNUITY": ["max", "mean"],
        "AMT_APPLICATION": ["max", "mean", "min"],
        "AMT_CREDIT": ["max", "mean", "min", "sum"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "AMT_GOODS_PRICE": ["mean"],
        "HOUR_APPR_PROCESS_START": ["mean"],
        "RATE_DOWN_PAYMENT": ["mean"],
        "CNT_PAYMENT": ["mean", "max"],
        "DAYS_DECISION": ["min", "max", "mean", "var"],
        "APPLICATION_CREDIT_DIFF": ["mean", "max", "min"],
        "APPLICATION_CREDIT_RATIO": ["mean", "max"],
        "CREDIT_TO_ANNUITY_RATIO": ["mean"],
        "DOWN_PAYMENT_TO_CREDIT": ["mean"],
        "SIMPLE_INTERESTS": ["mean"],
    }
    for cat in prev_cat:
        prev_agg[cat] = ["mean"]

    prev_grouped = agg_and_prefix(prev, "SK_ID_CURR", prev_agg, prefix="PREVIOUS_")

    for status, prefix in [("Approved", "PREVIOUS_APPROVED_"), ("Refused", "PREVIOUS_REFUSED_")]:
        flag = f"NAME_CONTRACT_STATUS_{status}"
        if flag in prev.columns:
            subset = prev[prev[flag] == 1]
            if not subset.empty:
                agg = agg_and_prefix(subset, "SK_ID_CURR", prev_agg, prefix=prefix)
                prev_grouped = prev_grouped.merge(agg, on="SK_ID_CURR", how="left")

    for loan_type, prefix in [("Consumer loans", "PREVIOUS_CONSUMER_"), ("Cash loans", "PREVIOUS_CASH_")]:
        col = f"NAME_CONTRACT_TYPE_{loan_type}"
        if col in prev.columns:
            subset = prev[prev[col] == 1]
            if not subset.empty:
                agg = agg_and_prefix(subset, "SK_ID_CURR", prev_agg, prefix=prefix)
                prev_grouped = prev_grouped.merge(agg, on="SK_ID_CURR", how="left")

    late_pay = agg_and_prefix(pay, "SK_ID_CURR", {"LATE_PAYMENT": ["mean", "sum"], "DAYS_INSTALMENT": ["mean"]}, prefix="PREVIOUS_LATE_")
    prev_grouped = prev_grouped.merge(late_pay, on="SK_ID_CURR", how="left")

    for months in [12, 24]:
        recent = prev[prev["DAYS_DECISION"] >= -30 * months]
        if not recent.empty:
            agg = agg_and_prefix(recent, "SK_ID_CURR", prev_agg, prefix=f"PREVIOUS_LAST{months}M_")
            prev_grouped = prev_grouped.merge(agg, on="SK_ID_CURR", how="left")

    if not total_active.empty:
        prev_grouped = prev_grouped.merge(total_active, on="SK_ID_CURR", how="left")

    return prev_grouped
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def get_pos_cash(data_path: Path = DATA_PATH) -> pd.DataFrame:
    pos = pd.read_csv(data_path / "POS_CASH_balance.csv")
    pos["LATE_PAYMENT"] = (pos["SK_DPD"] > 0).astype(int)
    pos, pos_cat = one_hot_encoder(pos, nan_as_category=True)

    pos_agg = {"MONTHS_BALANCE": ["max", "mean", "size"], "SK_DPD": ["mean", "max"], "SK_DPD_DEF": ["mean", "max"], "LATE_PAYMENT": ["mean", "sum"]}
    for c in pos_cat:
        pos_agg[c] = ["mean"]
    pos_grouped = agg_and_prefix(pos, "SK_ID_CURR", pos_agg, prefix="POS_")

    pos_sorted = pos.sort_values(["SK_ID_PREV", "MONTHS_BALANCE"])
    last_per_prev = pos_sorted.groupby("SK_ID_PREV").agg({"MONTHS_BALANCE": "max", "LATE_PAYMENT": "sum"}).rename(columns={"MONTHS_BALANCE": "MONTHS_BALANCE_MAX", "LATE_PAYMENT": "LATE_PAYMENT_SUM"})
    last_per_prev["POS_REMAINING_INSTALMENTS"] = -last_per_prev["MONTHS_BALANCE_MAX"]
    last_per_prev["POS_REMAINING_INSTALMENTS_RATIO"] = safe_div(last_per_prev["POS_REMAINING_INSTALMENTS"], last_per_prev["MONTHS_BALANCE_MAX"].abs() + 1)
    if "NAME_CONTRACT_STATUS_Completed" in pos.columns:
        completed = pos_sorted.groupby("SK_ID_PREV")["NAME_CONTRACT_STATUS_Completed"].mean().rename("POS_LOAN_COMPLETED_MEAN")
        last_per_prev = last_per_prev.join(completed, how="left")
    if "NAME_CONTRACT_STATUS_Completed" in pos.columns:
        completed_before = pos_sorted[pos_sorted["MONTHS_BALANCE"] < -1].groupby("SK_ID_PREV")["NAME_CONTRACT_STATUS_Completed"].mean().rename("POS_COMPLETED_BEFORE_MEAN")
        last_per_prev = last_per_prev.join(completed_before, how="left")

    prev_merge = pos_sorted[["SK_ID_CURR", "SK_ID_PREV"]].drop_duplicates().merge(last_per_prev.reset_index(), on="SK_ID_PREV", how="left")
    recent_prev = prev_merge.sort_values("MONTHS_BALANCE_MAX").groupby("SK_ID_CURR").tail(3)
    late_recent = recent_prev.groupby("SK_ID_CURR")["LATE_PAYMENT_SUM"].mean().reset_index().rename(columns={"LATE_PAYMENT_SUM": "POS_RECENT3_LATE_PAYMENT_MEAN"})

    pos_grouped = pos_grouped.merge(
        prev_merge.groupby("SK_ID_CURR").agg(
            {
                "MONTHS_BALANCE_MAX": "mean",
                "POS_LOAN_COMPLETED_MEAN": "mean",
                "POS_COMPLETED_BEFORE_MEAN": "mean",
                "POS_REMAINING_INSTALMENTS": "mean",
                "POS_REMAINING_INSTALMENTS_RATIO": "mean",
            }
        ).reset_index(),
        on="SK_ID_CURR",
        how="left",
    )

    pos_grouped = pos_grouped.merge(late_recent, on="SK_ID_CURR", how="left")

    for col in [
        "POS_NAME_CONTRACT_STATUS_Canceled_MEAN",
        "POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN",
        "POS_NAME_CONTRACT_STATUS_XNA_MEAN",
    ]:
        if col in pos_grouped.columns:
            pos_grouped.drop(columns=[col], inplace=True)
    return pos_grouped
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def compute_trend_features(df: pd.DataFrame, value_col: str, periods: list[int]) -> pd.DataFrame:
    trends = []
    for key, group in df.sort_values("DAYS_INSTALMENT").groupby("SK_ID_CURR"):
        for p in periods:
            tail = group.tail(p)
            if len(tail) < 2:
                continue
            x = np.arange(len(tail)).reshape(-1, 1)
            y = tail[value_col].values.reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            trends.append({"SK_ID_CURR": key, f"INS_{value_col}_TREND_{p}": float(model.coef_[0])})
    if not trends:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    trend_df = pd.DataFrame(trends)
    trend_df = trend_df.groupby("SK_ID_CURR").first().reset_index()
    return trend_df


def get_installment_payments(data_path: Path = DATA_PATH) -> pd.DataFrame:
    ins = pd.read_csv(data_path / "installments_payments.csv")

    ins["AMT_PAYMENT_GROUPED"] = ins.groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])["AMT_PAYMENT"].transform("sum")
    ins["PAYMENT_DIFFERENCE"] = ins["AMT_PAYMENT_GROUPED"] - ins["AMT_INSTALMENT"]
    ins["PAYMENT_RATIO"] = safe_div(ins["AMT_PAYMENT_GROUPED"], ins["AMT_INSTALMENT"])
    ins["PAID_OVER_AMOUNT"] = ins["AMT_PAYMENT_GROUPED"] - ins["AMT_INSTALMENT"]
    ins["PAID_OVER"] = (ins["PAID_OVER_AMOUNT"] > 0).astype(int)
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    ins["LATE_PAYMENT"] = (ins["DBD"] > 0).astype(int)
    ins["INSTALMENT_PAYMENT_RATIO"] = safe_div(ins["AMT_PAYMENT_GROUPED"], ins["AMT_INSTALMENT"])
    ins["LATE_PAYMENT_RATIO"] = safe_div(ins["LATE_PAYMENT"], ins["AMT_PAYMENT_GROUPED"].replace(0, np.nan))
    ins["SIGNIFICANT_LATE_PAYMENT"] = (ins["INSTALMENT_PAYMENT_RATIO"] > 1.05).astype(int)
    ins["DPD_7"] = (ins["DPD"] > 7).astype(int)
    ins["DPD_15"] = (ins["DPD"] > 15).astype(int)

    ins_agg = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["mean", "max"],
        "DBD": ["mean", "max"],
        "PAYMENT_DIFFERENCE": ["mean", "max"],
        "PAYMENT_RATIO": ["mean", "max"],
        "PAID_OVER_AMOUNT": ["mean", "max"],
        "PAID_OVER": ["mean"],
        "LATE_PAYMENT": ["mean", "sum"],
        "INSTALMENT_PAYMENT_RATIO": ["mean"],
        "LATE_PAYMENT_RATIO": ["mean"],
        "SIGNIFICANT_LATE_PAYMENT": ["mean"],
        "DPD_7": ["mean"],
        "DPD_15": ["mean"],
        "AMT_PAYMENT": ["sum"],
        "AMT_INSTALMENT": ["sum"],
    }

    ins_grouped = agg_and_prefix(ins, "SK_ID_CURR", ins_agg, prefix="INS_")

    for months in [36, 60]:
        recent = ins[ins["DAYS_INSTALMENT"] >= -30 * months]
        if not recent.empty:
            agg = agg_and_prefix(recent, "SK_ID_CURR", ins_agg, prefix=f"INS_LAST{months}M_")
            ins_grouped = ins_grouped.merge(agg, on="SK_ID_CURR", how="left")

    trends_dpd = compute_trend_features(ins, "DPD", periods=[12, 24, 60, 120])
    trends_paid = compute_trend_features(ins, "PAID_OVER_AMOUNT", periods=[12, 24, 60, 120])
    for tr in [trends_dpd, trends_paid]:
        if not tr.empty:
            ins_grouped = ins_grouped.merge(tr, on="SK_ID_CURR", how="left")

    last_prev = ins.sort_values("DAYS_INSTALMENT").groupby("SK_ID_CURR").tail(1)
    last_features = last_prev[["SK_ID_CURR", "PAYMENT_RATIO", "PAYMENT_DIFFERENCE", "DPD", "PAID_OVER_AMOUNT"]]
    last_features = last_features.rename(
        columns={
            "PAYMENT_RATIO": "INS_LAST_PAYMENT_RATIO",
            "PAYMENT_DIFFERENCE": "INS_LAST_PAYMENT_DIFF",
            "DPD": "INS_LAST_DPD",
            "PAID_OVER_AMOUNT": "INS_LAST_PAID_OVER_AMOUNT",
        }
    )
    ins_grouped = ins_grouped.merge(last_features, on="SK_ID_CURR", how="left")

    return ins_grouped
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def get_credit_card(data_path: Path = DATA_PATH) -> pd.DataFrame:
    cc = pd.read_csv(data_path / "credit_card_balance.csv")
    if "AMT_RECIVABLE" in cc.columns:
        cc = cc.rename(columns={"AMT_RECIVABLE": "AMT_RECEIVABLE"})

    cc, cc_cat = one_hot_encoder(cc, nan_as_category=True)
    cc["LIMIT_USE"] = safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["PAYMENT_DIV_MIN"] = safe_div(cc["AMT_PAYMENT_TOTAL_CURRENT"], cc["AMT_INST_MIN_REGULARITY"])
    cc["LATE_PAYMENT"] = (cc["SK_DPD"] > 0).astype(int)
    cc["DRAWING_LIMIT_RATIO"] = safe_div(cc["AMT_DRAWINGS_ATM_CURRENT"], cc["AMT_CREDIT_LIMIT_ACTUAL"])

    cc_agg = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "AMT_BALANCE": ["mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["max", "mean"],
        "AMT_DRAWINGS_ATM_CURRENT": ["mean"],
        "AMT_DRAWINGS_CURRENT": ["mean"],
        "AMT_DRAWINGS_POS_CURRENT": ["mean"],
        "AMT_INST_MIN_REGULARITY": ["mean"],
        "AMT_PAYMENT_CURRENT": ["mean"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["mean"],
        "AMT_RECEIVABLE_PRINCIPAL": ["mean"],
        "AMT_TOTAL_RECEIVABLE": ["mean"],
        "CNT_DRAWINGS_ATM_CURRENT": ["mean"],
        "CNT_DRAWINGS_CURRENT": ["mean"],
        "CNT_DRAWINGS_POS_CURRENT": ["mean"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "LIMIT_USE": ["mean", "max"],
        "PAYMENT_DIV_MIN": ["mean"],
        "LATE_PAYMENT": ["mean"],
        "DRAWING_LIMIT_RATIO": ["mean"],
    }
    for c in cc_cat:
        cc_agg[c] = ["mean"]

    cc_grouped = agg_and_prefix(cc, "SK_ID_CURR", cc_agg, prefix="CC_")

    last_cc = cc.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    last_cc = (
        last_cc.groupby("SK_ID_CURR")
        .agg({"AMT_BALANCE": "mean", "AMT_DRAWINGS_CURRENT": "mean", "AMT_PAYMENT_TOTAL_CURRENT": "mean", "LIMIT_USE": "mean"})
        .reset_index()
    )
    last_cc = last_cc.rename(
        columns={
            "AMT_BALANCE": "CC_LAST_AMT_BALANCE",
            "AMT_DRAWINGS_CURRENT": "CC_LAST_DRAWING",
            "AMT_PAYMENT_TOTAL_CURRENT": "CC_LAST_PAYMENT",
            "LIMIT_USE": "CC_LAST_LIMIT_USE",
        }
    )
    cc_grouped = cc_grouped.merge(last_cc, on="SK_ID_CURR", how="left")

    for months in [12, 24, 48]:
        recent = cc[cc["MONTHS_BALANCE"] >= -months]
        if not recent.empty:
            agg = agg_and_prefix(recent, "SK_ID_CURR", cc_agg, prefix=f"CC_LAST{months}M_")
            cc_grouped = cc_grouped.merge(agg, on="SK_ID_CURR", how="left")

    return cc_grouped
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def add_ratios_features(df: pd.DataFrame) -> pd.DataFrame:
    df["BUREAU_INCOME_CREDIT_RATIO"] = safe_div(df.get("BUREAU_AMT_CREDIT_SUM_SUM"), df["AMT_INCOME_TOTAL"])
    df["BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO"] = safe_div(df.get("BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM"), df["AMT_INCOME_TOTAL"])

    for stat in ["MIN", "MAX", "MEAN"]:
        df[f"CURRENT_TO_APPROVED_CREDIT_{stat}_RATIO"] = safe_div(
            df.get("AMT_CREDIT"),
            df.get(f"PREVIOUS_APPROVED_AMT_CREDIT_{stat.lower()}") if f"PREVIOUS_APPROVED_AMT_CREDIT_{stat.lower()}" in df.columns else df.get(f"PREVIOUS_APPROVED_AMT_CREDIT_{stat.lower()}", np.nan),
        )
    for stat in ["MAX", "MEAN"]:
        df[f"CURRENT_TO_APPROVED_ANNUITY_{stat}_RATIO"] = safe_div(
            df.get("AMT_ANNUITY"),
            df.get(f"PREVIOUS_APPROVED_AMT_ANNUITY_{stat.lower()}") if f"PREVIOUS_APPROVED_AMT_ANNUITY_{stat.lower()}" in df.columns else df.get(f"PREVIOUS_APPROVED_AMT_ANNUITY_{stat.lower()}", np.nan),
        )

    for stat in ["MIN", "MAX", "MEAN"]:
        df[f"PAYMENT_{stat}_TO_ANNUITY_RATIO"] = safe_div(
            df.get(f"INS_AMT_PAYMENT_{stat.lower()}") if f"INS_AMT_PAYMENT_{stat.lower()}" in df.columns else df.get(f"INS_AMT_PAYMENT_{stat.lower()}", np.nan),
            df.get("AMT_ANNUITY"),
        )

    df["CTA_CREDIT_TO_ANNUITY_MAX_RATIO"] = safe_div(df.get("AMT_CREDIT"), df.get("AMT_ANNUITY"))
    df["CTA_CREDIT_TO_ANNUITY_MEAN_RATIO"] = safe_div(df.get("AMT_CREDIT"), df.get("AMT_ANNUITY"))

    df["DAYS_DECISION_MEAN_TO_BIRTH"] = safe_div(df.get("PREVIOUS_DAYS_DECISION_mean"), df.get("DAYS_BIRTH"))
    df["DAYS_CREDIT_MEAN_TO_BIRTH"] = safe_div(df.get("BUREAU_DAYS_CREDIT_MEAN"), df.get("DAYS_BIRTH"))
    df["DAYS_DECISION_MEAN_TO_EMPLOYED"] = safe_div(df.get("PREVIOUS_DAYS_DECISION_mean"), df.get("DAYS_EMPLOYED"))
    df["DAYS_CREDIT_MEAN_TO_EMPLOYED"] = safe_div(df.get("BUREAU_DAYS_CREDIT_MEAN"), df.get("DAYS_EMPLOYED"))
    return df
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def kfold_lightgbm_sklearn(df: pd.DataFrame, test_df: pd.DataFrame, categorical_feature=None):
    predictors = [c for c in df.columns if c not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index", "level_0"]]
    cat_features = categorical_feature or []

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(df.shape[0])
    predictions = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    params = {
        "objective": "binary",
        "boosting_type": "goss",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": 34,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    for fold, (train_idx, valid_idx) in enumerate(folds.split(df[predictors], df["TARGET"])):
        X_train, y_train = df[predictors].iloc[train_idx], df["TARGET"].iloc[train_idx]
        X_valid, y_valid = df[predictors].iloc[valid_idx], df["TARGET"].iloc[valid_idx]

        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features, free_raw_data=False)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            early_stopping_rounds=200,
            verbose_eval=200,
        )

        oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
        predictions += model.predict(test_df[predictors], num_iteration=model.best_iteration) / folds.n_splits

        fold_importance = pd.DataFrame(
            {
                "feature": predictors,
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
                "fold": fold + 1,
            }
        )
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
        print(f"Fold {fold + 1} AUC: {roc_auc_score(y_valid, oof[valid_idx]):.5f}")

    full_auc = roc_auc_score(df["TARGET"], oof)
    print(f"Full AUC: {full_auc:.5f}")

    if GENERATE_SUBMISSION_FILES:
        submission = pd.DataFrame({"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": predictions})
        oof_df = pd.DataFrame({"SK_ID_CURR": df["SK_ID_CURR"], "TARGET": df["TARGET"], "OOF_PRED": oof})
        fi_mean = feature_importance_df.groupby("feature")["importance_gain"].mean().reset_index().sort_values(by="importance_gain", ascending=False)
        submission.to_csv("submission.csv", index=False)
        feature_importance_df.to_csv("feature_importance.csv", index=False)
        oof_df.to_csv("oof.csv", index=False)
        fi_mean.to_csv("feature_importance_mean.csv", index=False)
    return oof, predictions, feature_importance_df
"""
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(
                """

def main():
    lgbm_categorical_feat = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "NAME_CONTRACT_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "NAME_INCOME_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "NAME_TYPE_SUITE",
        "WALLSMATERIAL_MODE",
    ]

    with timer("Full pipeline"):
        with timer("Load application"):
            df = get_train_test(DATA_PATH)
        with timer("Bureau"):
            bureau = get_bureau(DATA_PATH)
            df = df.merge(bureau, on="SK_ID_CURR", how="left")
        with timer("Previous applications"):
            prev = get_previous_applications(DATA_PATH)
            df = df.merge(prev, on="SK_ID_CURR", how="left")
        with timer("POS_CASH"):
            pos = get_pos_cash(DATA_PATH)
            df = df.merge(pos, on="SK_ID_CURR", how="left")
        with timer("Installments"):
            ins = get_installment_payments(DATA_PATH)
            df = df.merge(ins, on="SK_ID_CURR", how="left")
        with timer("Credit card"):
            cc = get_credit_card(DATA_PATH)
            df = df.merge(cc, on="SK_ID_CURR", how="left")

        with timer("Ratios"):
            df = add_ratios_features(df)

        with timer("Reduce memory"):
            df = reduce_memory(df)

        train_df = df[df["TARGET"].notnull()].copy()
        test_df = df[df["TARGET"].isnull()].copy()
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        with timer("LightGBM CV"):
            kfold_lightgbm_sklearn(train_df, test_df, categorical_feature=lgbm_categorical_feat)


if __name__ == "__main__":
    main()
"""
            ),
        }
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = Path("notebooks/00_home_credit_full_pipeline.ipynb")
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written {out_path} with {len(cells)} cells")


if __name__ == "__main__":
    main()


