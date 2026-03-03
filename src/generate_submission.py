"""Generate a Kaggle-ready submission.csv for Home Credit Default Risk.

This script is designed to run locally (Windows) or on Kaggle.
It reads raw CSVs, builds a feature table at the applicant level (SK_ID_CURR),
trains a LightGBM model, and writes outputs/submission.csv.

Default paths assume this repository layout:
- input:  data/raw/
- output: outputs/

If auxiliary tables (bureau, previous_application, etc.) are missing,
features will be built from application_{train,test}.csv only.

Usage (VS Code, no terminal):
- Open this file and click "Run Python File".

Usage (terminal):
- python -m src.generate_submission --input-dir data/raw --output outputs/submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"


def _safe_read_csv(input_dir: Path, filename: str, nrows: int | None) -> pd.DataFrame | None:
    path = input_dir / filename
    if not path.exists():
        print(f"[WARN] Missing optional file: {path}")
        return None
    return pd.read_csv(path, nrows=nrows)


def _one_hot_low_card_else_factorize(
    df: pd.DataFrame,
    cols: Iterable[str],
    max_ohe: int = 20,
    prefix: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= max_ohe:
            dummies = pd.get_dummies(df[col].astype("string"), prefix=(prefix or col), dummy_na=True)
            df = df.drop(columns=[col]).join(dummies)
        else:
            codes, _ = pd.factorize(df[col].astype("string").fillna("__NA__"), sort=False)
            df[col] = codes.astype("int32")
    return df


def _aggregate_numeric(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != key]
    if not numeric_cols:
        out = df.groupby(key).size().to_frame(f"{prefix}_COUNT")
        return out

    agg = df.groupby(key)[numeric_cols].agg(["mean", "std", "min", "max", "sum"])
    agg.columns = pd.Index([f"{prefix}_{c}_{stat.upper()}" for c, stat in agg.columns.to_list()])
    agg[f"{prefix}_COUNT"] = df.groupby(key).size()
    return agg


def build_home_credit_dataset(input_dir: str | Path, debug: bool = True) -> pd.DataFrame:
    """Builds an applicant-level dataset with TARGET for train rows.

    Expects at least:
    - application_train.csv
    - application_test.csv

    Optionally uses:
    - bureau.csv, bureau_balance.csv
    - previous_application.csv
    - POS_CASH_balance.csv
    - installments_payments.csv
    - credit_card_balance.csv

    Returns a DataFrame containing both train+test rows.
    """

    input_dir = Path(input_dir)
    nrows = 10000 if debug else None

    train = _safe_read_csv(input_dir, "application_train.csv", nrows)
    test = _safe_read_csv(input_dir, "application_test.csv", nrows)

    if train is None or test is None:
        raise FileNotFoundError(
            "Missing application_train.csv or application_test.csv. "
            f"Expected them in: {input_dir.resolve()}"
        )

    df = pd.concat([train, test], axis=0, ignore_index=True)

    # Basic cleaning / standard Home Credit fixes
    if "CODE_GENDER" in df.columns:
        df = df[df["CODE_GENDER"].astype(str) != "XNA"]

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Binary factorize some common columns (if present)
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if col in df.columns:
            df[col], _ = pd.factorize(df[col].astype("string"), sort=False)
            df[col] = df[col].astype("int32")

    # Controlled encoding for remaining categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = _one_hot_low_card_else_factorize(df, cat_cols, max_ohe=20)

    # Simple ratios (only if columns exist)
    def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        out = a / b.replace(0, np.nan)
        return out

    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df.columns):
        df["INCOME_CREDIT_PERC"] = safe_div(df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"]).astype("float32")
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_PERC"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"]).astype("float32")
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["PAYMENT_RATE"] = safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"]).astype("float32")

    # ---------- Optional auxiliary tables ----------

    # bureau + bureau_balance
    bureau = _safe_read_csv(input_dir, "bureau.csv", nrows)
    bb = _safe_read_csv(input_dir, "bureau_balance.csv", nrows)
    if bureau is not None:
        if bb is not None and "SK_ID_BUREAU" in bb.columns:
            bb_cat = bb.select_dtypes(include=["object", "category"]).columns.tolist()
            bb = _one_hot_low_card_else_factorize(bb, bb_cat, max_ohe=20, prefix="BB")
            bb_agg = _aggregate_numeric(bb, key="SK_ID_BUREAU", prefix="BB")
            bureau = bureau.merge(bb_agg, left_on="SK_ID_BUREAU", right_index=True, how="left")

        buro_cat = bureau.select_dtypes(include=["object", "category"]).columns.tolist()
        bureau = _one_hot_low_card_else_factorize(bureau, buro_cat, max_ohe=20, prefix="BURO")
        buro_agg = _aggregate_numeric(bureau.drop(columns=["SK_ID_BUREAU"], errors="ignore"), key=ID_COL, prefix="BURO")
        df = df.merge(buro_agg, left_on=ID_COL, right_index=True, how="left")

    # previous_application
    prev = _safe_read_csv(input_dir, "previous_application.csv", nrows)
    if prev is not None and ID_COL in prev.columns:
        prev_cat = prev.select_dtypes(include=["object", "category"]).columns.tolist()
        prev = _one_hot_low_card_else_factorize(prev, prev_cat, max_ohe=20, prefix="PREV")
        prev_agg = _aggregate_numeric(prev.drop(columns=["SK_ID_PREV"], errors="ignore"), key=ID_COL, prefix="PREV")
        df = df.merge(prev_agg, left_on=ID_COL, right_index=True, how="left")

    # POS_CASH_balance
    pos = _safe_read_csv(input_dir, "POS_CASH_balance.csv", nrows)
    if pos is not None and ID_COL in pos.columns:
        pos_cat = pos.select_dtypes(include=["object", "category"]).columns.tolist()
        pos = _one_hot_low_card_else_factorize(pos, pos_cat, max_ohe=20, prefix="POS")
        pos_agg = _aggregate_numeric(pos, key=ID_COL, prefix="POS")
        df = df.merge(pos_agg, left_on=ID_COL, right_index=True, how="left")

    # installments_payments
    ins = _safe_read_csv(input_dir, "installments_payments.csv", nrows)
    if ins is not None and ID_COL in ins.columns:
        # basic engineered columns
        if {"AMT_INSTALMENT", "AMT_PAYMENT"}.issubset(ins.columns):
            ins["PAYMENT_PERC"] = safe_div(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"]).astype("float32")
            ins["PAYMENT_DIFF"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).astype("float32")
        if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(ins.columns):
            ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
            ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

        ins_cat = ins.select_dtypes(include=["object", "category"]).columns.tolist()
        ins = _one_hot_low_card_else_factorize(ins, ins_cat, max_ohe=20, prefix="INS")
        ins_agg = _aggregate_numeric(ins, key=ID_COL, prefix="INS")
        df = df.merge(ins_agg, left_on=ID_COL, right_index=True, how="left")

    # credit_card_balance
    cc = _safe_read_csv(input_dir, "credit_card_balance.csv", nrows)
    if cc is not None and ID_COL in cc.columns:
        cc_cat = cc.select_dtypes(include=["object", "category"]).columns.tolist()
        cc = _one_hot_low_card_else_factorize(cc, cc_cat, max_ohe=20, prefix="CC")
        cc_agg = _aggregate_numeric(cc.drop(columns=["SK_ID_PREV"], errors="ignore"), key=ID_COL, prefix="CC")
        df = df.merge(cc_agg, left_on=ID_COL, right_index=True, how="left")

    # Final cleanup
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def train_lightgbm_and_predict(df_all: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Trains a LightGBM model and returns a submission dataframe."""

    if TARGET_COL not in df_all.columns:
        raise ValueError("TARGET column not found in dataset")

    train_df = df_all[df_all[TARGET_COL].notnull()].copy()
    test_df = df_all[df_all[TARGET_COL].isnull()].copy()

    feats = [c for c in df_all.columns if c not in {TARGET_COL, ID_COL}]

    X = train_df[feats]
    y = train_df[TARGET_COL].astype(int)
    X_test = test_df[feats]

    # Impute with training medians
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    # Ensure numeric dtypes
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Scale_pos_weight for imbalance
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=34,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[],
    )

    proba = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: proba})
    return submission


def generate_submission(
    input_dir: str | Path = "data/raw",
    output_path: str | Path = "outputs/submission.csv",
    debug: bool = True,
) -> Path:
    """High-level helper: build dataset, train, and write submission CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_all = build_home_credit_dataset(input_dir=input_dir, debug=debug)
    submission = train_lightgbm_and_predict(df_all)

    submission.to_csv(output_path, index=False)
    print(f"✓ Wrote submission: {output_path.resolve()}")
    return output_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="data/raw", help="Folder containing Home Credit CSVs")
    p.add_argument("--output", type=str, default="outputs/submission.csv", help="Submission CSV output path")
    p.add_argument("--debug", action="store_true", help="Use a small subset of rows (fast smoke test)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    generate_submission(input_dir=args.input_dir, output_path=args.output, debug=args.debug)


if __name__ == "__main__":
    main()
