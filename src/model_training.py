"""
Module d'entraînement des modèles de scoring de crédit.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, 
    precision_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


def check_class_imbalance(y):
    """
    Vérifie le déséquilibre des classes.
    
    Args:
        y (pd.Series): Variable cible
        
    Returns:
        dict: Statistiques sur la distribution des classes
    """
    value_counts = y.value_counts()
    proportions = y.value_counts(normalize=True) * 100
    
    print("\n" + "="*50)
    print("Distribution des classes cibles")
    print("="*50)
    for cls in value_counts.index:
        print(f"Classe {cls}: {value_counts[cls]} ({proportions[cls]:.2f}%)")
    
    imbalance_ratio = value_counts.max() / value_counts.min()
    print(f"\nRatio de déséquilibre : {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2:
        print("⚠️  Dataset déséquilibré - Utiliser SMOTE ou class_weight")
    
    return {
        'counts': value_counts.to_dict(),
        'proportions': proportions.to_dict(),
        'imbalance_ratio': imbalance_ratio
    }


def apply_smote(X_train, y_train, random_state=42):
    """
    Applique SMOTE pour équilibrer les classes.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        random_state: Seed aléatoire
        
    Returns:
        X_resampled, y_resampled: Données rééchantillonnées
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"✓ SMOTE appliqué : {len(y_train)} → {len(y_resampled)} échantillons")
    return X_resampled, y_resampled


def get_models(use_class_weight=True):
    """
    Retourne un dictionnaire de modèles à tester.
    
    Args:
        use_class_weight (bool): Utiliser la pondération des classes
        
    Returns:
        dict: Modèles configurés
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced' if use_class_weight else None,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced' if use_class_weight else None,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            scale_pos_weight=3 if use_class_weight else 1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            class_weight='balanced' if use_class_weight else None,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
    }
    return models


def train_with_cv(model, X, y, cv=5, scoring=None):
    """
    Entraîne un modèle avec validation croisée stratifiée.
    
    Args:
        model: Modèle à entraîner
        X: Features
        y: Cible
        cv (int): Nombre de folds
        scoring (list): Métriques à calculer
        
    Returns:
        dict: Résultats de la validation croisée
    """
    if scoring is None:
        scoring = ['roc_auc', 'recall', 'precision', 'f1']
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    return cv_results


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Évalue un modèle sur le jeu de test.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Cible de test
        threshold (float): Seuil de décision
        
    Returns:
        dict: Métriques d'évaluation
    """
    # Prédictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Métriques
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
    
    return metrics


def calculate_business_cost(y_true, y_pred, cost_fn=10, cost_fp=1):
    """
    Calcule le coût métier total.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        cost_fn (float): Coût d'un faux négatif
        cost_fp (float): Coût d'un faux positif
        
    Returns:
        float: Coût total
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return total_cost


def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test, 
                        experiment_name="Credit_Scoring"):
    """
    Entraîne un modèle et log les résultats dans MLflow.
    
    Args:
        model_name (str): Nom du modèle
        model: Modèle à entraîner
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        experiment_name (str): Nom de l'expérience MLflow
        
    Returns:
        model: Modèle entraîné
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name):
        # Entraînement
        model.fit(X_train, y_train)
        
        # Évaluation
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log des paramètres
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
        
        # Log des métriques
        mlflow.log_metric("auc_roc", metrics['auc_roc'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("true_positives", metrics['tp'])
        mlflow.log_metric("false_positives", metrics['fp'])
        mlflow.log_metric("true_negatives", metrics['tn'])
        mlflow.log_metric("false_negatives", metrics['fn'])
        
        # Coût métier
        y_pred = model.predict(X_test)
        business_cost = calculate_business_cost(y_test, y_pred)
        mlflow.log_metric("business_cost", business_cost)
        
        # Log du modèle
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"\n✓ {model_name} entraîné et loggé dans MLflow")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Coût métier: {business_cost:.2f}")
    
    return model


def compare_models(models_dict, X_train, y_train, X_test, y_test):
    """
    Compare plusieurs modèles et retourne un résumé.
    
    Args:
        models_dict (dict): Dictionnaire de modèles
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        
    Returns:
        pd.DataFrame: Tableau comparatif des performances
    """
    results = []
    
    for name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Entraînement : {name}")
        print(f"{'='*60}")
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Évaluation
        metrics = evaluate_model(model, X_test, y_test)
        y_pred = model.predict(X_test)
        cost = calculate_business_cost(y_test, y_pred)
        
        results.append({
            'Modèle': name,
            'AUC-ROC': metrics['auc_roc'],
            'Recall': metrics['recall'],
            'Precision': metrics['precision'],
            'F1-Score': metrics['f1_score'],
            'Coût métier': cost
        })
        
        print(f"✓ Terminé")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('AUC-ROC', ascending=False)
    
    return df_results
