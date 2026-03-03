"""
Module d'évaluation et d'optimisation des modèles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import optuna
from optuna.samplers import TPESampler
import mlflow


def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Trace la courbe ROC.
    
    Args:
        y_true: Vraies étiquettes
        y_pred_proba: Probabilités prédites
        model_name (str): Nom du modèle
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Trace la courbe Precision-Recall.
    
    Args:
        y_true: Vraies étiquettes
        y_pred_proba: Probabilités prédites
        model_name (str): Nom du modèle
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Courbe Precision-Recall - {model_name}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def find_optimal_threshold(model, X_test, y_test, cost_fn=10, cost_fp=1):
    """
    Trouve le seuil optimal basé sur le coût métier.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Cible de test
        cost_fn (float): Coût d'un faux négatif
        cost_fp (float): Coût d'un faux positif
        
    Returns:
        tuple: (seuil optimal, coût minimal, résultats)
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    costs = []
    recalls = []
    precisions = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calcul des FN et FP
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        
        cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(cost)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls.append(recall)
        precisions.append(precision)
    
    # Trouver le seuil optimal
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique du coût
    ax1.plot(thresholds, costs, 'b-', linewidth=2)
    ax1.axvline(optimal_threshold, color='r', linestyle='--', 
                label=f'Seuil optimal = {optimal_threshold:.2f}')
    ax1.axhline(min_cost, color='g', linestyle='--', alpha=0.5,
                label=f'Coût minimal = {min_cost:.2f}')
    ax1.set_xlabel('Seuil de décision')
    ax1.set_ylabel('Coût métier total')
    ax1.set_title('Coût métier vs Seuil')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Graphique Recall et Precision
    ax2.plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
    ax2.plot(thresholds, precisions, 'g-', label='Precision', linewidth=2)
    ax2.axvline(optimal_threshold, color='r', linestyle='--', 
                label=f'Seuil optimal = {optimal_threshold:.2f}')
    ax2.set_xlabel('Seuil de décision')
    ax2.set_ylabel('Score')
    ax2.set_title('Recall et Precision vs Seuil')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    results = pd.DataFrame({
        'Threshold': thresholds,
        'Cost': costs,
        'Recall': recalls,
        'Precision': precisions
    })
    
    print(f"\n✓ Seuil optimal trouvé : {optimal_threshold:.2f}")
    print(f"  Coût métier minimal : {min_cost:.2f}")
    print(f"  Recall : {recalls[optimal_idx]:.4f}")
    print(f"  Precision : {precisions[optimal_idx]:.4f}")
    
    return optimal_threshold, min_cost, results


def grid_search_optimization(model, param_grid, X_train, y_train, cv=5):
    """
    Optimise les hyperparamètres avec GridSearchCV.
    
    Args:
        model: Modèle à optimiser
        param_grid (dict): Grille de paramètres
        X_train, y_train: Données d'entraînement
        cv (int): Nombre de folds
        
    Returns:
        GridSearchCV: Objet GridSearchCV ajusté
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n{'='*60}")
    print("Optimisation GridSearchCV en cours...")
    print(f"{'='*60}")
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Optimisation terminée")
    print(f"  Meilleur score (AUC-ROC) : {grid_search.best_score_:.4f}")
    print(f"  Meilleurs paramètres : {grid_search.best_params_}")
    
    return grid_search


def optuna_optimization(model_class, X_train, y_train, X_test, y_test, 
                        n_trials=50, cost_fn=10, cost_fp=1):
    """
    Optimise les hyperparamètres avec Optuna.
    
    Args:
        model_class: Classe du modèle
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        n_trials (int): Nombre d'essais
        cost_fn (float): Coût d'un faux négatif
        cost_fp (float): Coût d'un faux positif
        
    Returns:
        dict: Meilleurs paramètres trouvés
    """
    def objective(trial):
        # Définir l'espace de recherche selon le modèle
        if model_class.__name__ == 'RandomForestClassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        elif model_class.__name__ == 'XGBClassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 10),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
        elif model_class.__name__ == 'LGBMClassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        else:
            return 0
        
        # Entraîner le modèle
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        # Prédire
        y_pred = model.predict(X_test)
        
        # Calculer le coût métier (à minimiser)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = (fn * cost_fn) + (fp * cost_fp)
        
        return cost
    
    print(f"\n{'='*60}")
    print(f"Optimisation Optuna - {model_class.__name__}")
    print(f"{'='*60}")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✓ Optimisation terminée")
    print(f"  Meilleur coût métier : {study.best_value:.2f}")
    print(f"  Meilleurs paramètres : {study.best_params}")
    
    return study.best_params


def get_param_grids():
    """
    Retourne les grilles de paramètres pour GridSearchCV.
    
    Returns:
        dict: Grilles de paramètres par modèle
    """
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'scale_pos_weight': [1, 3, 5]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50, 70]
        },
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    }
    return param_grids
