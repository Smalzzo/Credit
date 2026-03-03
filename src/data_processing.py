"""
Module de traitement et préparation des données pour le scoring de crédit.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Charge un fichier de données.
    
    Args:
        file_path (str): Chemin vers le fichier
        
    Returns:
        pd.DataFrame: Données chargées
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Format de fichier non supporté. Utiliser .csv ou .xlsx")
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return None


def explore_data(df, name="Dataset"):
    """
    Affiche un résumé exploratoire du dataset.
    
    Args:
        df (pd.DataFrame): Dataset à explorer
        name (str): Nom du dataset
    """
    print(f"\n{'='*60}")
    print(f"Exploration : {name}")
    print(f"{'='*60}")
    print(f"\nDimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"\nTypes de colonnes :")
    print(df.dtypes.value_counts())
    print(f"\nValeurs manquantes :")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Colonnes': missing.index,
            'Manquants': missing.values,
            'Pourcentage': missing_pct.values
        })
        print(missing_df[missing_df['Manquants'] > 0].sort_values('Manquants', ascending=False))
    else:
        print("Aucune valeur manquante")
    print(f"\nDoublons : {df.duplicated().sum()}")


def check_duplicates(df):
    """
    Vérifie et supprime les doublons.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset sans doublons
    """
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        print(f"⚠️  {n_duplicates} doublons trouvés et supprimés")
        return df.drop_duplicates()
    print("✓ Aucun doublon trouvé")
    return df


def merge_datasets(df_list, keys, how='inner'):
    """
    Fusionne plusieurs datasets.
    
    Args:
        df_list (list): Liste de DataFrames
        keys (list): Liste des clés de fusion
        how (str): Type de jointure
        
    Returns:
        pd.DataFrame: Dataset fusionné
    """
    if len(df_list) < 2:
        return df_list[0] if df_list else None
    
    result = df_list[0]
    for i, df in enumerate(df_list[1:], 1):
        before = len(result)
        result = result.merge(df, on=keys[i-1] if isinstance(keys, list) else keys, how=how)
        after = len(result)
        print(f"Fusion {i}: {before} lignes → {after} lignes")
    
    return result


def handle_missing_values(df, strategy='auto', threshold=0.5):
    """
    Gère les valeurs manquantes avec différentes stratégies.
    
    Args:
        df (pd.DataFrame): Dataset
        strategy (str): Stratégie ('auto', 'drop', 'mean', 'median', 'mode')
        threshold (float): Seuil de suppression de colonnes (% de valeurs manquantes)
        
    Returns:
        pd.DataFrame: Dataset avec valeurs manquantes traitées
    """
    df_clean = df.copy()
    
    # Colonnes avec trop de valeurs manquantes
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"⚠️  Suppression de {len(cols_to_drop)} colonnes avec >{threshold*100}% de valeurs manquantes:")
        print(cols_to_drop)
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Imputation pour les colonnes restantes
    if strategy == 'auto':
        # Colonnes numériques : médiane
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer_num = SimpleImputer(strategy='median')
            df_clean[numeric_cols] = imputer_num.fit_transform(df_clean[numeric_cols])
        
        # Colonnes catégorielles : mode
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])
    
    print(f"✓ Valeurs manquantes traitées. Colonnes restantes : {df_clean.shape[1]}")
    return df_clean


def encode_categorical(df, method='label', target_col=None):
    """
    Encode les variables catégorielles.
    
    Args:
        df (pd.DataFrame): Dataset
        method (str): Méthode d'encodage ('label', 'onehot')
        target_col (str): Colonne cible à ne pas encoder
        
    Returns:
        pd.DataFrame: Dataset avec variables encodées
    """
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)
    
    if method == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        print(f"✓ {len(cat_cols)} colonnes encodées (Label Encoding)")
    
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
        print(f"✓ {len(cat_cols)} colonnes encodées (One-Hot Encoding)")
    
    return df_encoded


def create_features(df):
    """
    Crée de nouvelles features à partir des variables existantes.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset avec nouvelles features
    """
    df_features = df.copy()
    
    # Exemple : ajouter des features de date si présentes
    date_cols = df_features.select_dtypes(include=['datetime64']).columns
    for col in date_cols:
        df_features[f'{col}_year'] = df_features[col].dt.year
        df_features[f'{col}_month'] = df_features[col].dt.month
        df_features[f'{col}_dayofweek'] = df_features[col].dt.dayofweek
    
    print(f"✓ Features créées. Nouvelles dimensions : {df_features.shape}")
    return df_features


def scale_features(df, target_col=None):
    """
    Normalise les features numériques.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Colonne cible à ne pas normaliser
        
    Returns:
        pd.DataFrame: Dataset avec features normalisées
        StandardScaler: Scaler ajusté
    """
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        print(f"✓ {len(numeric_cols)} features numériques normalisées")
        return df_scaled, scaler
    
    return df_scaled, None


def save_processed_data(df, file_path):
    """
    Sauvegarde les données préparées.
    
    Args:
        df (pd.DataFrame): Dataset à sauvegarder
        file_path (str): Chemin de sauvegarde
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"✓ Données sauvegardées : {file_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")
