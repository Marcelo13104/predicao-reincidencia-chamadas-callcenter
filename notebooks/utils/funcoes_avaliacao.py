# Importações
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import os 
import joblib


# --- Função para avaliação ---
def avaliar_modelo(nome, y_true, y_pred, y_prob):
     # Calcula métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred)

    print(f"\n🔍 {nome}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:\n", report)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "classification_report": report
    }

# --- Função para avaliação ---
def avaliar_modelo_pkl(nome, y_true, y_pred, y_prob):
     # Calcula métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred)

    print(f"\n🔍 {nome}")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "classification_report": report
    }


def plot_matriz_confusao(y_true, y_pred, nome):
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm)
    cm_perc = cm / cm_sum * 100
    
    # Criar labels com valor absoluto e percentual
    labels = np.array([f"{v}\n{p:.1f}%" for v, p in zip(cm.flatten(), cm_perc.flatten())])
    labels = labels.reshape(cm.shape)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.show()

def plot_distribuicao_probabilidades(nome,y_prob):
    plt.figure(figsize=(10, 6))
    sns.histplot(y_prob, bins=50, kde=True, color="skyblue")
    plt.title(f"Distribuição das probabilidades previstas para a classe 1  - {nome}")
    plt.xlabel("Probabilidade prevista (predict_proba)")
    plt.ylabel("Frequência")
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.legend()
    plt.grid(True)
    plt.show()

def avaliar_threshold(y_true, y_prob, threshold=0.5):
    y_pred_thresh = (y_prob >= threshold).astype(int)
    
    print(f"\n🔧 Threshold = {threshold:.2f}")
    print("Precision:", precision_score(y_true, y_pred_thresh))
    print("Recall:", recall_score(y_true, y_pred_thresh))
    print("F1 Score:", f1_score(y_true, y_pred_thresh))
    print("AUC:", roc_auc_score(y_true, y_prob))

    plot_matriz_confusao(y_true, y_pred_thresh, f'Threshold {threshold:.2f}')