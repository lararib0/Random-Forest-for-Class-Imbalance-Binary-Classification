import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, wilcoxon
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from collections import defaultdict


def preprocess_datasets(df, target_column=None):

    df.columns = df.columns.str.strip()

    if target_column is None:
        target_column = df.columns[-1]

    df = df.dropna(subset=[target_column])
    y = df[target_column]
    X = df.drop(columns=[target_column])

    X = pd.get_dummies(X, drop_first=True)

    if X.isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(X)
    else:
        X = X.to_numpy()

    return X, y, target_column



def encode_labels(y):
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        target_names = list(le.classes_)
    else:
        y_encoded = y.to_numpy()
        unique_vals = np.unique(y_encoded)
        if set(unique_vals) == {-1, 1}:
            y_encoded = (y_encoded == 1).astype(int)
            target_names = ['-1', '1']
        elif np.any(y_encoded < 0):
            raise ValueError(f"Rótulos com valores negativos inesperados: {set(unique_vals)}")
        else:
            target_names = [str(c) for c in sorted(unique_vals)]

    return y_encoded, target_names


def identify_classes(y):
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)

    return minority_class, majority_class, class_counts



def calculate_fold_metrics(y_test, y_pred, y_proba):
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    return report, auc


def aggregate_metrics(metrics_sum, aucs, n_folds, minority_class, majority_class):
    def avg(metric, label):
        return metrics_sum[str(label)][metric] / n_folds

    result_dict = {
        'dataset': None,
        'f1_minority': avg('f1-score', minority_class),
        'f1_majority': avg('f1-score', majority_class),
        'precision_minority': avg('precision', minority_class),
        'precision_majority': avg('precision', majority_class),
        'recall_minority': avg('recall', minority_class),
        'recall_majority': avg('recall', majority_class),
        'accuracy': metrics_sum['accuracy']['accuracy'] / n_folds,
        'auc_roc': np.mean(aucs)
    }

    return result_dict



def calculate_class_metrics(rf_file, rf_weighted_file):
    rf = pd.read_csv(rf_file)
    rf_weighted = pd.read_csv(rf_weighted_file)

    metrics_class = [
        'f1_minority', 'f1_majority',
        'recall_minority', 'recall_majority',
        'precision_minority', 'precision_majority'
    ]

    rf_means_class = rf[metrics_class].mean()
    rf_weighted_means_class = rf_weighted[metrics_class].mean()

    tabela_class = pd.DataFrame({
        'Random Forest': rf_means_class,
        'Random Forest Weighted': rf_weighted_means_class
    })

    return tabela_class



def media_duas_colunas(df, metrica1, metrica2):
    return (df[metrica1] + df[metrica2]) / 2


def calculate_global_metrics(rf_file, rf_weighted_file):

    rf = pd.read_csv(rf_file)
    rf_weighted = pd.read_csv(rf_weighted_file)


    rf_f1 = media_duas_colunas(rf, 'f1_minority', 'f1_majority').mean()
    rf_recall = media_duas_colunas(rf, 'recall_minority', 'recall_majority').mean()
    rf_precision = media_duas_colunas(rf, 'precision_minority', 'precision_majority').mean()
    rf_accuracy = rf['accuracy'].mean()
    rf_auc = rf['auc_roc'].mean()


    rf_weighted_f1 = media_duas_colunas(rf_weighted, 'f1_minority', 'f1_majority').mean()
    rf_weighted_recall = media_duas_colunas(rf_weighted, 'recall_minority', 'recall_majority').mean()
    rf_weighted_precision = media_duas_colunas(rf_weighted, 'precision_minority', 'precision_majority').mean()
    rf_weighted_accuracy = rf_weighted['accuracy'].mean()
    rf_weighted_auc = rf_weighted['auc_roc'].mean()


    tabela_geral = pd.DataFrame({
        'Random Forest': [rf_f1, rf_recall, rf_precision, rf_accuracy, rf_auc],
        'Random Forest Weighted': [rf_weighted_f1, rf_weighted_recall, rf_weighted_precision, rf_weighted_accuracy,
                                   rf_weighted_auc]
    }, index=['F1', 'Recall', 'Precision', 'Accuracy', 'AUC-ROC'])

    return tabela_geral


def display_comparison_tables(rf_file, rf_weighted_file):
    tabela_class = calculate_class_metrics(rf_file, rf_weighted_file)
    tabela_geral = calculate_global_metrics(rf_file, rf_weighted_file)

    return tabela_class, tabela_geral


def prepare_comparison_data(rf_file, rf_weighted_file):

    rf_df = pd.read_csv(rf_file)
    rf_melhorado_df = pd.read_csv(rf_weighted_file)


    rf_df["f1_score"] = (rf_df["f1_minority"] + rf_df["f1_majority"]) / 2
    rf_melhorado_df["f1_score"] = (rf_melhorado_df["f1_minority"] + rf_melhorado_df["f1_majority"]) / 2

    f1_data = pd.DataFrame({
        "F1-score": pd.concat([rf_df["f1_score"], rf_melhorado_df["f1_score"]], ignore_index=True),
        "Modelo": ["Random Forest"] * len(rf_df) + ["Random Forest Modificado"] * len(rf_melhorado_df)
    })

    auc_data = pd.DataFrame({
        "AUC ROC": pd.concat([rf_df["auc_roc"], rf_melhorado_df["auc_roc"]], ignore_index=True),
        "Modelo": ["Random Forest"] * len(rf_df) + ["Random Forest Modificado"] * len(rf_melhorado_df)
    })

    return f1_data, auc_data


def plot_comparison_boxplots(rf_file, rf_weighted_file, figsize=(12, 5), palette=None):

    f1_data, auc_data = prepare_comparison_data(rf_file, rf_weighted_file)


    sns.set(style="whitegrid", font_scale=1.0)
    plt.rcParams['figure.facecolor'] = 'white'

    if palette is None:
        palette = ["#FF9AA2", "#A2E1FF"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sns.boxplot(data=f1_data, x="Modelo", y="F1-score", ax=ax1, width=0.4, palette=palette)
    ax1.set_title("F1-score comparison\nbetween models", pad=15, fontweight='bold', fontsize=12)
    ax1.set_xlabel("", labelpad=10)
    ax1.set_ylabel("F1-score médio", labelpad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)


    sns.boxplot(data=auc_data, x="Modelo", y="AUC ROC", ax=ax2, width=0.4, palette=palette)
    ax2.set_title("AUC ROC comparison\nbetween models", pad=15, fontweight='bold', fontsize=12)
    ax2.set_xlabel("", labelpad=10)
    ax2.set_ylabel("AUC ROC", labelpad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)


    plt.tight_layout(pad=2.0)

    return fig


def verificar_normalidade(df1, df2, metricas, nome_modelo1="Modelo 1", nome_modelo2="Modelo 2"):

    assert all(df1["dataset"] == df2["dataset"]), "Os datasets não estão alinhados corretamente"

    normalidade_resultados = {}

    for metrica in metricas:
        stat_norm, p_norm = shapiro(df1[metrica])
        stat_bal, p_bal = shapiro(df2[metrica])
        normalidade_resultados[metrica] = {
            nome_modelo1: p_norm > 0.05,
            "p-value": p_norm,
            nome_modelo2: p_bal > 0.05,
            "p_val_modelo2": p_bal
        }

    return pd.DataFrame(normalidade_resultados).T


def realizar_testes_estatisticos(df1, df2, metricas):

    resultados_testes = []

    for metrica in metricas:
        x = df1[metrica]
        y = df2[metrica]

        stat, p = wilcoxon(x, y)
        mean_diff = (y - x).mean()

        resultados_testes.append({
            "metric": metrica,
            "statistic": stat,
            "raw_p_value": p,
            "mean_difference": mean_diff
        })

    return pd.DataFrame(resultados_testes)



