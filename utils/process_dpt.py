import sys
sys.path.append('./')
import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns


def RandomForest_classify(X, y, n_splits=5, random_state=42):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    all_y_true = []
    all_y_pred = []

    # 用于累积特征重要性（可选：取平均）
    feature_importances = np.zeros(X.shape[1])

    for fold, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results.append(accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(preds)

        # 累加特征重要性
        feature_importances += model.feature_importances_

        print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, preds))

    avg_importance = feature_importances / n_splits
    print(f"Random Forest Average Accuracy: {np.mean(results):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'AML'],
                yticklabels=['Healthy', 'AML'])
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('/mnt/sda/ljw/Code/AML/LMAResult/random_forest.png')
    plt.show()

    # 特征重要性图（保持原始顺序）
    plt.figure(figsize=(14, 6))
    plt.plot(avg_importance, color='tab:blue')
    plt.fill_between(range(len(avg_importance)), avg_importance, alpha=0.3, color='tab:blue')
    plt.title("Random Forest Feature Importance (by Wavelength Index)")
    plt.xlabel("Wavelength Index (0–1865)")
    plt.ylabel("Mean Feature Importance")
    plt.tight_layout()
    plt.savefig('/mnt/sda/ljw/Code/AML/LMAResult/rf_feature_importance.png')
    
    return 0


def Xgboost_classify(X, y, n_splits=5):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42
    }

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    all_y_true = []
    all_y_pred = []

    # 累积特征重要性（使用 'gain' 更稳定）
    total_importance = np.zeros(X.shape[1])

    for fold, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(params, dtrain, num_boost_round=30, verbose_eval=False)

        preds = model.predict(dtest)
        preds_binary = (preds > 0.5).astype(int)

        accuracy = accuracy_score(y_test, preds_binary)
        results.append(accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(preds_binary)

        # 获取特征重要性（按 'gain'）
        importance_dict = model.get_score(importance_type='gain')
        # 初始化当前 fold 的重要性向量
        imp_vec = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            key = f'f{i}'
            if key in importance_dict:
                imp_vec[i] = importance_dict[key]
        total_importance += imp_vec

        print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, preds_binary))

    avg_importance = total_importance / n_splits
    print(f"XGBoost Average Accuracy: {np.mean(results):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'AML'],
                yticklabels=['Healthy', 'AML'])
    plt.title('XGBoost Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('/mnt/sda/ljw/Code/AML/LMAResult/xgboost.png')
    plt.show()

    # 特征重要性图（保持原始顺序）
    plt.figure(figsize=(14, 6))
    plt.plot(avg_importance, color='tab:orange')
    plt.fill_between(range(len(avg_importance)), avg_importance, alpha=0.3, color='tab:orange')
    plt.title("XGBoost Feature Importance (by Wavelength Index, gain)")
    plt.xlabel("Wavelength Index (0–1865)")
    plt.ylabel("Mean Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig('/mnt/sda/ljw/Code/AML/LMAResult/xgb_feature_importance.png')
    
    return 0

def get_spectrum(folder_path, file_types=None):
    if file_types is None:
        file_types = ['*.mat', '*.dpt']
    
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_type in file_types:
            extension = file_type.replace('*', '')
            for file in files:
                if file.endswith(extension):
                    all_files.append(os.path.join(root, file))
    
    all_files = sorted(all_files)
    res_dict = {}
    
    for path in all_files:
        filename = os.path.basename(path)
        file_ext = filename.split('.')[-1].lower()
        
        try:
            if file_ext == 'mat':
                mat_data = sio.loadmat(path)
                data_key = 'TR' if 'TR' in mat_data.keys() else 'TR\x00'
                data = mat_data[data_key]
            elif file_ext == 'dpt':
                data = np.loadtxt(path)
            else:
                print(f"Warning: Unsupported file type {file_ext} for {filename}")
                continue

            if data.shape[0] != 1866:
                raise ValueError(f"Unexpected data shape {data.shape} in file {filename}")

            # 提取第二列数据
            res_dict[filename.split('.')[0]] = data[:, 1]
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
    
    return res_dict

if __name__ == '__main__':
    group_list = ['AML', 'healthy']
    dataset = dict()
    
    for group in group_list:
        results = get_spectrum(folder_path=rf'/mnt/sda/ljw/Code/AML/Data/LMA/rawdata/{group}', file_types=['*.dpt'])
        dataset[group] = np.array(list(results.values()))
    
    plt.figure(figsize=(12, 8))

    for label, data in dataset.items():
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        plt.plot(mean, label=f'{label} Mean Spectrum')
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, label=f'{label} Std')

    plt.xlabel('Wave Length')
    plt.ylabel('Intensity')
    plt.title('Mean Spectra Comparison with Standard Deviation between AML and Healthy Groups')
    plt.legend()
    plt.show()
    plt.savefig('/mnt/sda/ljw/Code/AML/LMAResult/mean_spectrum_comparison.png')


    # 先构造特征X和标签y
    X = []
    y = []

    min_samples = min(len(dataset['AML']), len(dataset['healthy']))
    dataset['AML'] = dataset['AML'][:min_samples]
    dataset['healthy'] = dataset['healthy'][:min_samples]

    for label, data in dataset.items():
        for sample in data:
            X.append(sample)
            y.append(1 if label == 'AML' else 0)

    X = np.array(X)
    y = np.array(y)

    Xgboost_classify(X, y, 5)
    RandomForest_classify(X, y, n_splits=5)