import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           matthews_corrcoef, roc_auc_score, classification_report,
                           confusion_matrix, hamming_loss, cohen_kappa_score, 
                           jaccard_score, balanced_accuracy_score, roc_curve, auc)

# Cấu hình log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================
# CONFIGURATION
# =====================
DATA_PATH = "dataset_new/train"  # Thay đổi đường dẫn tới tập dữ liệu của bạn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_FOLDS = 10
CACHE_DIR = "extracted_features"
OUTPUT_DIR = "../output-model"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# 1. MODEL FACTORY (THEO REVIEW 2025)
# =====================
def get_extractor(model_name):
    """Khởi tạo các bộ trích xuất đặc trưng theo tài liệu [cite: 9, 1371, 1569]"""
    if model_name == "InceptionV3":
        model = models.inception_v3(weights='DEFAULT', aux_logits=True)
        model.fc = torch.nn.Identity()
        input_size = 299
    elif model_name == "VGG19":
        model = models.vgg19(weights='DEFAULT')
        # Lấy đặc trưng từ lớp FC thứ 2 (4096 chiều) [cite: 181, 188]
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
        input_size = 224
    elif model_name == "EfficientNet_B5":
        model = models.efficientnet_b5(weights='DEFAULT')
        model.classifier = torch.nn.Identity()
        input_size = 456
    elif model_name == "ViT":
        model = models.vit_b_16(weights='DEFAULT')
        model.heads = torch.nn.Identity()
        input_size = 224
    else:
        raise ValueError("Model không hỗ trợ.")

    return model.to(DEVICE).eval(), input_size

# =====================
# 2. FEATURE EXTRACTION & CACHING
# =====================
def get_features(model_name):
    """Trích xuất và lưu trữ đặc trưng để tiết kiệm thời gian chạy lại"""
    cache_path_X = f"{CACHE_DIR}/{model_name}_X.npy"
    cache_path_y = f"{CACHE_DIR}/{model_name}_y.npy"

    if os.path.exists(cache_path_X):
        logger.info(f"Đang tải {model_name} đặc trưng từ cache...")
        return np.load(cache_path_X), np.load(cache_path_y)

    model, input_size = get_extractor(model_name)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    features, labels = [], []
    logger.info(f"Bắt đầu trích xuất đặc trưng với {model_name}...")
    
    with torch.no_grad():
        for imgs, y in loader:
            out = model(imgs.to(DEVICE))
            # Xử lý output của InceptionV3 nếu cần
            if isinstance(out, models.inception.InceptionOutputs):
                out = out.logits
            features.append(out.cpu().numpy())
            labels.append(y.numpy())

    X = np.vstack(features)
    y = np.hstack(labels)
    
    np.save(cache_path_X, X)
    np.save(cache_path_y, y)
    return X, y

# =====================
# 3. EVALUATION (STRATIFIED 10-FOLD CV)
# =====================
def run_experiment(model_name, classifier_name, dataset, class_names):
    """Chạy 10-Fold CV với chi tiết metrics đầy đủ"""
    X, y = get_features(model_name)
    
    # Khởi tạo bộ phân loại ML [cite: 85, 201, 208, 211, 215]
    if classifier_name == "SVM":
        clf = SVC(kernel='rbf', probability=True)
    elif classifier_name == "kNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Classifier không hỗ trợ.")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # Lưu trữ metrics từng fold
    metrics = {
        'acc': [], 'pre': [], 'rec': [], 'f1': [], 'mcc': [], 
        'hamming': [], 'kappa': [], 'jaccard': [], 'balanced_acc': [], 'auc': []
    }
    all_y_test = []
    all_y_pred = []

    logger.info(f"Đang chạy 10-Fold CV cho {model_name} + {classifier_name}...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Chuẩn hóa (Cần thiết cho SVM/kNN) [cite: 212]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
        
        # Tính toán metrics từng fold
        metrics['acc'].append(accuracy_score(y_test, y_pred))
        metrics['pre'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['rec'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['mcc'].append(matthews_corrcoef(y_test, y_pred))
        metrics['hamming'].append(hamming_loss(y_test, y_pred))
        metrics['kappa'].append(cohen_kappa_score(y_test, y_pred))
        metrics['jaccard'].append(jaccard_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['balanced_acc'].append(balanced_accuracy_score(y_test, y_pred))
        
        if y_proba is not None:
            try:
                metrics['auc'].append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
            except:
                metrics['auc'].append(0.0)
        
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    # Average across folds
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    # In result
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ: {model_name} + {classifier_name} (10-Fold CV)")
    print(f"{'='*60}")
    print(f"Accuracy:           {avg_metrics['acc']:.4f}")
    print(f"Precision:          {avg_metrics['pre']:.4f}")
    print(f"Recall:             {avg_metrics['rec']:.4f}")
    print(f"F1 Score:           {avg_metrics['f1']:.4f}")
    print(f"MCC:                {avg_metrics['mcc']:.4f}")
    print(f"Cohen Kappa Score:  {avg_metrics['kappa']:.4f}")
    print(f"Balanced Accuracy:  {avg_metrics['balanced_acc']:.4f}")
    print(f"Jaccard Score:      {avg_metrics['jaccard']:.4f}")
    print(f"Hamming Loss:       {avg_metrics['hamming']:.4f}")
    print(f"AUC Score:          {avg_metrics['auc']:.4f}")
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT (PER-CLASS METRICS)")
    print(f"{'='*60}")
    print(classification_report(all_y_test, all_y_pred, 
                               target_names=class_names,
                               digits=4))
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    cm = confusion_matrix(all_y_test, all_y_pred)
    print(cm)
    
    return avg_metrics

# =====================
# MAIN EXECUTION
# =====================
if __name__ == "__main__":
    # Load dataset để lấy class names
    dataset = datasets.ImageFolder(DATA_PATH)
    class_names = dataset.classes
    
    logger.info(f"Classes: {class_names}")
    
    # Tất cả combinations
    extractors = ["InceptionV3", "VGG19", "EfficientNet_B5", "ViT"]
    classifiers = ["SVM", "kNN"]
    
    results = {}
    
    for extractor in extractors:
        results[extractor] = {}
        for classifier in classifiers:
            try:
                avg_metrics = run_experiment(extractor, classifier, dataset, class_names)
                results[extractor][classifier] = avg_metrics
            except Exception as e:
                logger.error(f"Error with {extractor} + {classifier}: {e}")
    
    # Lưu summary
    summary_file = os.path.join(OUTPUT_DIR, "bao2_evaluation_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BÁNG CÓ KỂTQUẢ ĐÁNH GIÁ MÔ HÌNH - BAO 2\n")
        f.write("(4 Feature Extractors × 2 Classifiers - 10-Fold CV)\n")
        f.write("="*80 + "\n\n")
        
        for extractor_name, classifiers_results in results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"FEATURE EXTRACTOR: {extractor_name}\n")
            f.write(f"{'='*80}\n")
            
            for classifier_name, metrics_dict in classifiers_results.items():
                f.write(f"\n{'-'*80}\n")
                f.write(f"CLASSIFIER: {classifier_name}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Accuracy:           {metrics_dict['acc']:.4f}\n")
                f.write(f"Precision:          {metrics_dict['pre']:.4f}\n")
                f.write(f"Recall:             {metrics_dict['rec']:.4f}\n")
                f.write(f"F1 Score:           {metrics_dict['f1']:.4f}\n")
                f.write(f"Matthews Corr Coef: {metrics_dict['mcc']:.4f}\n")
                f.write(f"Cohen Kappa Score:  {metrics_dict['kappa']:.4f}\n")
                f.write(f"Balanced Accuracy:  {metrics_dict['balanced_acc']:.4f}\n")
                f.write(f"Jaccard Score:      {metrics_dict['jaccard']:.4f}\n")
                f.write(f"Hamming Loss:       {metrics_dict['hamming']:.4f}\n")
                f.write(f"AUC Score:          {metrics_dict['auc']:.4f}\n")
    
    logger.info(f"✓ Summary lưu tại: {summary_file}")