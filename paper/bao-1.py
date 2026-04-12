import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

import numpy as np
import logging
from PIL import Image
import warnings

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix,
                           hamming_loss, cohen_kappa_score, jaccard_score, balanced_accuracy_score,
                           roc_curve, auc)
import joblib
import os
import matplotlib.pyplot as plt
from itertools import cycle

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
DATA_PATH = "dataset_new/train"
TEST_DATA_PATH = "dataset_new/val"
MODEL_SAVE_PATH = "../output-model/sklearn_models"
ROC_SAVE_PATH = "../output-model/roc_curves"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(ROC_SAVE_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# =====================
# Helper: Count Parameters
# =====================

def count_model_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_statistics():
    """Print model parameter statistics"""
    print("\n" + "="*80)
    print("MODEL PARAMETER STATISTICS")
    print("="*80)
    
    inception = get_inception_extractor()
    vgg19 = get_vgg19_extractor()
    
    inc_params = count_model_parameters(inception)
    vgg_params = count_model_parameters(vgg19)
    
    print(f"\nFeature Extractors:")
    print(f"  Inception V3:  {inc_params:,} parameters")
    print(f"  VGG19:         {vgg_params:,} parameters")
    
    classifiers = {
        "SVM": SVC(kernel='rbf', probability=True),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }
    
    print(f"\nClassifiers:")
    for name, clf in classifiers.items():
        params = len(clf.get_params())
        print(f"  {name:15} - {params} hyperparameters")
    
    return inc_params, vgg_params

# =====================
# Section 1: Feature Extractors
# =====================

def get_inception_extractor():
    model = models.inception_v3(weights='DEFAULT', aux_logits=True)
    model.fc = torch.nn.Identity()
    return model.to(DEVICE).eval()

def get_vgg19_extractor():
    model = models.vgg19(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
    return model.to(DEVICE).eval()

# =====================
# Section 2: Extract Features
# =====================

def extract_features(model, loader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            features.append(out.cpu().numpy())
            labels.append(y.numpy())
    return np.vstack(features), np.hstack(labels)

# =====================
# Section 3: Plot ROC Curves
# =====================

def plot_roc_curves(y_test, y_proba, class_names, classifier_name, feature_extractor_name):
    if y_proba is None:
        return
    
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=np.arange(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - {feature_extractor_name} + {classifier_name}')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    filename = os.path.join(ROC_SAVE_PATH, f"roc_{feature_extractor_name}_{classifier_name}.png")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved ROC curve: {filename}")

# =====================
# Section 4: Main Execution
# =====================

def run_reproduction():
    print_model_statistics()
    
    transform_inc = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_vgg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset_train = datasets.ImageFolder(DATA_PATH, transform=transform_inc)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    dataset_test = datasets.ImageFolder(TEST_DATA_PATH, transform=transform_inc)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    logger.info("Extracting features from Inception V3...")
    model_inc = get_inception_extractor()
    X_inc_train, y_train = extract_features(model_inc, loader_train)
    X_inc_test, y_test = extract_features(model_inc, loader_test)
    
    logger.info("Extracting features from VGG19...")
    transform_vgg_final = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_train_vgg = datasets.ImageFolder(DATA_PATH, transform=transform_vgg_final)
    loader_train_vgg = DataLoader(dataset_train_vgg, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    dataset_test_vgg = datasets.ImageFolder(TEST_DATA_PATH, transform=transform_vgg_final)
    loader_test_vgg = DataLoader(dataset_test_vgg, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model_vgg = get_vgg19_extractor()
    X_vgg_train, _ = extract_features(model_vgg, loader_train_vgg)
    X_vgg_test, _ = extract_features(model_vgg, loader_test_vgg)
    
    classifiers = {
        "SVM": SVC(kernel='rbf', probability=True),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }
    
    feature_extractors = {
        "InceptionV3": (X_inc_train, X_inc_test),
        "VGG19": (X_vgg_train, X_vgg_test)
    }
    
    results = {}

    for extractor_name, (X_train, X_test) in feature_extractors.items():
        logger.info(f"\n{'='*60}\nEvaluating {extractor_name}\n{'='*60}")
        results[extractor_name] = {}
        
        for name, clf in classifiers.items():
            logger.info(f"Evaluating: {extractor_name} + {name}")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None
            
            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)
            hamming = hamming_loss(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            jaccard = jaccard_score(y_test, y_pred, average='weighted')
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            auc_score = None
            if y_proba is not None:
                auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
            
            results[extractor_name][name] = {
                'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'mcc': mcc,
                'hamming': hamming, 'kappa': kappa, 'jaccard': jaccard,
                'balanced_acc': balanced_acc, 'auc': auc_score,
                'y_pred': y_pred, 'y_proba': y_proba
            }
            
            print(f"\n{'='*60}")
            print(f"Results: {extractor_name} + {name}")
            print(f"{'='*60}")
            print(f"Accuracy:           {acc:.4f}")
            print(f"Precision:          {pre:.4f}")
            print(f"Recall:             {rec:.4f}")
            print(f"F1 Score:           {f1:.4f}")
            print(f"MCC:                {mcc:.4f}")
            print(f"Cohen Kappa Score:  {kappa:.4f}")
            print(f"Balanced Accuracy:  {balanced_acc:.4f}")
            print(f"Jaccard Score:      {jaccard:.4f}")
            print(f"Hamming Loss:       {hamming:.4f}")
            if auc_score is not None:
                print(f"AUC Score:          {auc_score:.4f}")
            
            print(f"\n{'='*60}")
            print("Classification Report")
            print(f"{'='*60}")
            print(classification_report(y_test, y_pred, 
                                       target_names=dataset_train.classes,
                                       digits=4))
            
            print(f"\n{'='*60}")
            print("Confusion Matrix")
            print(f"{'='*60}")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            
            if y_proba is not None:
                plot_roc_curves(y_test, y_proba, dataset_train.classes, name, extractor_name)
            
            model_filename = os.path.join(MODEL_SAVE_PATH, f"{extractor_name}_{name}_model.joblib")
            scaler_filename = os.path.join(MODEL_SAVE_PATH, f"{extractor_name}_{name}_scaler.joblib")
            joblib.dump(clf, model_filename)
            joblib.dump(scaler, scaler_filename)
            logger.info(f"✓ Saved model: {model_filename}")
            logger.info(f"✓ Saved scaler: {scaler_filename}")

    logger.info("Model evaluation completed.")
    
    summary_file = os.path.join(MODEL_SAVE_PATH, "evaluation_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Evaluation Results - Inception V3 + VGG19 + Sklearn\n")
        f.write("="*80 + "\n\n")
        
        for extractor_name, classifiers_results in results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Feature Extractor: {extractor_name}\n")
            f.write(f"{'='*80}\n")
            
            for classifier_name, metrics_dict in classifiers_results.items():
                f.write(f"\n{'-'*80}\n")
                f.write(f"Classifier: {classifier_name}\n")
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
                if metrics_dict['auc'] is not None and not np.isnan(metrics_dict['auc']):
                    f.write(f"AUC Score:          {metrics_dict['auc']:.4f}\n")
    
    logger.info(f"✓ Saved evaluation summary: {summary_file}")
    
    # Save model statistics
    stats_file = os.path.join(MODEL_SAVE_PATH, "model_parameters.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        inception = get_inception_extractor()
        vgg19 = get_vgg19_extractor()
        
        inc_params = count_model_parameters(inception)
        vgg_params = count_model_parameters(vgg19)
        
        classifiers = {
            "SVM": SVC(kernel='rbf', probability=True),
            "kNN": KNeighborsClassifier(n_neighbors=5),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "AdaBoost": AdaBoostClassifier(),
            "DecisionTree": DecisionTreeClassifier()
        }
        
        f.write("="*80 + "\n")
        f.write("MODEL PARAMETER STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("FEATURE EXTRACTORS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model Name':<20} {'Parameters':>15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Inception V3':<20} {inc_params:>15,}\n")
        f.write(f"{'VGG19':<20} {vgg_params:>15,}\n")
        f.write(f"{'TOTAL':<20} {inc_params + vgg_params:>15,}\n")
        f.write("\n")
        
        f.write("CLASSIFIERS (Hyperparameters):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Classifier Name':<20} {'Hyperparameters':>15}\n")
        f.write("-" * 80 + "\n")
        for name, clf in classifiers.items():
            params = len(clf.get_params())
            f.write(f"{name:<20} {params:>15}\n")
        f.write("-" * 80 + "\n")
        
        f.write("\n\nCLASSIFIER DETAILS:\n")
        f.write("-" * 80 + "\n")
        for name, clf in classifiers.items():
            f.write(f"\n{name}:\n")
            for key, value in clf.get_params().items():
                f.write(f"  {key}: {value}\n")
    
    logger.info(f"✓ Saved model parameters: {stats_file}")

if __name__ == "__main__":
    run_reproduction()