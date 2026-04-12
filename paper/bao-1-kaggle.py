# %% [markdown]
# # Inception V3 + VGG19 + Sklearn Classifiers
# Pipeline trích xuất đặc trưng bằng deep learning và phân loại bằng sklearn

# %% [1] IMPORTS
try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
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

# %% [2] CONFIG & SETUP
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Kaggle paths — chỉnh lại nếu cần
DATA_PATH      = "/kaggle/input/your-dataset/train"
TEST_DATA_PATH = "/kaggle/input/your-dataset/val"
MODEL_SAVE_PATH = "/kaggle/working/sklearn_models"
ROC_SAVE_PATH   = "/kaggle/working/roc_curves"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(ROC_SAVE_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

print(f"Device: {DEVICE}")
print(f"Train path : {DATA_PATH}")
print(f"Test  path : {TEST_DATA_PATH}")

# %% [3] HELPER FUNCTIONS
def count_model_parameters(model):
    """Đếm số tham số có thể huấn luyện"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %% [4] FEATURE EXTRACTORS
def get_inception_extractor():
    model = models.inception_v3(weights='DEFAULT', aux_logits=True)
    model.fc = torch.nn.Identity()
    return model.to(DEVICE).eval()

def get_vgg19_extractor():
    model = models.vgg19(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
    return model.to(DEVICE).eval()

def print_model_summary(model, model_name, input_size):
    """In model summary dạng bảng: layer, output shape, params (giống Keras model.summary())"""
    print("\n" + "="*80)
    print(f"Model Summary: {model_name}")
    print("="*80)

    if TORCHINFO_AVAILABLE:
        torchinfo_summary(
            model,
            input_size=input_size,
            col_names=["output_size", "num_params"],
            col_width=25,
            row_settings=["var_names"],
            depth=3,
            verbose=1,
        )
    else:
        # Fallback: in thủ công từng layer
        print(f"  {'Layer (type)':<45} {'Output Shape':<25} {'Param #':>12}")
        print(f"  {'='*45} {'='*25} {'='*12}")
        total_params     = 0
        trainable_params = 0
        for name, module in model.named_modules():
            if not list(module.children()):          # chỉ in leaf modules
                params     = sum(p.numel() for p in module.parameters())
                trainable  = sum(p.numel() for p in module.parameters() if p.requires_grad)
                layer_type = module.__class__.__name__
                short_name = name if name else "(root)"
                print(f"  {(short_name + ' (' + layer_type + ')'):<45} {'?':<25} {params:>12,}")
                total_params     += params
                trainable_params += trainable
        non_trainable = total_params - trainable_params
        print(f"  {'='*84}")
        print(f"  Total params:         {total_params:>12,}  ({total_params*4/1024**2:.2f} MB)")
        print(f"  Trainable params:     {trainable_params:>12,}  ({trainable_params*4/1024**2:.2f} MB)")
        print(f"  Non-trainable params: {non_trainable:>12,}  ({non_trainable*4/1024**2:.2f} MB)")

def print_model_statistics():
    """In thống kê tham số và model summary của các feature extractor"""
    inception = get_inception_extractor()
    vgg19     = get_vgg19_extractor()

    # In summary chi tiết (layer-by-layer)
    print_model_summary(inception, "Inception V3", input_size=(1, 3, 299, 299))
    print_model_summary(vgg19,     "VGG19",        input_size=(1, 3, 224, 224))

    inc_params = count_model_parameters(inception)
    vgg_params = count_model_parameters(vgg19)

    print("\n" + "="*80)
    print("CLASSIFIER HYPERPARAMETERS")
    print("="*80)
    classifiers_info = {
        "SVM":          SVC(kernel='rbf', probability=True),
        "kNN":          KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "AdaBoost":     AdaBoostClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }
    print(f"  {'Classifier':<20} {'# Hyperparams':>15}")
    print(f"  {'─'*20} {'─'*15}")
    for name, clf in classifiers_info.items():
        params = len(clf.get_params())
        print(f"  {name:<20} {params:>15}")

    return inc_params, vgg_params

# In thống kê ngay khi chạy cell này
print_model_statistics()

# %% [5] DATALOADER SETUP
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

# Inception loaders
dataset_train     = datasets.ImageFolder(DATA_PATH, transform=transform_inc)
loader_train      = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

dataset_test      = datasets.ImageFolder(TEST_DATA_PATH, transform=transform_inc)
loader_test       = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# VGG19 loaders
dataset_train_vgg = datasets.ImageFolder(DATA_PATH, transform=transform_vgg)
loader_train_vgg  = DataLoader(dataset_train_vgg, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

dataset_test_vgg  = datasets.ImageFolder(TEST_DATA_PATH, transform=transform_vgg)
loader_test_vgg   = DataLoader(dataset_test_vgg, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"\n{'='*60}")
print(f"DATASET SUMMARY")
print(f"{'='*60}")
print(f"Classes ({len(dataset_train.classes)}): {dataset_train.classes}")

# Log số lượng file theo từng class — Train
print(f"\n{'─'*60}")
print(f"{'Train Dataset':} — {DATA_PATH}")
print(f"{'─'*60}")
train_counts = {cls: 0 for cls in dataset_train.classes}
for _, label in dataset_train.samples:
    train_counts[dataset_train.classes[label]] += 1
for cls, cnt in train_counts.items():
    print(f"  {cls:<30} {cnt:>6} files")
print(f"  {'TOTAL':<30} {len(dataset_train):>6} files")

# Log số lượng file theo từng class — Test/Val
print(f"\n{'─'*60}")
print(f"{'Test/Val Dataset':} — {TEST_DATA_PATH}")
print(f"{'─'*60}")
test_counts = {cls: 0 for cls in dataset_test.classes}
for _, label in dataset_test.samples:
    test_counts[dataset_test.classes[label]] += 1
for cls, cnt in test_counts.items():
    print(f"  {cls:<30} {cnt:>6} files")
print(f"  {'TOTAL':<30} {len(dataset_test):>6} files")
print(f"{'='*60}\n")

# %% [6] EXTRACT FEATURES
def extract_features(model, loader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            features.append(out.cpu().numpy())
            labels.append(y.numpy())
    return np.vstack(features), np.hstack(labels)

# --- Inception V3 ---
logger.info("Extracting features from Inception V3...")
model_inc = get_inception_extractor()
X_inc_train, y_train = extract_features(model_inc, loader_train)
X_inc_test,  y_test  = extract_features(model_inc, loader_test)
print(f"[InceptionV3] Train features: {X_inc_train.shape} | Test features: {X_inc_test.shape}")

# --- VGG19 ---
logger.info("Extracting features from VGG19...")
model_vgg = get_vgg19_extractor()
X_vgg_train, _ = extract_features(model_vgg, loader_train_vgg)
X_vgg_test,  _ = extract_features(model_vgg, loader_test_vgg)
print(f"[VGG19]       Train features: {X_vgg_train.shape} | Test features: {X_vgg_test.shape}")

# %% [7] DEFINE CLASSIFIERS
classifiers = {
    "SVM":          SVC(kernel='rbf', probability=True),
    "kNN":          KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "AdaBoost":     AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

feature_extractors = {
    "InceptionV3": (X_inc_train, X_inc_test),
    "VGG19":       (X_vgg_train, X_vgg_test)
}

print("Classifiers và Feature Extractors đã sẵn sàng.")

# %% [8] PLOT ROC CURVES (helper)
def plot_roc_curves(y_test, y_proba, class_names, classifier_name, feature_extractor_name):
    if y_proba is None:
        return

    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=np.arange(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
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

# %% [9] TRAINING & EVALUATION
results = {}

for extractor_name, (X_train, X_test) in feature_extractors.items():
    logger.info(f"\n{'='*60}\nEvaluating {extractor_name}\n{'='*60}")
    results[extractor_name] = {}

    for name, clf in classifiers.items():
        logger.info(f"Training: {extractor_name} + {name}")

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        clf.fit(X_train_scaled, y_train)
        y_pred  = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None

        acc          = accuracy_score(y_test, y_pred)
        pre          = precision_score(y_test, y_pred, average='weighted')
        rec          = recall_score(y_test, y_pred, average='weighted')
        f1           = f1_score(y_test, y_pred, average='weighted')
        mcc          = matthews_corrcoef(y_test, y_pred)
        hamming      = hamming_loss(y_test, y_pred)
        kappa        = cohen_kappa_score(y_test, y_pred)
        jaccard      = jaccard_score(y_test, y_pred, average='weighted')
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        auc_score    = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None

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
        print(confusion_matrix(y_test, y_pred))

        if y_proba is not None:
            plot_roc_curves(y_test, y_proba, dataset_train.classes, name, extractor_name)

        # Lưu model và scaler
        model_filename  = os.path.join(MODEL_SAVE_PATH, f"{extractor_name}_{name}_model.joblib")
        scaler_filename = os.path.join(MODEL_SAVE_PATH, f"{extractor_name}_{name}_scaler.joblib")
        joblib.dump(clf, model_filename)
        joblib.dump(scaler, scaler_filename)
        logger.info(f"✓ Saved: {model_filename}")

logger.info("✅ Training & Evaluation completed.")

# %% [10] SAVE SUMMARY REPORT
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

# %% [11] SAVE MODEL PARAMETER STATS
stats_file = os.path.join(MODEL_SAVE_PATH, "model_parameters.txt")

inception  = get_inception_extractor()
vgg19      = get_vgg19_extractor()
inc_params = count_model_parameters(inception)
vgg_params = count_model_parameters(vgg19)

classifiers_stats = {
    "SVM":          SVC(kernel='rbf', probability=True),
    "kNN":          KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "AdaBoost":     AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("MODEL PARAMETER STATISTICS\n")
    f.write("="*80 + "\n\n")

    f.write("FEATURE EXTRACTORS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Model Name':<20} {'Parameters':>15}\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Inception V3':<20} {inc_params:>15,}\n")
    f.write(f"{'VGG19':<20} {vgg_params:>15,}\n")
    f.write(f"{'TOTAL':<20} {inc_params + vgg_params:>15,}\n\n")

    f.write("CLASSIFIERS (Hyperparameters):\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Classifier Name':<20} {'Hyperparameters':>15}\n")
    f.write("-" * 80 + "\n")
    for name, clf in classifiers_stats.items():
        params = len(clf.get_params())
        f.write(f"{name:<20} {params:>15}\n")
    f.write("-" * 80 + "\n")

    f.write("\n\nCLASSIFIER DETAILS:\n")
    f.write("-" * 80 + "\n")
    for name, clf in classifiers_stats.items():
        f.write(f"\n{name}:\n")
        for key, value in clf.get_params().items():
            f.write(f"  {key}: {value}\n")

logger.info(f"✓ Saved model parameters: {stats_file}")
print("✅ All done!")
