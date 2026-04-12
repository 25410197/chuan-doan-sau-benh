import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import warnings
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import os
from pathlib import Path
import contextlib
import time

try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────
def get_autocast_context():
    """Trả về autocast context phù hợp với thiết bị hiện tại."""
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type='cuda')
    return contextlib.nullcontext()


def check_gpu():
    """In thông tin GPU/CUDA."""
    print("=" * 60)
    print("GPU / CUDA CHECK")
    print("=" * 60)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    print(f"  CUDA version    : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

    if torch.cuda.is_available():
        print(f"  GPU Device      : {torch.cuda.get_device_name(0)}")
        print(f"  GPU Count       : {torch.cuda.device_count()}")
        total_mem     = torch.cuda.get_device_properties(0).total_memory / 1e9
        alloc_mem     = torch.cuda.memory_allocated(0) / 1e9
        print(f"  GPU Memory      : {alloc_mem:.2f} GB / {total_mem:.2f} GB")
    else:
        print("  ⚠️  CUDA not detected — training will run on CPU (slow)")
    print("=" * 60)


def print_model_summary(model, model_name: str, input_size: tuple):
    """In model summary dạng bảng (layer, output shape, #params) giống Keras model.summary()."""
    print("\n" + "=" * 80)
    print(f"Model Summary: {model_name}")
    print("=" * 80)

    if TORCHINFO_AVAILABLE:
        torchinfo_summary(
            model,
            input_size=input_size,
            col_names=["output_size", "num_params"],
            col_width=25,
            row_settings=["var_names"],
            depth=4,
            verbose=1,
        )
    else:
        print(f"  {'Layer (type)':<45} {'Param #':>12}")
        print(f"  {'='*45} {'='*12}")
        total_params     = 0
        trainable_params = 0
        for name, module in model.named_modules():
            if not list(module.children()):
                params    = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                label     = f"{name} ({module.__class__.__name__})" if name else f"(root) ({module.__class__.__name__})"
                print(f"  {label:<45} {params:>12,}")
                total_params     += params
                trainable_params += trainable
        non_trainable = total_params - trainable_params
        print(f"  {'='*58}")
        print(f"  Total params:         {total_params:>12,}  ({total_params * 4 / 1024**2:.2f} MB)")
        print(f"  Trainable params:     {trainable_params:>12,}  ({trainable_params * 4 / 1024**2:.2f} MB)")
        print(f"  Non-trainable params: {non_trainable:>12,}  ({non_trainable * 4 / 1024**2:.2f} MB)")


def get_model_params_detail(model, model_name: str):
    """In chi tiết params từng nhóm layer: tên, shape, #elements, trainable."""
    print("\n" + "=" * 80)
    print(f"Parameter Detail: {model_name}")
    print("=" * 80)
    print(f"  {'Name':<50} {'Shape':<25} {'#Params':>10}  {'Trainable'}")
    print(f"  {'─'*50} {'─'*25} {'─'*10}  {'─'*9}")

    total_params     = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        n         = param.numel()
        trainable = param.requires_grad
        total_params     += n
        trainable_params += n if trainable else 0
        print(f"  {name:<50} {str(list(param.shape)):<25} {n:>10,}  {'✓' if trainable else '✗'}")

    non_trainable = total_params - trainable_params
    print(f"  {'─'*80}")
    print(f"  Total params         : {total_params:>12,}  ({total_params * 4 / 1024**2:.2f} MB)")
    print(f"  Trainable params     : {trainable_params:>12,}  ({trainable_params * 4 / 1024**2:.2f} MB)")
    print(f"  Non-trainable params : {non_trainable:>12,}  ({non_trainable * 4 / 1024**2:.2f} MB)")
    print("=" * 80)

    return {"total": total_params, "trainable": trainable_params, "non_trainable": non_trainable}

# ──────────────────────────────────────────────────────────────
# 1. DATA AUGMENTATION
# ──────────────────────────────────────────────────────────────
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ──────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────
def get_custom_mobilenet_v3(num_classes):
    model = models.mobilenet_v3_large(weights='DEFAULT')
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    n_inputs = 960
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes)
    )
    return model

def unfreeze_backbone_for_finetuning(model, unfreeze_from_layer=35):
    for idx, param in enumerate(model.features.parameters()):
        if idx >= unfreeze_from_layer:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    for param in model.classifier.parameters():
        param.requires_grad = True

# ──────────────────────────────────────────────────────────────
# 3. TRAINING PROCESS
# ──────────────────────────────────────────────────────────────
def train_model_2stage(model, train_loader, val_loader, test_loader=None, 
                       stage1_epochs=20, stage2_epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\n✓ Using device: {device}")
    print(f"✓ Model moved to {device}")
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    ckpt_dir = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    # ===== STAGE 1: FEATURE EXTRACTION =====
    print("\n" + "="*60)
    print("STAGE 1: FEATURE EXTRACTION (Freeze backbone)")
    print("="*60)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    for epoch in range(stage1_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}/{stage1_epochs}"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with get_autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{stage1_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / 'best_model_stage1.pt')
            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠️  Early stopping at epoch {epoch+1}")
                break
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared before Stage 2")
    
    # ===== STAGE 2: FINE-TUNING =====
    print("\n" + "="*60)
    print("STAGE 2: FINE-TUNING (Unfreeze deeper layers)")
    print("="*60)
    
    model.load_state_dict(torch.load(ckpt_dir / 'best_model_stage1.pt', map_location=device, weights_only=True))
    unfreeze_backbone_for_finetuning(model, unfreeze_from_layer=35)
    print("✓ Unfrozen layers from index 35 onwards")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(stage2_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{stage2_epochs}"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with get_autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{stage2_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / 'best_model_stage2.pt')
            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠️  Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(ckpt_dir / 'best_model_stage2.pt', map_location=device, weights_only=True))
    print("\n✓ Loaded best model")
    
    return model

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with get_autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate_metrics(model, data_loader, num_classes, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            
            with get_autocast_context():
                outputs = model(inputs)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probas.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    if num_classes == 2:
        roc_auc = roc_auc_score(all_labels, all_probas[:, 1])
    else:
        roc_auc = roc_auc_score(all_labels, all_probas, multi_class='ovr', average='macro')
    
    accuracy = np.mean(all_preds == all_labels)
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    print("\n" + "-"*60)
    print("Per-Class Metrics:")
    print("-"*60)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    for i in range(num_classes):
        print(f"{class_names[i]:30s} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1: {f1[i]:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probas
    }

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()

def plot_roc_curves(y_true, y_scores, class_names, save_path):
    num_classes = len(class_names)
    plt.figure(figsize=(10, 8))
    colors = plt.colormaps['hsv'](np.linspace(0, 0.9, num_classes))
    
    for i in range(num_classes):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ ROC curves saved: {save_path}")
    plt.close()

# ──────────────────────────────────────────────────────────────
# 4. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    check_gpu()
    
    BASE_INPUT_PATH = "/kaggle/input/datasets/tminhhi/smart-data/dataset_smartcrop"
    WORKING_PATH = "/kaggle/working/results"
    os.makedirs(WORKING_PATH, exist_ok=True)
    
    TRAIN_DIR = os.path.join(BASE_INPUT_PATH, 'train')
    VAL_DIR = os.path.join(BASE_INPUT_PATH, 'val')
    TEST_DIR = os.path.join(BASE_INPUT_PATH, 'val')
    
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"❌ TRAIN_DIR không tìm thấy: {TRAIN_DIR}. Hãy chắc chắn bạn đã copy data vào thư mục working!")
    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"❌ VAL_DIR không tìm thấy: {VAL_DIR}")
    print("✓ Dataset paths validated successfully")
    
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms['val'])
    
    test_loader = None
    if os.path.exists(TEST_DIR):
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=data_transforms['test'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        print(f"✓ Test set loaded: {len(test_dataset)} images")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Classes ({num_classes}): {', '.join(class_names)}")

    print(f"\n  {'─'*56}")
    print(f"  Train — {TRAIN_DIR}")
    print(f"  {'─'*56}")
    train_counts = {cls: 0 for cls in class_names}
    for _, lbl in train_dataset.samples:
        train_counts[class_names[lbl]] += 1
    for cls, cnt in train_counts.items():
        print(f"    {cls:<30} {cnt:>6} files")
    print(f"    {'TOTAL':<30} {len(train_dataset):>6} files")

    print(f"\n  {'─'*56}")
    print(f"  Val — {VAL_DIR}")
    print(f"  {'─'*56}")
    val_counts = {cls: 0 for cls in class_names}
    for _, lbl in val_dataset.samples:
        val_counts[class_names[lbl]] += 1
    for cls, cnt in val_counts.items():
        print(f"    {cls:<30} {cnt:>6} files")
    print(f"    {'TOTAL':<30} {len(val_dataset):>6} files")
    print(f"{'='*60}\n")

    plant_model = get_custom_mobilenet_v3(num_classes)

    # Log model summary + parameter detail
    print_model_summary(plant_model, "MobileNetV3-Large (custom)", input_size=(1, 3, 224, 224))
    get_model_params_detail(plant_model, "MobileNetV3-Large (custom)")
    
    # Smoke-test: xác nhận DataLoader hoạt động
    print("\nSmoke-test DataLoader...")
    start_time = time.time()
    try:
        test_inputs, _ = next(iter(train_loader))
        print(f"  ✓ Loaded {test_inputs.size(0)} images in {time.time() - start_time:.2f}s — DataLoader OK")
    except Exception as e:
        print(f"  ❌ DataLoader error: {e}")

    print(f"\n🚀 Starting 2-stage training...")
    plant_model = train_model_2stage(
        plant_model, 
        train_loader, 
        val_loader,
        test_loader,
        stage1_epochs=20,
        stage2_epochs=15
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n📈 Evaluating on validation set...")
    val_metrics = evaluate_metrics(plant_model, val_loader, num_classes, device, class_names)
    
    if test_loader is not None:
        print(f"\n📊 Evaluating on test set...")
        test_metrics = evaluate_metrics(plant_model, test_loader, num_classes, device, class_names)
    
    print(f"\n📊 Generating visualizations...")
    plot_confusion_matrix(val_metrics['confusion_matrix'], class_names, os.path.join(WORKING_PATH, 'confusion_matrix_val.png'))
    plot_roc_curves(val_metrics['labels'], val_metrics['probabilities'], class_names, os.path.join(WORKING_PATH, 'roc_curves_val.png'))
    
    print(f"\n✅ Training completed!")
    print(f"✓ Results saved to: {WORKING_PATH}")