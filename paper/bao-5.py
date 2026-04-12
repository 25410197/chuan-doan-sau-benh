import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim import lr_scheduler
from collections import Counter

import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
IMG_SIZE   = (256, 256)
SEED       = 42
torch.manual_seed(SEED)

TRAIN_DIR    = "/kaggle/input/datasets/bananalatraichuoi/rambutan-cropped/rambutan-dataset-cropped/train"
VAL_DIR      = "/kaggle/input/datasets/bananalatraichuoi/rambutan-cropped/rambutan-dataset-cropped/val"
TEST_DIR     = "/kaggle/input/datasets/bananalatraichuoi/rambutan-cropped/rambutan-dataset-cropped/test"
WORKING_PATH = "/kaggle/working/results"
os.makedirs(WORKING_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────
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
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        alloc_mem = torch.cuda.memory_allocated(0) / 1e9
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
    """In chi tiết params từng tensor: tên, shape, #elements, trainable."""
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
# 1. MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel + spatial attention."""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        x = x * self.sigmoid_channel(avg_out + max_out).view(b, c, 1, 1)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_out


class InceptionResNetBlock(nn.Module):
    """Multi-scale feature extraction with residual connection."""
    def __init__(self, in_channels, out_channels):
        super(InceptionResNetBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        )
        self.reduction = nn.Conv2d(out_channels // 4 * 3, out_channels, kernel_size=1)
        self.shortcut  = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu      = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        out = torch.cat([self.branch1x1(x), self.branch3x3(x), self.branch5x5(x)], dim=1)
        return self.relu(self.reduction(out) + res)


class CoordinateAttention(nn.Module):
    """Coordinate Attention — encodes spatial location info along X and Y axes."""
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))


class GrapeLeafNet(nn.Module):
    """Dual-track CNN + Transformer architecture with attention fusion."""
    def __init__(self, num_classes):
        super(GrapeLeafNet, self).__init__()

        # Track 1: CNN Branch
        self.cnn_track = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            InceptionResNetBlock(32, 64),
            nn.MaxPool2d(2),
            CBAM(64),
            InceptionResNetBlock(64, 128),
            nn.MaxPool2d(2),
            CBAM(128)
        )

        # Track 2: Transformer Branch
        self.patch_embed = nn.Conv2d(3, 128, kernel_size=4, stride=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fusion & Classification
        self.ca          = CoordinateAttention(256, 256)
        self.flatten     = nn.Flatten()
        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x1 = self.cnn_track(x)

        x2 = self.patch_embed(x)
        b, c, h, w = x2.shape
        x2 = self.transformer_encoder(x2.flatten(2).permute(0, 2, 1))
        x2 = x2.permute(0, 2, 1).view(b, c, h, w)

        out = self.ca(torch.cat([x1, x2], dim=1))
        out = self.global_pool(out)
        return self.fc(self.flatten(out))

# ──────────────────────────────────────────────────────────────
# 2. DATA PIPELINE
# ──────────────────────────────────────────────────────────────
class MapDataset(torch.utils.data.Dataset):
    """Wrapper để áp dụng transform riêng cho từng split."""
    def __init__(self, dataset, transform=None):
        self.dataset   = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


def get_dataloaders():
    """Load dữ liệu từ TRAIN_DIR / VAL_DIR / TEST_DIR cố định."""
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=(0.5, 1.5)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    val_dataset   = datasets.ImageFolder(VAL_DIR)
    test_dataset  = datasets.ImageFolder(TEST_DIR)
    class_names   = train_dataset.classes

    # Tính class weights từ tập Train để xử lý mất cân bằng
    counts       = Counter(train_dataset.targets)
    class_counts = [counts[i] for i in range(len(class_names))]
    weights      = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights      = (weights / weights.sum()).to(device)

    train_loader = DataLoader(MapDataset(train_dataset, train_transform), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(MapDataset(val_dataset,   test_transform),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(MapDataset(test_dataset,  test_transform),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names, weights

# ──────────────────────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=50):
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted   = torch.max(model(images).data, 1)
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(WORKING_PATH, 'best_model.pth'))
            print("  ✓ Best model saved!")

        if (epoch + 1) % 20 == 0:
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }
            ckpt_path = os.path.join(WORKING_PATH, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)
            print(f"  ✓ Periodic checkpoint saved: {ckpt_path}")

# ──────────────────────────────────────────────────────────────
# 4. EVALUATION
# ──────────────────────────────────────────────────────────────
def evaluate_detailed_performance(model, test_loader, class_names):
    """Đánh giá chi tiết: classification report + confusion matrix."""
    model.eval()
    y_true, y_pred = [], []

    logger.info("Predicting on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, preds = torch.max(model(images), 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path = os.path.join(WORKING_PATH, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()

# ──────────────────────────────────────────────────────────────
# 5. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    check_gpu()

    # Validate paths
    for path, name in [(TRAIN_DIR, "TRAIN_DIR"), (VAL_DIR, "VAL_DIR"), (TEST_DIR, "TEST_DIR")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ {name} not found: {path}")
    print("✓ Dataset paths validated")

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()

    # Dataset summary
    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    val_dataset   = datasets.ImageFolder(VAL_DIR)
    test_dataset  = datasets.ImageFolder(TEST_DIR)

    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Classes ({len(class_names)}): {', '.join(class_names)}")
    for split_name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        print(f"\n  {'─'*56}")
        print(f"  {split_name}")
        print(f"  {'─'*56}")
        counts = Counter(ds.targets)
        for i, cls in enumerate(class_names):
            print(f"    {cls:<30} {counts[i]:>6} files")
        print(f"    {'TOTAL':<30} {len(ds):>6} files")
    print(f"{'='*60}\n")

    # Build model
    model = GrapeLeafNet(num_classes=len(class_names))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"✓ DataParallel on {torch.cuda.device_count()} GPUs")
    model = model.to(device)

    # Log model summary + parameter detail
    print_model_summary(model, "GrapeLeafNet", input_size=(1, 3, 256, 256))
    get_model_params_detail(model, "GrapeLeafNet")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_model(model, train_loader, val_loader, epochs=120)

    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(WORKING_PATH, 'best_model.pth')))
    evaluate_detailed_performance(model, test_loader, class_names)