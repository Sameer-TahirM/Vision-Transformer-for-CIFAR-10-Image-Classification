from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# VIT architecture
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.contiguous().view(batch_size, -1, self.patch_dim)
        embeddings = self.projection(patches)
        return embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.positional_encoding

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout(attn_output)
        return attn_output.transpose(0, 1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.self_attn(x))
        x = self.norm1(x)
        x = x + self.dropout2(self.fc(x))
        x = self.norm2(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64, num_heads=4, num_layers=8, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, self.patch_embed.num_patches)
        self.transformer_layers = nn.Sequential(
            *[TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 2, dropout=dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_encoding(x)
        x = self.transformer_layers(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# Hybrid Model Architecture
class HybridCNN(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(1024, 512)
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        patch_size = int(num_patches**0.5)
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x))
        return x

class HybridMLPClassifier(nn.Module):
    def __init__(self, in_features=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cnn_feature_extractor = HybridCNN(embed_dim)
        self.classifier = HybridMLPClassifier(in_features=512, num_classes=num_classes)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cnn_feature_extractor(x)
        x = self.classifier(x)
        return x

# Initialize models
vit_model = VisionTransformer(img_size=32, patch_size=4, in_channels=3, embed_dim=64, num_heads=4, num_layers=8, num_classes=10, dropout=0.1)
hybrid_model = HybridModel(img_size=32, patch_size=4, in_channels=3, embed_dim=64, num_classes=10)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







# Load weights for ViT and Hybrid models
vit_state_dict = torch.load(r'C:\Users\samee\Desktop\University\FAST\Semester 9\GenAI\Assignments\Ass3\Q2\vision_transformer_best_model.pth')
vit_state_dict = {key.replace("patch_embed.projection", "patch_embed.proj"): value for key, value in vit_state_dict.items()}
vit_model.load_state_dict(vit_state_dict, strict=False)
vit_model = vit_model.to(device).eval()

# Load ResNet model directly
resnet_model = torch.load(r"C:\Users\samee\Desktop\University\FAST\Semester 9\GenAI\Assignments\Ass3\Q2\resnet18_PRETRAIN_full_model.pth").to(device).eval()

# Load Hybrid model
hybrid_state_dict = torch.load(r"C:\Users\samee\Desktop\University\FAST\Semester 9\GenAI\Assignments\Ass3\Q2\hybrid_cnn_mlp_best_model.pth")
hybrid_model.load_state_dict(hybrid_state_dict, strict=False)
hybrid_model = hybrid_model.to(device).eval()

# CIFAR-10 class labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define test transformations
test_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = test_transforms(img).unsqueeze(0).to(device)

    # Make predictions
    vit_output = vit_model(img)
    resnet_output = resnet_model(img)
    hybrid_output = hybrid_model(img)

    # Get the top predicted label for each model
    vit_label = labels[vit_output.argmax(1).item()]
    resnet_label = labels[resnet_output.argmax(1).item()]
    hybrid_label = labels[hybrid_output.argmax(1).item()]

    results = {
        'Vision Transformer (ViT)': vit_label,
        'ResNet-18': resnet_label,
        'Hybrid MLP-CNN': hybrid_label
    }

    return render_template('predict.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
