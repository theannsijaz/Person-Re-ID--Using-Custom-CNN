import os
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import shutil
import re

# New imports for enhanced functionality
import faiss
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import umap
import pandas as pd
from collections import defaultdict
import cv2
import torch.nn.functional as F
from torch.nn import functional as func

# Dataset class for gallery and query images
class ReIDDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []
        self.person_ids = []
        self.camera_ids = []
        
        # Parse Market-1501 naming convention
        for img_name in os.listdir(data_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(data_path, img_name)
            self.image_paths.append(img_path)
            
            # Extract IDs using regex pattern for Market-1501
            # Format is usually: id_cameraID_sequenceNumber.jpg or similar variations
            # Example: 1490_c5s3_062240_00.jpg
            parts = img_name.split('_')
            if len(parts) >= 2:
                # First part is person ID
                try:
                    person_id = int(parts[0])
                    
                    # Second part contains camera info (c5s3)
                    camera_part = parts[1]
                    # Extract camera number (just the digit after 'c')
                    camera_match = re.search(r'c(\d+)', camera_part)
                    if camera_match:
                        camera_id = int(camera_match.group(1))
                    else:
                        camera_id = 0  # Default if pattern doesn't match
                        
                    self.person_ids.append(person_id)
                    self.camera_ids.append(camera_id)
                except (ValueError, IndexError):
                    # Skip files that don't match the expected format
                    self.image_paths.pop()  # Remove the path we just added
                    continue
            else:
                # Skip files that don't match expected format
                continue
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        person_id = self.person_ids[idx]
        camera_id = self.camera_ids[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, person_id, camera_id, img_path

# Cell 5: Model definition
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction_ratio, 8), kernel_size=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(max(channels // reduction_ratio, 8), channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),  # Reduced kernel size for smaller inputs
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa_input = torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        return x

class Person_ReID(nn.Module):
    def __init__(self):
        super(Person_ReID, self).__init__()
        
        # Static filters with no downsampling
        self.static_conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # First layer with reduced stride to preserve spatial info
        self.trainable_conv = nn.Conv2d(3, 13, kernel_size=3, stride=1, padding=1, bias=False)
        self._initialize_static_filters()
        
        self.bn1 = nn.BatchNorm2d(16)
        
        # Reduced number of downsamplings for 128x64 input
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128x64 -> 64x32
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x32 -> 32x16
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x16 -> 16x8
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 16x8 -> 16x8
            nn.BatchNorm2d(256),
            nn.ReLU6()
        )
        
        # CBAM attention modules
        self.cbam4 = CBAM(128, reduction_ratio=8)
        self.cbam5 = CBAM(256, reduction_ratio=8)
        
        # Skip connections with appropriate strides
        self.skip_16_to_32 = nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False)
        self.skip_32_to_64 = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False)
        self.skip_64_to_128 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.skip_128_to_256 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        
        # Dropout for regularization
        self.dropout5 = nn.Dropout(0.3)
        
        # Part-based features - 3 parts instead of 2 for a 16x8 feature map (head, torso, legs)
        self.part_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),  # Global
            nn.AdaptiveMaxPool2d((6, 1))   # 3 vertical parts for person Re-ID
        ])
        
        # Global pool
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Enhanced global feature projection
        self.fc_global = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            #nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        # Part-based feature projections (6 parts)
        self.fc_parts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128)
            ) for _ in range(6)  # For 6 body parts
        ])
    
    def _initialize_static_filters(self):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        smooth = torch.full((3, 3), 1/9.0, dtype=torch.float32)
        
        with torch.no_grad():
            self.static_conv.weight[0] = sobel_x.repeat(3, 1, 1)
            self.static_conv.weight[1] = laplacian.repeat(3, 1, 1)
            self.static_conv.weight[2] = smooth.repeat(3, 1, 1)
        
        self.static_conv.weight.requires_grad = False

    def forward(self, x):
        # Feature extraction
        x_static = self.static_conv(x)
        x_trainable = self.trainable_conv(x)
        x1 = torch.cat([x_static, x_trainable], dim=1)
        x1 = F.leaky_relu(self.bn1(x1), 0.1)  # 128x64 spatial size
        
        # Layer 2 with skip connection
        x2 = self.conv2(x1)  # 64x32 spatial size
        x2_skip = self.skip_16_to_32(x1)
        x2 = x2 + x2_skip
        
        # Layer 3 with skip connection
        x3 = self.conv3(x2)  # 32x16 spatial size
        x3_skip = self.skip_32_to_64(x2)
        x3 = x3 + x3_skip
        
        # Layer 4 with skip connection and attention
        x4 = self.conv4(x3)  # 16x8 spatial size
        x4_skip = self.skip_64_to_128(x3)
        x4 = x4 + x4_skip
        x4 = self.cbam4(x4)
        
        # Layer 5 with skip connection and attention
        x5 = self.conv5(x4)  # 16x8 spatial size
        x5_skip = self.skip_128_to_256(x4)
        x5 = x5 + x5_skip
        x5 = self.cbam5(x5)
        x5 = self.dropout5(x5)
        
        # Global feature extraction
        global_feat = self.global_pool(x5)
        global_feat = self.flatten(global_feat)
        global_embedding = self.fc_global(global_feat)
        global_embedding = F.normalize(global_embedding, p=2, dim=1)
        
        # Part-based feature extraction (during training only)
        if self.training:
            # Get 3 vertical parts (head, torso, legs)
            parts = self.part_pools[1](x5)  # Shape: [B, C, 3, 1]
            part_features = []
            
            for i in range(6):
                part = parts[:, :, i, :]
                part = part.view(part.size(0), -1)
                part_emb = self.fc_parts[i](part)
                part_emb = F.normalize(part_emb, p=2, dim=1)
                part_features.append(part_emb)
            
            return global_embedding, part_features
        
        return global_embedding
    
    def forward_with_attention(self, x):
        """Forward pass that captures attention maps for visualization"""
        attention_maps = {}
        
        # Feature extraction
        x_static = self.static_conv(x)
        x_trainable = self.trainable_conv(x)
        x1 = torch.cat([x_static, x_trainable], dim=1)
        x1 = F.leaky_relu(self.bn1(x1), 0.1)
        
        # Layer 2 with skip connection
        x2 = self.conv2(x1)
        x2_skip = self.skip_16_to_32(x1)
        x2 = x2 + x2_skip
        
        # Layer 3 with skip connection
        x3 = self.conv3(x2)
        x3_skip = self.skip_32_to_64(x2)
        x3 = x3 + x3_skip
        
        # Layer 4 with skip connection and attention
        x4 = self.conv4(x3)
        x4_skip = self.skip_64_to_128(x3)
        x4 = x4 + x4_skip
        
        # Capture CBAM4 attention
        ca4 = self.cbam4.channel_attention(x4)
        x4_ca = x4 * ca4
        sa4_input = torch.cat([
            torch.mean(x4_ca, dim=1, keepdim=True),
            torch.max(x4_ca, dim=1, keepdim=True)[0]
        ], dim=1)
        sa4 = self.cbam4.spatial_attention(sa4_input)
        x4 = x4_ca * sa4
        attention_maps['cbam4_spatial'] = sa4
        attention_maps['cbam4_channel'] = ca4
        
        # Layer 5 with skip connection and attention
        x5 = self.conv5(x4)
        x5_skip = self.skip_128_to_256(x4)
        x5 = x5 + x5_skip
        
        # Capture CBAM5 attention
        ca5 = self.cbam5.channel_attention(x5)
        x5_ca = x5 * ca5
        sa5_input = torch.cat([
            torch.mean(x5_ca, dim=1, keepdim=True),
            torch.max(x5_ca, dim=1, keepdim=True)[0]
        ], dim=1)
        sa5 = self.cbam5.spatial_attention(sa5_input)
        x5 = x5_ca * sa5
        attention_maps['cbam5_spatial'] = sa5
        attention_maps['cbam5_channel'] = ca5
        attention_maps['feature_map'] = x5
        
        x5 = self.dropout5(x5)
        
        # Global feature extraction
        global_feat = self.global_pool(x5)
        global_feat = self.flatten(global_feat)
        global_embedding = self.fc_global(global_feat)
        global_embedding = F.normalize(global_embedding, p=2, dim=1)
        
        return global_embedding, attention_maps

# Compute metrics for retrieval results
def compute_metrics(matches, k_values=[1, 3, 5]):
    """
    Compute top-k accuracy for different k values
    
    Args:
        matches: List of boolean values indicating if retrievals were correct
        k_values: List of k values to compute accuracy for
        
    Returns:
        Dictionary of top-k accuracies
    """
    results = {}
    num_queries = len(matches)
    
    for k in k_values:
        if k <= len(matches[0]):
            # For each query, check if any of the top-k predictions are correct
            correct = sum(1 for query_matches in matches if any(query_matches[:k]))
            results[f'top_{k}'] = correct / num_queries
    
    return results

def compute_map(matches, similarities):
    """
    Compute mean Average Precision (mAP)
    
    Args:
        matches: List of lists, each containing boolean values for query matches
        similarities: List of lists, each containing similarity scores
        
    Returns:
        mAP score
    """
    aps = []
    for query_matches, query_sims in zip(matches, similarities):
        # Sort by similarity (descending)
        sorted_indices = np.argsort(query_sims)[::-1]
        sorted_matches = np.array(query_matches)[sorted_indices]
        
        # Compute Average Precision for this query
        num_relevant = np.sum(sorted_matches)
        if num_relevant == 0:
            aps.append(0)
            continue
            
        precision_at_k = []
        num_correct = 0
        for k, match in enumerate(sorted_matches):
            if match:
                num_correct += 1
                precision_at_k.append(num_correct / (k + 1))
        
        if precision_at_k:
            ap = np.mean(precision_at_k)
        else:
            ap = 0
        aps.append(ap)
    
    return np.mean(aps)

def compute_cmc_curve(matches, max_rank=50):
    """
    Compute Cumulative Matching Characteristics (CMC) curve
    
    Args:
        matches: List of lists, each containing boolean values for query matches
        max_rank: Maximum rank to compute
        
    Returns:
        CMC curve as numpy array
    """
    cmc = np.zeros(max_rank)
    num_queries = len(matches)
    
    for query_matches in matches:
        # Find the rank of the first correct match
        for rank, match in enumerate(query_matches[:max_rank]):
            if match:
                cmc[rank:] += 1
                break
    
    cmc = cmc / num_queries
    return cmc

def create_confusion_matrix(true_ids, predicted_ids, output_path):
    """
    Create confusion matrix for person ID predictions
    
    Args:
        true_ids: List of true person IDs
        predicted_ids: List of predicted person IDs (rank-1)
        output_path: Path to save the confusion matrix
    """
    # Get unique IDs
    unique_ids = sorted(list(set(true_ids + predicted_ids)))
    
    # Create confusion matrix
    cm = confusion_matrix(true_ids, predicted_ids, labels=unique_ids)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=unique_ids, yticklabels=unique_ids)
    plt.title('Person Re-ID Confusion Matrix (Rank-1 Predictions)')
    plt.xlabel('Predicted Person ID')
    plt.ylabel('True Person ID')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_embedding_visualization(embeddings, person_ids, output_dir, method='tsne'):
    """
    Create t-SNE or UMAP visualization of embeddings
    
    Args:
        embeddings: Numpy array of embeddings
        person_ids: List of person IDs
        output_dir: Directory to save visualizations
        method: 'tsne' or 'umap'
    """
    print(f"Creating {method.upper()} visualization...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:  # umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    
    # Reduce dimensionality
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    unique_ids = list(set(person_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    
    for i, pid in enumerate(unique_ids):
        mask = np.array(person_ids) == pid
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   c=[colors[i]], label=f'Person {pid}', alpha=0.7, s=20)
    
    plt.title(f'{method.upper()} Visualization of Person Re-ID Embeddings')
    plt.xlabel(f'{method.upper()}-1')
    plt.ylabel(f'{method.upper()}-2')
    
    # Only show legend if not too many people
    if len(unique_ids) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'embedding_visualization_{method}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{method.upper()} visualization saved to: {output_path}")

def create_attention_heatmap(original_img, attention_maps, output_path):
    """
    Create attention heatmap overlay on original image
    
    Args:
        original_img: PIL Image or numpy array
        attention_maps: Dictionary containing attention maps
        output_path: Path to save the heatmap
    """
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Attention Heatmaps', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # CBAM4 spatial attention
    if attention_maps['cbam4_spatial'] is not None:
        spatial_att = attention_maps['cbam4_spatial'][0, 0].cpu().numpy()
        spatial_att_resized = cv2.resize(spatial_att, (original_img.shape[1], original_img.shape[0]))
        
        axes[0, 1].imshow(original_img)
        axes[0, 1].imshow(spatial_att_resized, alpha=0.6, cmap='jet')
        axes[0, 1].set_title('CBAM4 Spatial Attention')
        axes[0, 1].axis('off')
    
    # CBAM5 spatial attention
    if attention_maps['cbam5_spatial'] is not None:
        spatial_att = attention_maps['cbam5_spatial'][0, 0].cpu().numpy()
        spatial_att_resized = cv2.resize(spatial_att, (original_img.shape[1], original_img.shape[0]))
        
        axes[0, 2].imshow(original_img)
        axes[0, 2].imshow(spatial_att_resized, alpha=0.6, cmap='jet')
        axes[0, 2].set_title('CBAM5 Spatial Attention')
        axes[0, 2].axis('off')
    
    # Feature map visualization (average across channels)
    if attention_maps['feature_map'] is not None:
        feature_map = attention_maps['feature_map'][0].cpu().numpy()
        feature_map_avg = np.mean(feature_map, axis=0)
        feature_map_resized = cv2.resize(feature_map_avg, (original_img.shape[1], original_img.shape[0]))
        
        axes[1, 0].imshow(original_img)
        axes[1, 0].imshow(feature_map_resized, alpha=0.6, cmap='hot')
        axes[1, 0].set_title('Feature Map Activation')
        axes[1, 0].axis('off')
    
    # Combined attention visualization
    if (attention_maps['cbam4_spatial'] is not None and 
        attention_maps['cbam5_spatial'] is not None):
        
        combined_att = (attention_maps['cbam4_spatial'][0, 0] + 
                       attention_maps['cbam5_spatial'][0, 0]) / 2
        combined_att = combined_att.cpu().numpy()
        combined_att_resized = cv2.resize(combined_att, (original_img.shape[1], original_img.shape[0]))
        
        axes[1, 1].imshow(original_img)
        axes[1, 1].imshow(combined_att_resized, alpha=0.6, cmap='jet')
        axes[1, 1].set_title('Combined Attention')
        axes[1, 1].axis('off')
    
    # Attention heatmap only
    if attention_maps['cbam5_spatial'] is not None:
        spatial_att = attention_maps['cbam5_spatial'][0, 0].cpu().numpy()
        axes[1, 2].imshow(spatial_att, cmap='jet')
        axes[1, 2].set_title('Pure Attention Map')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Create visualization grid for a query and its top matches (preserved original function)
def create_visualization_grid(query_path, gallery_paths, similarities, matches, output_path, top_k=5):
    """
    Create a visualization showing the query image and its top-k matches
    
    Args:
        query_path: Path to the query image
        gallery_paths: Paths to the top-k gallery images
        similarities: Similarity scores for each match
        matches: Boolean values indicating if each match is correct
        output_path: Path to save the visualization
        top_k: Number of top matches to display
    """
    # Number of images to display (1 query + top_k matches)
    n_imgs = min(top_k + 1, len(gallery_paths) + 1)
    
    plt.figure(figsize=(12, 3))
    
    # Display query image
    plt.subplot(1, n_imgs, 1)
    query_img = Image.open(query_path).convert('RGB')
    plt.imshow(query_img)
    plt.title("Query", fontsize=12)
    plt.axis('off')
    
    # Display top-k matches
    for i in range(top_k):
        if i < len(gallery_paths):
            plt.subplot(1, n_imgs, i + 2)
            
            match_img = Image.open(gallery_paths[i]).convert('RGB')
            plt.imshow(match_img)
            
            # Set border color based on match (green for correct, red for incorrect)
            border_color = 'green' if matches[i] else 'red'
            plt.gca().spines['top'].set_color(border_color)
            plt.gca().spines['bottom'].set_color(border_color)
            plt.gca().spines['left'].set_color(border_color)
            plt.gca().spines['right'].set_color(border_color)
            plt.gca().spines['top'].set_linewidth(5)
            plt.gca().spines['bottom'].set_linewidth(5)
            plt.gca().spines['left'].set_linewidth(5)
            plt.gca().spines['right'].set_linewidth(5)
            
            plt.title(f"Rank {i+1}\nSimilarity: {similarities[i]:.2f}", fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def build_faiss_index(embeddings):
    """
    Build FAISS index for efficient similarity search
    
    Args:
        embeddings: Normalized embeddings tensor
        
    Returns:
        FAISS index
    """
    # Convert to numpy and ensure float32
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    
    # Build index - using inner product since embeddings are normalized
    # Inner product of normalized vectors = cosine similarity
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    return index

def faiss_search(query_embedding, index, k=10):
    """
    Perform efficient similarity search using FAISS
    
    Args:
        query_embedding: Single query embedding
        index: FAISS index
        k: Number of top results to return
        
    Returns:
        similarities, indices
    """
    query_np = query_embedding.cpu().numpy().astype('float32').reshape(1, -1)
    similarities, indices = index.search(query_np, k)
    
    return similarities[0], indices[0]

def main():
    # Set paths - update these paths to match your actual file structure
    model_path = 'reid_model_3.2.pth'  # Actual model file found in workspace
    query_path = './Dataset/query'
    gallery_path = './Dataset/gallery'  
    output_root = 'Detailed Result Analysis - 2nd Attempt'  # Updated output directory name
    
    # Ensure we have results directories
    output_metrics_dir = os.path.join(output_root, 'metrics')
    output_visualizations_dir = os.path.join(output_root, 'visualizations')
    output_analysis_dir = os.path.join(output_root, 'advanced_analysis')
    output_attention_dir = os.path.join(output_root, 'attention_maps')
    
    os.makedirs(output_metrics_dir, exist_ok=True)
    os.makedirs(output_visualizations_dir, exist_ok=True)
    os.makedirs(output_analysis_dir, exist_ok=True)
    os.makedirs(output_attention_dir, exist_ok=True)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Transformations for test images (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((128, 64), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    
    # Load datasets
    query_dataset = ReIDDataset(query_path, transform=test_transform)
    gallery_dataset = ReIDDataset(gallery_path, transform=test_transform)
    
    print(f"Number of query images: {len(query_dataset)}")
    print(f"Number of gallery images: {len(gallery_dataset)}")
    
    if len(query_dataset) == 0 or len(gallery_dataset) == 0:
        print("ERROR: No images found in query or gallery directories!")
        print(f"Query path: {query_path}")
        print(f"Gallery path: {gallery_path}")
        return
    
    # Create dataloaders
    batch_size = 64  # Adjust based on available memory
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load model
    try:
        model = Person_ReID().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Please check if the model path is correct and the model is compatible.")
        return
    
    # Extract gallery embeddings
    print("Extracting gallery embeddings...")
    gallery_embeddings = []
    gallery_pids = []
    gallery_camids = []
    gallery_img_paths = []
    
    with torch.no_grad():
        for imgs, pids, camids, img_paths in tqdm(gallery_loader, desc="Processing gallery"):
            imgs = imgs.to(device)
            
            # Forward pass (use regular forward for speed)
            embeddings = model(imgs)
            
            # Store data
            gallery_embeddings.append(embeddings.cpu())
            gallery_pids.extend(pids.numpy().tolist())
            gallery_camids.extend(camids.numpy().tolist())
            gallery_img_paths.extend(img_paths)
    
    # Concatenate gallery embeddings
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    print(f"Extracted embeddings for {len(gallery_img_paths)} gallery images")
    
    # Build FAISS index for efficient similarity search
    print("Building FAISS index for efficient similarity search...")
    faiss_index = build_faiss_index(gallery_embeddings)
    
    # Extract query embeddings and perform retrieval
    print("Extracting query embeddings and performing retrieval...")
    
    # Prepare metrics lists
    all_matches = []
    all_similarities = []
    all_query_pids = []
    all_predicted_pids = []
    
    # Collect data for embedding visualization
    all_embeddings = []
    all_embedding_pids = []
    
    # Add gallery embeddings to visualization data
    all_embeddings.append(gallery_embeddings.numpy())
    all_embedding_pids.extend(gallery_pids)
    
    # Sample queries for attention visualization (limit to avoid too many files)
    attention_sample_indices = np.random.choice(len(query_dataset), 
                                              min(10, len(query_dataset)), 
                                              replace=False)
    attention_counter = 0
    
    # Process each query
    with torch.no_grad():
        for query_batch_idx, (query_imgs, query_pids, query_camids, query_img_paths) in enumerate(tqdm(query_loader, desc="Processing queries")):
            query_imgs = query_imgs.to(device)
            
            # Get query embeddings
            query_embeddings = model(query_imgs)
            
            # Add query embeddings to visualization data
            all_embeddings.append(query_embeddings.cpu().numpy())
            all_embedding_pids.extend(query_pids.numpy().tolist())
            
            # Process each query in the batch
            for i in range(len(query_imgs)):
                query_pid = query_pids[i].item()
                query_camid = query_camids[i].item()
                query_img_path = query_img_paths[i]
                query_embedding = query_embeddings[i].cpu()
                
                # Use FAISS for efficient similarity search
                similarities, indices = faiss_search(query_embeddings[i], faiss_index, k=50)
                
                # Filter out same camera results
                filtered_indices = []
                filtered_similarities = []
                for idx, sim in zip(indices, similarities):
                    if gallery_camids[idx] != query_camid:
                        filtered_indices.append(idx)
                        filtered_similarities.append(sim)
                        if len(filtered_indices) >= 10:  # Get top 10 after filtering
                            break
                
                # Get top matches info
                top_similarities = filtered_similarities
                top_pids = [gallery_pids[idx] for idx in filtered_indices]
                top_paths = [gallery_img_paths[idx] for idx in filtered_indices]
                
                # Determine if matches are correct (same person ID)
                match_results = [pid == query_pid for pid in top_pids]
                all_matches.append(match_results)
                all_similarities.append(top_similarities)
                all_query_pids.append(query_pid)
                all_predicted_pids.append(top_pids[0] if top_pids else -1)  # Rank-1 prediction
                
                # Create directory for this query
                query_id = f"query_{query_pid:04d}_cam_{query_camid:02d}_{query_batch_idx*batch_size + i:04d}"
                query_output_dir = os.path.join(output_visualizations_dir, query_id)
                os.makedirs(query_output_dir, exist_ok=True)
                
                # Copy query image to results directory
                shutil.copy(query_img_path, os.path.join(query_output_dir, "query.jpg"))
                
                # Create original visualization grid
                vis_path = os.path.join(query_output_dir, "topk_results.jpg")
                create_visualization_grid(
                    query_img_path,
                    top_paths,
                    top_similarities,
                    match_results,
                    vis_path,
                    top_k=5
                )
                
                # ============ ATTENTION HEATMAP GENERATION ============
                # TOGGLE: Comment/uncomment the block below to disable/enable heatmaps
                # Generate attention heatmaps for selected queries
                current_query_idx = query_batch_idx * len(query_imgs) + i
                if current_query_idx in attention_sample_indices and attention_counter < 10:
                    print(f"Generating attention heatmap for query {current_query_idx}")
                    
                    # Get attention maps using special forward pass
                    single_query = query_imgs[i:i+1]
                    embedding_with_attention, attention_maps = model.forward_with_attention(single_query)
                    
                    # Create attention heatmap
                    original_img = Image.open(query_img_path).convert('RGB')
                    attention_output_path = os.path.join(output_attention_dir, f"attention_{query_id}.png")
                    create_attention_heatmap(original_img, attention_maps, attention_output_path)
                    attention_counter += 1
                # =====================================================
    
    # Compute overall metrics
    print("Computing comprehensive metrics...")
    
    # Original top-k metrics
    metrics = compute_metrics(all_matches, k_values=[1, 3, 5, 10])
    
    # Compute mAP
    map_score = compute_map(all_matches, all_similarities)
    metrics['mAP'] = map_score
    
    # Compute CMC curve
    cmc_curve = compute_cmc_curve(all_matches, max_rank=50)
    
    # Create advanced visualizations and analysis
    print("Creating advanced analysis visualizations...")
    
    # 1. Confusion Matrix
    confusion_matrix_path = os.path.join(output_analysis_dir, "confusion_matrix.png")
    create_confusion_matrix(all_query_pids, all_predicted_pids, confusion_matrix_path)
    
    # 2. CMC Curve Plot
    plt.figure(figsize=(10, 6))
    ranks = np.arange(1, len(cmc_curve) + 1)
    plt.plot(ranks, cmc_curve, 'b-', linewidth=2, label='CMC Curve')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Matching Accuracy')
    plt.title('Cumulative Matching Characteristics (CMC) Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(1, 20)  # Show first 20 ranks
    plt.ylim(0, 1)
    cmc_plot_path = os.path.join(output_analysis_dir, "cmc_curve.png")
    plt.savefig(cmc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Embedding Visualizations
    print("Creating embedding visualizations...")
    all_embeddings_combined = np.vstack(all_embeddings)
    
    # Limit to reasonable number for visualization
    if len(all_embeddings_combined) > 1000:
        sample_indices = np.random.choice(len(all_embeddings_combined), 1000, replace=False)
        embedding_sample = all_embeddings_combined[sample_indices]
        pid_sample = [all_embedding_pids[i] for i in sample_indices]
    else:
        embedding_sample = all_embeddings_combined
        pid_sample = all_embedding_pids
    
    # Create t-SNE visualization
    create_embedding_visualization(embedding_sample, pid_sample, output_analysis_dir, method='tsne')
    
    # Create UMAP visualization
    create_embedding_visualization(embedding_sample, pid_sample, output_analysis_dir, method='umap')
    
    # 4. Performance Analysis Charts
    # Top-k accuracy bar chart
    plt.figure(figsize=(10, 6))
    k_values = [1, 3, 5, 10]
    accuracies = [metrics[f'top_{k}'] for k in k_values]
    bars = plt.bar([f'Top-{k}' for k in k_values], accuracies, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('Accuracy')
    plt.title('Top-K Retrieval Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    performance_chart_path = os.path.join(output_analysis_dir, "performance_chart.png")
    plt.savefig(performance_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive metrics to CSV
    metrics_file = os.path.join(output_metrics_dir, "comprehensive_metrics.csv")
    with open(metrics_file, 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for k, value in metrics.items():
            writer.writerow({'Metric': k, 'Value': f"{value:.4f}"})
        
        # Add CMC metrics
        for i, cmc_val in enumerate(cmc_curve[:10]):  # First 10 ranks
            writer.writerow({'Metric': f'CMC_Rank_{i+1}', 'Value': f"{cmc_val:.4f}"})
    
    # Save detailed CMC curve data
    cmc_data_file = os.path.join(output_metrics_dir, "cmc_curve_data.csv")
    with open(cmc_data_file, 'w', newline='') as csvfile:
        fieldnames = ['Rank', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rank, acc in enumerate(cmc_curve):
            writer.writerow({'Rank': rank + 1, 'Accuracy': f"{acc:.4f}"})
    
    # Create summary report
    summary_file = os.path.join(output_analysis_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Person Re-ID Retrieval Analysis Summary ===\n\n")
        f.write(f"Total Queries Processed: {len(all_matches)}\n")
        f.write(f"Total Gallery Images: {len(gallery_embeddings)}\n")
        f.write(f"Unique Person IDs in Query: {len(set(all_query_pids))}\n")
        f.write(f"Unique Person IDs in Gallery: {len(set(gallery_pids))}\n\n")
        
        f.write("=== Performance Metrics ===\n")
        for k, value in metrics.items():
            if k == 'mAP':
                f.write(f"mAP: {value:.4f}\n")
            else:
                f.write(f"{k.capitalize()}: {value:.4f}\n")
        
        f.write(f"\n=== Rank-based Performance (CMC) ===\n")
        for i in range(min(10, len(cmc_curve))):
            f.write(f"Rank-{i+1}: {cmc_curve[i]:.4f}\n")
        
        f.write(f"\n=== Analysis Files Generated ===\n")
        f.write(f"- Confusion Matrix: {confusion_matrix_path}\n")
        f.write(f"- CMC Curve: {cmc_plot_path}\n")
        f.write(f"- t-SNE Embedding Plot: {output_analysis_dir}/embedding_visualization_tsne.png\n")
        f.write(f"- UMAP Embedding Plot: {output_analysis_dir}/embedding_visualization_umap.png\n")
        f.write(f"- Performance Chart: {performance_chart_path}\n")
        f.write(f"- Attention Heatmaps: {output_attention_dir}/ ({attention_counter} samples)\n")
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE RETRIEVAL ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nDataset Information:")
    print(f"  Total queries processed: {len(all_matches)}")
    print(f"  Total gallery images: {len(gallery_embeddings)}")
    print(f"  Unique persons in queries: {len(set(all_query_pids))}")
    print(f"  Unique persons in gallery: {len(set(gallery_pids))}")
    
    print(f"\nRetrieval Metrics:")
    for k, value in metrics.items():
        if k == 'mAP':
            print(f"  mAP: {value:.4f}")
        else:
            print(f"  {k.capitalize()}: {value:.4f}")
    
    print(f"\nCMC Curve (First 10 ranks):")
    for i in range(min(10, len(cmc_curve))):
        print(f"  Rank-{i+1}: {cmc_curve[i]:.4f}")
    
    print(f"\nAdvanced Analysis Files Generated:")
    print(f"  • Confusion Matrix: {confusion_matrix_path}")
    print(f"  • CMC Curve Plot: {cmc_plot_path}")
    print(f"  • t-SNE Embedding Visualization: {output_analysis_dir}/embedding_visualization_tsne.png")
    print(f"  • UMAP Embedding Visualization: {output_analysis_dir}/embedding_visualization_umap.png")
    print(f"  • Performance Chart: {performance_chart_path}")
    print(f"  • Attention Heatmaps: {output_attention_dir}/ ({attention_counter} samples)")
    
    print(f"\nAll results saved to: {output_root}/")
    print("="*80)

if __name__ == "__main__":
    main() 