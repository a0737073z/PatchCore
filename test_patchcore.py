import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import argparse
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

# 定義簡化版的模型（取得 feature map 而非分類結果）
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        full_model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        self.backbone = torch.nn.Sequential(*list(full_model.children())[:-2])  # 去掉 avgpool 和 fc

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)  # shape: [1, 2048, H, W]
        return x

def load_model(ckpt_path=None):
    model = SimpleModel()
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model

# 預處理圖片
def preprocess_image(image_path, input_size=224):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

# 簡單 KNN 類別
class KNN:
    def __init__(self, memory_bank, k=9):
        self.memory_bank = F.normalize(memory_bank, dim=1)
        self.k = k

    def __call__(self, query):
        query = F.normalize(query, dim=1)
        similarity = torch.mm(query, self.memory_bank.t())
        scores, _ = torch.topk(similarity, self.k, dim=1, largest=True, sorted=True)
        return 1 - scores, _

def detect_anomaly(model, image_tensor, embedding_coreset, n_neighbors=9):
    with torch.no_grad():
        features = model(image_tensor.cuda())  # [1, 2048, H, W]
        h, w = features.shape[2], features.shape[3]

        # 不轉 numpy，直接保留 tensor
        embedding = features.mean(dim=[2, 3])  # shape: [1, 2048]

        knn = KNN(torch.from_numpy(embedding_coreset).cuda(), k=n_neighbors)
        score_patches = knn(embedding)[0].cpu().detach().numpy()

        anomaly_score = score_patches[:, 0].reshape((h, w))
    return anomaly_score
# 儲存結果
def save_result(image_path, anomaly_map, output_dir):
    img = cv2.imread(image_path)
    anomaly_map_resized = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]))
    anomaly_map_blurred = gaussian_filter(anomaly_map_resized, sigma=4)
    
    # 異常熱圖
    heatmap = cv2.applyColorMap((anomaly_map_blurred * 255 / anomaly_map_blurred.max()).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.putText(overlay, f"Anomaly Score: {np.max(anomaly_map_blurred):.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(output_path, overlay)

# 處理資料夾
def process_folder(folder_path, model, embedding_coreset, output_dir, input_size=224):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_tensor = preprocess_image(img_path, input_size).cuda()
            anomaly_map = detect_anomaly(model, img_tensor, embedding_coreset)
            save_result(img_path, anomaly_map, output_dir)
            print(f"Processed: {filename} | Anomaly Score: {np.max(anomaly_map):.4f}")

# 主流程
def main(args):
    print(f"Input Path: {args.input_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Checkpoint Path: {args.ckpt_path}")
    print(f"Embedding Coreset Path: {args.embedding_coreset}")
    print(f"Input Size: {args.input_size}")
    
    model = load_model(args.ckpt_path).cuda()

    with open(args.embedding_coreset, 'rb') as f:
        embedding_coreset = pickle.load(f)

    if os.path.isdir(args.input_path):
        process_folder(args.input_path, model, embedding_coreset, args.output_dir, input_size=args.input_size)
    else:
        img_tensor = preprocess_image(args.input_path, input_size=args.input_size).cuda()
        anomaly_map = detect_anomaly(model, img_tensor, embedding_coreset)
        save_result(args.input_path, anomaly_map, args.output_dir)
        print(f"Processed: {args.input_path} | Anomaly Score: {np.max(anomaly_map):.4f}")

# 測試引數
if __name__ == '__main__':
    args = argparse.Namespace(
        input_path="C:/Users/user/Desktop/0424/val/NG/NA_B_05.png",
        output_dir="C:/Users/user/Desktop/0424/output",
        ckpt_path="C:/Users/user/Desktop/project/code/SA-PatchCore/outputs/lightning_logs/version_0/checkpoints/epoch=0-step=407.ckpt",
        embedding_coreset="C:/Users/user/Desktop/project/code/SA-PatchCore/outputs/embeddings/embedding.pickle",
        input_size=224
    )
    
    main(args)
