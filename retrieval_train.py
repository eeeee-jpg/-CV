import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import pickle

# 配置
BASE_DIR = "base"
FEATURE_SAVE_PATH = "retrieval_features.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载特征提取模型
def get_extract_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(DEVICE)
    model.eval()
    return model

# 递归获取目录下所有图片
def get_all_images(folder):
    img_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG')):
                img_paths.append(os.path.join(root, f))
    return img_paths

# 单张图片特征提取（跳过损坏图片）
def extract_feature(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(img)
    feat = feat.squeeze().cpu().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat

def main():
    model = get_extract_model()
    img_paths = get_all_images(BASE_DIR)
    features = []
    img_names = []

    print(f"开始提取 base 图库特征，总计待处理: {len(img_paths)} 张图片")
    for path in tqdm(img_paths):
        feat = extract_feature(path, model)
        if feat is not None:
            features.append(feat)
            img_names.append(path)

    # 保存特征与对应图片路径
    save_data = {
        "features": np.array(features),
        "img_paths": img_names
    }
    with open(FEATURE_SAVE_PATH, "wb") as f:
        pickle.dump(save_data, f)

    print(f"特征库保存完成！有效图片数: {len(img_names)}")
    print(f"特征文件路径: {FEATURE_SAVE_PATH}")

if __name__ == "__main__":
    main()