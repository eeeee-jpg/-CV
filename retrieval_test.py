import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import pickle

# ==================== 配置项 ====================
QUERY_DIR = "query"
FEATURE_LOAD_PATH = "retrieval_features.pkl"
OUTPUT_FILE = "retrieval_results/result_top60.txt"
TOPK = 60  # 固定输出TOP60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 图像预处理（和训练保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载特征模型
def get_extract_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(DEVICE)
    model.eval()
    return model

# 递归获取所有图片路径
def get_all_images(folder):
    img_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG')):
                img_paths.append(os.path.join(root, f))
    return img_paths

# 提取单张图片特征
def extract_feature(img_path, model):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"跳过无效查询图: {img_path} | {e}")
        return None

    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(img)
    feat = feat.squeeze().cpu().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat

def main():
    # 检查特征文件
    if not os.path.exists(FEATURE_LOAD_PATH):
        print("错误：请先运行 retrieval_train.py 生成特征库！")
        return

    # 加载图库特征
    with open(FEATURE_LOAD_PATH, "rb") as f:
        data = pickle.load(f)
    base_features = data["features"]
    base_img_paths = data["img_paths"]

    model = get_extract_model()
    query_list = get_all_images(QUERY_DIR)
    os.makedirs("retrieval_results", exist_ok=True)

    print(f"开始检索，共 {len(query_list)} 张查询图，每张输出TOP{TOPK}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for q_path in tqdm(query_list):
            # 提取查询图特征
            q_feat = extract_feature(q_path, model)
            if q_feat is None:
                continue

            # 相似度计算 + 排序
            sim_scores = np.dot(base_features, q_feat)
            top_idx = np.argsort(sim_scores)[::-1][:TOPK]
            top_full_paths = [base_img_paths[i] for i in top_idx]
            # 取文件名
            top_names = [os.path.basename(p) for p in top_full_paths]
            q_name = os.path.basename(q_path)

            # 写入
            f_out.write(f"查询图: {q_name}\n")
            f_out.write(f"TOP{TOPK} 结果:\n")
            for idx, name in enumerate(top_names, 1):
                if idx < 10:
                    f_out.write(f"   {idx}. {name}\n")
                else:
                    f_out.write(f"  {idx}. {name}\n")
            # 分割线
            f_out.write("------------------------------------------------------------\n")

    print(f"检索完成！结果已保存至：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()