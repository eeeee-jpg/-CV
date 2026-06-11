import os
import cv2
import torch
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# ===================== 全局配置 =====================
# 检索相关配置
FEATURE_FILE = "retrieval_features.pkl"   # 检索特征库（retrieval_train.py 生成）
TOP_K = 10                                # 检索Top10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 文字检测相关配置
DET_MODEL_PATH = "text_det_model.pth"     # 文字检测模型（text_det_train.py 生成）
IMG_SIZE = (256, 256)

# 输出根目录，每组结果单独建文件夹
OUTPUT_ROOT = "visual_result"

# ===================== 1. 图像预处理（检索用，与训练一致） =====================
retrieval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 2. 文字检测模型（复用训练脚本结构） =====================
class TextDetNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32*64*64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.backbone(x)

# ===================== 3. 工具函数 =====================
def get_retrieval_model():
    """加载检索特征提取模型"""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(DEVICE)
    model.eval()
    return model

def extract_retrieval_feature(img_path, model):
    """提取单张图检索特征"""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"图片读取失败: {img_path} | {e}")
        return None
    img = retrieval_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(img)
    feat = feat.squeeze().cpu().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat

def detect_and_draw_box(img_path, save_path, det_model):
    """文字检测 + 绘制矩形框 + 保存图片"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"跳过无效图片: {img_path}")
        return False
    h, w = img.shape[:2]

    # 文字检测预处理
    img_resize = cv2.resize(img, IMG_SIZE)
    img_input = img_resize.transpose(2, 0, 1) / 255.0
    img_input = torch.from_numpy(img_input).float().unsqueeze(0).to(DEVICE)

    # 推理预测框
    with torch.no_grad():
        pred_box = det_model(img_input)[0].cpu().numpy()

    # 坐标还原 + 画红色框
    xmin = int(np.clip(pred_box[0] * w, 0, w))
    ymin = int(np.clip(pred_box[1] * h, 0, h))
    xmax = int(np.clip(pred_box[2] * w, 0, w))
    ymax = int(np.clip(pred_box[3] * h, 0, h))
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cv2.imwrite(save_path, img)
    return True

# ===================== 主逻辑 =====================
def main():
    # ========== 第一步：手动指定【当前要测试的单张查询图】 ==========
    # 这里为要测试的查询图完整路径/相对路径
    #QUERY_IMG_PATH = "query/fhy-4f6228a6c7fbd04aa26c0f777fadedc.jpg"
    #QUERY_IMG_PATH = "query/fhy-5e2877255e30fa15ed76f844332cf5fc.jpg"

    #QUERY_IMG_PATH = "query/jx-1nl3llnk.jpg"
    #QUERY_IMG_PATH = "query/jx-12bn3jol13.jpg"

    #QUERY_IMG_PATH = "query/kx-1vgyuhy.jpg"
    #QUERY_IMG_PATH = "query/kx-sklapp29080921.jpg"

    #QUERY_IMG_PATH = "query/mh-03e1ebaa-d9ca-4cb3-b0ed-770bec526323.png"
    #QUERY_IMG_PATH = "query/mh-4rft76yg.jpg"

    #QUERY_IMG_PATH = "query/nm-2egsd1x3e86s5f.jpg"
    #QUERY_IMG_PATH = "query/nm-76y4s0i1mvx7dzn.jpg"

   # QUERY_IMG_PATH = "query/sjz-2f3g4h5i6j7k.jpg"
    #QUERY_IMG_PATH = "query/sjz-4684144sada.jpg"

    #QUERY_IMG_PATH = "query/sy-89vcxuy456dft37beghirwjn.jpg"
    #QUERY_IMG_PATH = "query/sy-3784gfg8743fhr43.jpg"

    #QUERY_IMG_PATH = "query/tsg-04.jpg"
    #QUERY_IMG_PATH = "query/tsg-6xvb.png"

    #QUERY_IMG_PATH = "query/ty-1kln3l123lk1.png"
    #QUERY_IMG_PATH = "query/ty-902397dda144ad34c9ae6285dda20cf431ad859b.jpeg"

    QUERY_IMG_PATH = "query/yf-d6h1hgjdhwujd.jpg"
    #QUERY_IMG_PATH = "query/yf-dy789j1hdhsyghdjai.jpg"

    #QUERY_IMG_PATH = "query/yk-95ccd6cd9aa831373d98ed85e8b4b963.jpg"
    #QUERY_IMG_PATH = "query/yk-dwtutq13843780.jpg"

    #QUERY_IMG_PATH = "query/zx-2h3r4.jpg"
    #QUERY_IMG_PATH = "query/zx-308r0208w0rf0f.jpg"

    # 校验文件
    if not os.path.exists(QUERY_IMG_PATH):
        print(f"错误：查询图不存在 -> {QUERY_IMG_PATH}")
        return
    if not os.path.exists(FEATURE_FILE):
        print("错误：未找到检索特征库，请先运行 retrieval_train.py")
        return
    if not os.path.exists(DET_MODEL_PATH):
        print("错误：未找到文字检测模型，请先运行 text_det_train.py")
        return

    # ========== 第二步：加载模型 & 检索特征库 ==========
    # 加载检索模型 + 图库特征
    ret_model = get_retrieval_model()
    with open(FEATURE_FILE, "rb") as f:
        feature_data = pickle.load(f)
    base_features = feature_data["features"]
    base_img_paths = feature_data["img_paths"]

    # 加载文字检测模型
    det_model = TextDetNet().to(DEVICE)
    det_model.load_state_dict(torch.load(DET_MODEL_PATH, map_location=DEVICE))
    det_model.eval()

    # ========== 第三步：执行图片检索 Top10 ==========
    print(f"开始检索：{os.path.basename(QUERY_IMG_PATH)}")
    q_feat = extract_retrieval_feature(QUERY_IMG_PATH, ret_model)
    if q_feat is None:
        return

    # 相似度排序
    scores = np.dot(base_features, q_feat)
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    top10_img_paths = [base_img_paths[i] for i in top_idx]

    # ========== 第四步：创建当前组结果文件夹 ==========
    # 用查询图名称作为子文件夹，区分24组结果
    q_name = os.path.splitext(os.path.basename(QUERY_IMG_PATH))[0]
    group_save_dir = os.path.join(OUTPUT_ROOT, q_name)
    os.makedirs(group_save_dir, exist_ok=True)

    # ========== 第五步：10张检索图 全部做文字检测、画框保存 ==========

    # 保存Top10检索结果图
    for idx, img_path in enumerate(top10_img_paths, 1):
        save_path = os.path.join(group_save_dir, f"{idx:02d}_retrieve.jpg")
        detect_and_draw_box(img_path, save_path, det_model)
        print(f"已保存：检索图Top{idx} -> {save_path}")

    print(f"\n当前组可视化完成！结果目录：{group_save_dir}")

if __name__ == "__main__":
    main()