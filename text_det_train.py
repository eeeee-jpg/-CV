import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===================== 配置 =====================
DATA_DIR = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 10
MODEL_SAVE_PATH = "text_det_model.pth"

# ===================== 1. 自定义数据集：读取图片+JSON标注 =====================
class TextDetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_list = []
        self.ann_list = []
        # 遍历目录，匹配 图片 + 同名json
        for file in os.listdir(data_dir):
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(data_dir, file)
                json_path = os.path.join(data_dir, os.path.splitext(file)[0] + ".json")
                if os.path.exists(json_path):
                    self.img_list.append(img_path)
                    self.ann_list.append(json_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        ann_path = self.ann_list[idx]

        # 读取图片
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img = cv2.resize(img, (256, 256))
        img = img.transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(img).float()

        # 读取JSON标注框 (取第一个文字框做训练演示，多框可自行扩展)
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        # 适配常规标注格式：points = [[x1,y1],[x2,y2]...] 矩形框
        boxes = []
        if "shapes" in ann:
            for shape in ann["shapes"]:
                pts = np.array(shape["points"])
                xmin, ymin = pts.min(axis=0)
                xmax, ymax = pts.max(axis=0)
                # 归一化坐标 [0,1]
                boxes.append([xmin/w, ymin/h, xmax/w, ymax/h])

        # 固定取第一个框，实验简易实现
        if boxes:
            box = boxes[0]
        else:
            box = [0,0,0,0]
        box = torch.tensor(box, dtype=torch.float32)
        return img, box

# ===================== 2. 简易文字检测模型 =====================
class TextDetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*64*64, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 输出 xmin,ymin,xmax,ymax 四个坐标
        )

    def forward(self, x):
        return self.backbone(x)

# ===================== 3. 训练流程 =====================
def train():
    # 数据集 & 加载器
    dataset = TextDetDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"训练集图片数量: {len(dataset)}")

    # 模型、损失、优化器
    model = TextDetNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 开始训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, boxes in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            boxes = boxes.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, boxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"训练完成，模型已保存至 {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()