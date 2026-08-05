import os
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import shutil
from tqdm import tqdm

from inference import load_network, val_transforms

MODEL_PATH = "C:/graduate_assigment/codeSpace/best.pth"  # Đường dẫn đến model đã train của bạn
INPUT_DIR = "codeSpace/dataset/ThuVien"      # Thư mục chứa các folder person_1, person_2... từ bước YOLO
OUTPUT_DIR = "dataset/clustered_persons"   # Thư mục lưu kết quả sau khi đã gộp ID
SIMILARITY_THRESHOLD = 0.85        # Ngưỡng giống nhau (0.0 -> 1.0). Càng cao càng khắt khe.

device = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ Đang tải mô hình IRRA...")
model = load_network(MODEL_PATH)
if model is None:
    print("❌ Lỗi: Không tải được mô hình!")
    exit()

# --- 2. HÀM TRÍCH XUẤT ĐẶC TRƯNG BẰNG IRRA ---
def get_folder_feature_irra(folder_path):
    valid_images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    if not valid_images:
        return None
        
    features = []
    # Xử lý từng ảnh để tránh tràn RAM (Out of Memory)
    with torch.no_grad():
        for img_name in valid_images:
            img_path = os.path.join(folder_path, img_name)
            try:
                # Dùng val_transforms của IRRA
                img_tensor = val_transforms(Image.open(img_path).convert('RGB'))
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Trích xuất đặc trưng bằng bộ mã hóa ảnh của IRRA
                img_feat = model.encode_image(img_tensor)
                
                # Chuẩn hóa L2
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                features.append(img_feat.cpu().numpy())
            except Exception as e:
                pass
                
    if not features: return None
        
    # Lấy trung bình cộng các ảnh để tạo vector đại diện cho 1 ID
    folder_feature = np.mean(features, axis=0)
    folder_feature = folder_feature / np.linalg.norm(folder_feature)
    return folder_feature.squeeze()

# --- 3. QUÉT DỮ LIỆU ---
folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
folder_paths = []
all_features = []

print("🔍 Đang trích xuất đặc trưng nhận diện (ReID)...")
for folder in tqdm(folders):
    path = os.path.join(INPUT_DIR, folder)
    feat = get_folder_feature_irra(path)
    if feat is not None:
        all_features.append(feat)
        folder_paths.append(path)

all_features = np.array(all_features)

# --- 4. GỘP CỤM ---
print("🧠 Đang phân tích cụm...")
distance_threshold = 1.0 - SIMILARITY_THRESHOLD

clustering = AgglomerativeClustering(
    n_clusters=None, 
    metric='cosine', 
    linkage='average', 
    distance_threshold=distance_threshold
)
cluster_labels = clustering.fit_predict(all_features)

# --- 5. LƯU KẾT QUẢ ---
print("📁 Đang tái cấu trúc thư mục...")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR) # Xóa folder cũ nếu có
os.makedirs(OUTPUT_DIR, exist_ok=True)

for new_id, old_folder_path in zip(cluster_labels, folder_paths):
    new_person_dir = os.path.join(OUTPUT_DIR, f"person_ID_{new_id}")
    os.makedirs(new_person_dir, exist_ok=True)
    
    for img_name in os.listdir(old_folder_path):
        src = os.path.join(old_folder_path, img_name)
        dst = os.path.join(new_person_dir, f"{os.path.basename(old_folder_path)}_{img_name}")
        shutil.copy2(src, dst)

print(f"✅ HOÀN THÀNH! Đã gộp 57 thư mục gốc thành {len(set(cluster_labels))} ID duy nhất.")