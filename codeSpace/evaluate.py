import torch
import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm
from prettytable import PrettyTable
import torch.nn.functional as F

# Tận dụng lại các hàm đã sửa chuẩn ở inference.py
from inference import load_network, tokenize, val_transforms

# --- CẤU HÌNH ---
MODEL_PATH = "C:/graduate_assigment/codeSpace/best.pth"
DATA_ROOT = "/dataset/RSTPREID"  # Đường dẫn ra folder dataset gốc của bạn
JSON_PATH = os.path.join(DATA_ROOT, "data_captions.json")
IMG_ROOT = os.path.join(DATA_ROOT, "imgs")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_test_data(json_path):
    print(f"📂 Đang đọc dữ liệu từ: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Chỉ lấy tập TEST
    test_data = [d for d in data if d['split'] == 'test']
    
    # Tách Image paths, Captions và ID
    img_paths = []
    all_captions = []
    img_ids = []
    caption_ids = []
    
    # RSTPReid cấu trúc: mỗi ảnh có nhiều caption
    for item in test_data:
        path = os.path.join(IMG_ROOT, item['file_path'])
        pid = int(item['id'])
        
        # Lưu thông tin ảnh (Gallery)
        img_paths.append(path)
        img_ids.append(pid)
        
        # Lưu thông tin caption (Query)
        for cap in item['captions']:
            all_captions.append(cap)
            caption_ids.append(pid)
            
    return img_paths, img_ids, all_captions, caption_ids

def extract_img_feats(model, img_paths):
    feats = []
    model.eval()
    print("⏳ Đang trích xuất đặc trưng Ảnh (Gallery)...")
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), BATCH_SIZE)):
            batch_paths = img_paths[i : i+BATCH_SIZE]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    imgs.append(val_transforms(img))
                except:
                    # Nếu lỗi ảnh thì tạo ảnh đen bù vào để không lệch index
                    imgs.append(torch.zeros(3, 384, 128))
            
            if not imgs: continue
            
            imgs = torch.stack(imgs).to(DEVICE)
            feat = model.encode_image(imgs)
            feat = F.normalize(feat, p=2, dim=1) # Chuẩn hóa L2
            feats.append(feat.cpu())
            
    return torch.cat(feats, dim=0)

def extract_text_feats(model, captions):
    feats = []
    model.eval()
    print("⏳ Đang trích xuất đặc trưng Văn bản (Query)...")
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), BATCH_SIZE)):
            batch_caps = captions[i : i+BATCH_SIZE]
            tokens = tokenize(batch_caps, context_length=77).to(DEVICE)
            
            feat = model.encode_text(tokens)
            feat = F.normalize(feat, p=2, dim=1) # Chuẩn hóa L2
            feats.append(feat.cpu())
            
    return torch.cat(feats, dim=0)

def compute_metrics(img_feats, text_feats, img_ids, cap_ids):
    print("🧮 Đang tính toán ma trận khoảng cách...")
    
    # Tính Cosine Similarity: Text (Query) x Image (Gallery)
    # [Num_Text, Dim] x [Dim, Num_Img] = [Num_Text, Num_Img]
    sims = torch.mm(text_feats, img_feats.t())
    
    # Chuyển sang numpy để tính toán metric
    sims = sims.numpy()
    img_ids = np.array(img_ids)
    cap_ids = np.array(cap_ids)
    
    num_queries = len(cap_ids)
    
    all_cmc = []
    all_ap = []
    
    print("📊 Đang tổng hợp kết quả R1, R5, R10...")
    for i in tqdm(range(num_queries)):
        query_id = cap_ids[i]
        query_sim = sims[i]
        
        # Sắp xếp kết quả từ cao xuống thấp (giảm dần độ tương đồng)
        indices = np.argsort(-query_sim)
        matches = (img_ids[indices] == query_id) # True/False array
        
        if not np.any(matches): continue
            
        # Tính CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:10]) # Lấy top 10
        
        # Tính Average Precision (AP)
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (j + 1.) for j, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        ap = tmp_cmc.sum() / num_rel
        all_ap.append(ap)

    all_cmc = np.array(all_cmc).astype(np.float32)
    all_cmc = all_cmc.mean(axis=0)
    mAP = np.mean(all_ap)

    return all_cmc, mAP

def main():
    # 1. Load Model
    model = load_network(MODEL_PATH)
    if model is None:
        print("❌ Không tìm thấy model. Hãy đảm bảo best.pth ở đúng chỗ.")
        return

    # 2. Load Data
    # Kiểm tra đường dẫn dataset
    if not os.path.exists(JSON_PATH):
        print(f"❌ Không tìm thấy file json tại: {JSON_PATH}")
        print("👉 Hãy sửa biến DATA_ROOT ở đầu file evaluate.py cho đúng folder dataset của bạn.")
        return
        
    img_paths, img_ids, captions, caption_ids = get_test_data(JSON_PATH)
    print(f"✅ Đã load: {len(img_paths)} ảnh gallery và {len(captions)} câu query.")

    # 3. Extract Features
    img_feats = extract_img_feats(model, img_paths)
    text_feats = extract_text_feats(model, captions)

    # 4. Evaluate
    cmc, mAP = compute_metrics(img_feats, text_feats, img_ids, caption_ids)

    # 5. Hiển thị bảng kết quả đẹp
    table = PrettyTable()
    table.field_names = ["Metrics", "Kết quả (%)"]
    table.add_row(["Rank-1", f"{cmc[0]*100:.2f}"])
    table.add_row(["Rank-5", f"{cmc[4]*100:.2f}"])
    table.add_row(["Rank-10", f"{cmc[9]*100:.2f}"])
    table.add_row(["mAP", f"{mAP*100:.2f}"])
    
    print("\n" + "="*40)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH IRRA 🏆")
    print("="*40)
    print(table)
    print("="*40)
    
    # Lưu kết quả ra file txt để làm báo cáo
    with open("ket_qua_metric.txt", "w") as f:
        f.write(str(table))
    print("✅ Đã lưu kết quả vào file 'ket_qua_metric.txt'")

if __name__ == "__main__":
    main()