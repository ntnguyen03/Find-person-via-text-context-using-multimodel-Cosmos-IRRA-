import sys
import os
import torch
from torchvision import transforms
from PIL import Image

# --- 1. CẤU HÌNH ĐƯỜNG DẪN (HACK) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
irra_root = os.path.join(current_dir, 'IRRA')

# Thêm folder IRRA vào đầu danh sách tìm kiếm
if irra_root not in sys.path:
    sys.path.insert(0, irra_root)

# --- 2. IMPORT MODEL ---
try:
    from IRRA.model import build_model
    print("✅ Import IRRA.model thành công (Fallback)!")
except Exception as e2:
    print("❌ Không thể import model. Kiểm tra lại cấu trúc thư mục.")
    raise e2

# --- 3. XỬ LÝ CLIP ---
try:
    import clip
except ImportError:
    local_clip_path = os.path.join(current_dir, 'clip')
    if local_clip_path not in sys.path:
        sys.path.append(local_clip_path)
    import clip

def tokenize(text, context_length=77):
    return clip.tokenize(text, context_length=context_length, truncate=True)

# --- 4. LOGIC CHÍNH ---
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_network(model_path):
    print(f"⏳ Đang tải model từ {model_path}...")
    
    # --- CẤU HÌNH THAM SỐ (ARGS) ---
    class Args:
        pass
    args = Args()
    
    # == CÁC THAM SỐ CƠ BẢN ==
    args.pretrain_choice = 'ViT-B/16'
    args.img_size = (384, 128)
    args.stride_size = 16
    args.num_instance = 4
    args.vocab_size = 49408
    args.context_length = 77
    args.loss_names = 'sdm+id+mlm'
    args.device = device
    
    # == CÁC THAM SỐ KHẮC PHỤC LỖI (QUAN TRỌNG) ==
    args.temperature = 0.02
    args.height = 384
    args.width = 128
    
    # [FIX] Tham số kiến trúc Transformer (cmt_depth)
    args.cmt_depth = 4      # Độ sâu của Cross-Modal Transformer (Mặc định IRRA là 4)
    
    # [FIX] Các tham số Masked Language Modeling (phòng hờ lỗi tiếp theo)
    args.masked_token_rate = 0.8
    args.masked_token_unchanged_rate = 0.1
    args.drop_path_rate = 0.1
    args.mlm_loss_weight = 1.0
    args.id_loss_weight = 1.0
    
    # Xây dựng mô hình
    try:
        model = build_model(args, num_classes=3701) # 3701 là số class RSTPReid
    except Exception as e:
        print(f"❌ Lỗi khi build_model (trong hàm load): {e}")
        # In ra chi tiết tham số args để debug nếu cần
        print(f"Args hiện tại: {vars(args)}")
        raise e

    # Load trọng số
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)
            model.to(device)
            model.eval()
            print("✅ Model đã load weights thành công!")
            return model
        except Exception as e:
            print(f"❌ Lỗi khi load weights: {e}")
            return None
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy file {model_path}")
        # Trả về None nhưng không crash app để người dùng còn thấy giao diện
        return None

# Tiền xử lý ảnh
val_transforms = transforms.Compose([
    transforms.Resize((384, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
])

def compute_similarity(model, query_text, gallery_images, alpha=0.5):
    if model is None: return [], []

    text_token = tokenize(query_text, context_length=77).to(device)
    
    img_tensors = []
    valid_paths = []
    
    # print(f"🔍 Đang xử lý {len(gallery_images)} ảnh...") # Tắt log cho đỡ rối
    
    for path in gallery_images:
        try:
            img = Image.open(path).convert('RGB')
            img_tensors.append(val_transforms(img))
            valid_paths.append(path)
        except Exception as e:
            pass
            
    if not img_tensors: 
        return [], []
    
    img_batch = torch.stack(img_tensors).to(device)

    with torch.no_grad():
        image_feats = model.encode_image(img_batch)
        text_feats = model.encode_text(text_token)

        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        global_score = image_feats @ text_feats.t()
        global_score = global_score.squeeze()

        local_score = torch.pow((global_score + 1) / 2, 3)
        final_score = alpha * local_score + (1 - alpha) * global_score
        
    if final_score.ndim == 0:
        final_score = final_score.unsqueeze(0)
        
    return final_score.cpu().numpy(), valid_paths