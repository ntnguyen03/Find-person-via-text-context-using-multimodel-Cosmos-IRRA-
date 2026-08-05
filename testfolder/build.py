 from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

# --- [PHẪU THUẬT] THÊM MÃ NGUỒN COSMOS ---
import sys
sys.path.insert(0, '/kaggle/working/cosmos/src')
import open_clip
from huggingface_hub import hf_hub_download
# -----------------------------------------

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        # --- [PHẪU THUẬT] RÚT CLIP - CẮM COSMOS ---
        print("⏳ Đang khởi tạo 'thể xác' ViT-B-16...")
        self.base_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained=None)
        
        print("⬇️ Đang tải trọng số COSMOS từ HuggingFace...")
        ckpt_path = hf_hub_download(repo_id="sankim2/cosmos", filename="cosmos_vitb16_cc3m.pt")
        
        print("🧠 Đang cấy trọng số vào mô hình...")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        self.base_model.load_state_dict(state_dict, strict=False)

        self.embed_dim = self.base_model.text_projection.shape[1]
        print(f"✅ Đã cấy COSMOS thành công! (Embed Dimension: {self.embed_dim})")
        # ------------------------------------------

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        # Đã lược bỏ phần khởi tạo MLM cho gọn vì bài test này không dùng đến

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
    
    # --- PHỄU LỌC THÔNG MINH XỬ LÝ DICT CỦA COSMOS ---
    def _extract_img_feats(self, feats):
        if isinstance(feats, dict):
            g_feat, s_feat = None, None
            for k, v in feats.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim == 2 and g_feat is None: g_feat = v
                    elif v.ndim == 3 and s_feat is None: s_feat = v
            if g_feat is None and s_feat is not None: g_feat = s_feat[:, 0, :]
            return g_feat, s_feat
        elif isinstance(feats, torch.Tensor):
            if feats.ndim == 3: return feats[:, 0, :], feats
            return feats, None
        return feats, None

    def _extract_txt_feats(self, feats, caption_ids):
        if isinstance(feats, dict):
            g_feat, s_feat = None, None
            for k, v in feats.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim == 2 and g_feat is None: g_feat = v
                    elif v.ndim == 3 and s_feat is None: s_feat = v
            if g_feat is None and s_feat is not None:
                g_feat = s_feat[torch.arange(s_feat.shape[0]), caption_ids.argmax(dim=-1)]
            return g_feat, s_feat
        elif isinstance(feats, torch.Tensor):
            if feats.ndim == 3:
                return feats[torch.arange(feats.shape[0]), caption_ids.argmax(dim=-1)], feats
            return feats, None
        return feats, None
    # ------------------------------------------------

    def encode_image(self, image):
        raw_feats = self.base_model.encode_image(image)
        g_feat, _ = self._extract_img_feats(raw_feats)
        return g_feat.float()

    def encode_text(self, text):
        raw_feats = self.base_model.encode_text(text)
        g_feat, _ = self._extract_txt_feats(raw_feats, text)
        return g_feat.float()

    def forward(self, batch):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids']

        # Chạy qua COSMOS
        raw_i_feats = self.base_model.encode_image(images)
        raw_t_feats = self.base_model.encode_text(caption_ids)

        # Lọc lấy tensor bằng phễu
        i_feats, _ = self._extract_img_feats(raw_i_feats)
        t_feats, _ = self._extract_txt_feats(raw_t_feats, caption_ids)

        i_feats = i_feats.float()
        t_feats = t_feats.float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats).float()
            text_logits = self.classifier(t_feats).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)
            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        return ret

def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    return model