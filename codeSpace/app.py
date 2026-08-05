import streamlit as st
import os
from inference import load_network, compute_similarity

# --- CẤU HÌNH ---
MODEL_PATH = "best.pth" # Sẽ tải từ Colab về sau
GALLERY_DIR = "my_gallery"

st.set_page_config(page_title="I-IRRA Search", layout="wide")

# --- LOAD MODEL (Cache) ---
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_network(MODEL_PATH)
    return None

model = get_model()

# --- GIAO DIỆN ---
st.title("🤖 I-IRRA: Tìm kiếm Người Tương tác")
st.markdown("Hệ thống tìm kiếm người qua mô tả với cơ chế **Alpha Fusion** và **Hội thoại**.")

# 1. Sidebar: Cấu hình tham số của Thầy
with st.sidebar:
    st.header("⚙️ Tham số Thuật toán")
    
    # SLIDER ALPHA QUAN TRỌNG
    alpha = st.slider(
        "Hệ số Alpha (Global - Local)", 
        0.0, 1.0, 0.6,
        help="0.0: Nhìn tổng quát (Màu sắc, dáng). 1.0: Nhìn chi tiết (Phụ kiện, logo)."
    )
    st.caption(f"Trạng thái: {'Thiên về Tổng quát' if alpha < 0.5 else 'Thiên về Chi tiết'}")
    
    top_k = st.number_input("Số lượng ảnh trả về", 5, 20, 10)

# 2. Quản lý Hội thoại (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Nếu là tin nhắn của bot (kết quả tìm kiếm), hiển thị ảnh
        if "images" in message:
            cols = st.columns(5)
            for i, (path, score) in enumerate(message["images"]):
                with cols[i % 5]:
                    st.image(path, caption=f"Score: {score:.3f}")

# 3. Ô nhập liệu (Chat Input)
if prompt := st.chat_input("Mô tả thêm để lọc kỹ hơn (VD: đeo giày, tóc ngắn...)"):
    
    # 1. Lưu câu prompt mới vào giao diện chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý tìm kiếm
    with st.chat_message("assistant"):
        if model is None:
            st.error("Chưa load được Model!")
        else:
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Đang suy luận dựa trên hội thoại...")
            
            # --- [FIX QUAN TRỌNG] TẠO CONTEXT ---
            # Gom tất cả các câu user đã nói từ đầu đến giờ lại thành 1 đoạn văn
            full_context_query = ""
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    full_context_query += msg["content"] + ", "
            
            # Debug: In ra để bạn thấy nó đang tìm cái gì (hiển thị nhỏ ở dưới)
            st.caption(f"🤖 Đang tìm kiếm với query tổng hợp: *'{full_context_query}'*")
            
            # Lấy danh sách ảnh
            gallery_imgs = [os.path.join(GALLERY_DIR, f) for f in os.listdir(GALLERY_DIR)]
            
            # Gửi câu query ĐẦY ĐỦ (Full Context) vào model
            scores, paths = compute_similarity(model, full_context_query, gallery_imgs, alpha)
            
            # Sắp xếp
            results = sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Hiển thị kết quả
            if len(results) == 0:
                message_placeholder.markdown("Không tìm thấy ai phù hợp.")
            else:
                message_placeholder.markdown(f"Kết quả cho: **{prompt}** (trong ngữ cảnh cũ)")
                cols = st.columns(5)
                img_data_for_history = []
                
                for i, (path, score) in enumerate(results):
                    with cols[i % 5]:
                        st.image(path, caption=f"Score: {score:.3f}")
                    img_data_for_history.append((path, score))
                
                # Lưu kết quả vào lịch sử
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Kết quả lọc thêm: '{prompt}'",
                    "images": img_data_for_history
                })