import os
import sys

# Lấy đường dẫn hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
irra_path = os.path.join(current_dir, 'IRRA')
model_path = os.path.join(irra_path, 'model')

print("--- KẾT QUẢ KIỂM TRA ---")
print(f"1. Đường dẫn dự án: {current_dir}")
print(f"2. Đường dẫn IRRA mong đợi: {irra_path}")

# Kiểm tra folder IRRA
if os.path.exists(irra_path):
    print("   ✅ Folder IRRA: CÓ")
else:
    print("   ❌ Folder IRRA: KHÔNG CÓ (Bạn cần tạo folder tên là IRRA)")

# Kiểm tra folder model bên trong
if os.path.exists(model_path):
    print("   ✅ Folder model: CÓ")
    # Kiểm tra file __init__.py (QUAN TRỌNG NHẤT)
    init_file = os.path.join(model_path, '__init__.py')
    if os.path.exists(init_file):
        print("   ✅ File model/__init__.py: CÓ (Python có thể import)")
    else:
        print("   ❌ File model/__init__.py: KHÔNG CÓ (Đây là nguyên nhân lỗi!)")
else:
    print("   ❌ Folder model: KHÔNG CÓ (Bên trong IRRA thiếu folder model)")

print("-" * 30)