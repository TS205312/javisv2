import sys
import matplotlib.pyplot as plt
import matplotlib

# Cấu hình encoding cho stdout để hỗ trợ tiếng Việt trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Dùng backend không interaktif để lưu file
matplotlib.use('Agg')

# Dữ liệu mẫu
labels = ['Th1', 'Th2', 'Th3', 'Th4']
values = [15, 20, 12, 25]

# Vẽ biểu đồ cột
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color='steelblue')
plt.title('Ví dụ Biểu đồ Cột')
plt.xlabel('Tháng')
plt.ylabel('Giá trị')

# Lưu file
plt.savefig('output.png')
print('Đã lưu output.png')