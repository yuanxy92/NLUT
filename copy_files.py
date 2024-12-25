import os
import shutil

# 设置源文件夹和目标文件夹路径
source_folder = '/data/hdd/Data/Metalens/MetalensSR_20241220/hr_images'
destination_folder = '/data/hdd/Data/Metalens/MetalensSR_20241220/hr_images_first_1000'

# 获取源文件夹中的所有文件
files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
sorted(files)

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 复制前1000张图像
for i, file in enumerate(files[:1000]):
    source_path = os.path.join(source_folder, file)
    destination_path = os.path.join(destination_folder, file)
    shutil.copy(source_path, destination_path)

print("成功复制了前1000张图像。")
