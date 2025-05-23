import os

# 获取同级的一个文件夹static
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

uploads_dir = os.path.join(static_dir, "uploads")
