import sqlite3
import os
from app import app
from config import Config

def update_database():
    """直接更新SQLite数据库结构，添加is_admin列并设置admin用户"""
    
    # 获取数据库文件路径
    db_path = os.path.join(app.root_path, 'instance/detection_records.db')
    
    print(f"连接数据库: {db_path}")
    
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 检查是否已有is_admin列
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'is_admin' not in column_names:
            print("添加is_admin列到users表...")
            cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
            conn.commit()
            print("is_admin列添加成功")
        else:
            print("is_admin列已存在，无需添加")
        
        # 将admin用户设置为管理员
        print("设置admin用户为管理员...")
        cursor.execute("UPDATE users SET is_admin = 1 WHERE username = 'admin'")
        admin_count = cursor.rowcount
        conn.commit()
        
        if admin_count > 0:
            print(f"已将 {admin_count} 个admin用户设置为管理员")
        else:
            print("未找到admin用户，请确认用户名是否正确")
        
        print("数据库更新完成")
    
    except Exception as e:
        print(f"更新数据库时出错: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    update_database() 