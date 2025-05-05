import sqlite3
import os

def fix_database():
    """修复数据库表结构"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'detection_records.db')
    sql_path = os.path.join(current_dir, 'init_db.sql')
    
    try:
        # 如果数据库不存在，创建一个新的
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 读取 SQL 脚本
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # 执行 SQL 脚本
        cursor.executescript(sql_script)
        
        # 提交更改
        conn.commit()
        print("数据库修复成功！")
        print(f"数据库路径: {db_path}")
        
    except Exception as e:
        print(f"修复数据库时出错: {str(e)}")
        conn.rollback()
    finally:
        # 关闭连接
        cursor.close()
        conn.close()

if __name__ == '__main__':
    fix_database() 