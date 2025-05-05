import sqlite3

def init_db():
    # 连接到数据库（如果不存在则创建）
    with sqlite3.connect('detection.db') as conn:
        cursor = conn.cursor()
        
        # 创建检测记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            type TEXT NOT NULL,  -- 'image', 'video', 'realtime'
            accuracy REAL,
            processing_time REAL,
            object_count INTEGER DEFAULT 0,
            image_path TEXT
        )
        ''')
        
        # 创建检测结果表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            category TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (detection_id) REFERENCES detections (id)
        )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_time ON detections(detection_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_id ON detection_results(detection_id)')
        
        conn.commit()
        print("数据库初始化完成")

if __name__ == '__main__':
    init_db() 