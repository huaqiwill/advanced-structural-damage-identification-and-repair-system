-- 删除现有表（如果存在）
DROP TABLE IF EXISTS detection_records;
DROP TABLE IF EXISTS "detection records";

-- 创建新表
CREATE TABLE detection_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    source_type VARCHAR(20),
    source_name VARCHAR(255),
    original_path VARCHAR(255),
    result_path VARCHAR(255),
    duration FLOAT,
    total_objects INTEGER,
    detection_results TEXT,
    is_cleaned BOOLEAN DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 重命名原表为临时表
CREATE TABLE IF NOT EXISTS detection_records_temp (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    timestamp DATETIME,
    source_type VARCHAR(20),
    source_name VARCHAR(255),
    original_path VARCHAR(255),
    result_path VARCHAR(255),
    duration FLOAT,
    total_objects INTEGER,
    detection_results TEXT,
    is_cleaned BOOLEAN DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 复制数据
INSERT INTO detection_records_temp 
SELECT id, user_id, timestamp, source_type, source_name, original_path, result_path, duration, total_objects, detection_results, 0
FROM "detection records";

-- 删除原表
DROP TABLE "detection records";

-- 重命名临时表为正确的表名
ALTER TABLE detection_records_temp RENAME TO detection_records; 