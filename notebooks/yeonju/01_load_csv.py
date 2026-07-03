"""
CSV -> MySQL 적재 스크립트 (실제 스키마 확정 버전)
================================================
컬럼: module, timestamp, localtime, operation,
      voltageR/S/T, voltageRS/ST/TR, currentR/S/T,
      activePower, powerFactorR/S/T, reactivePowerLagging,
      accumActiveEnergy
"""

import pymysql  # 없으면: pip install pymysql --break-system-packages
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 읽어서 환경변수로 등록


# =========================================================
# 설정
# =========================================================
CSV_PATH = "/Users/yeonjulee/Desktop/ZEROWATT/ZeroWatt/rtu_data_full.csv"   # 실제 파일 경로 (절대경로 필수)
DB_CONFIG = dict(
    host=os.environ["DB_HOST"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    database=os.environ["DB_NAME"],
)
TABLE_NAME = "raw_power_data"

conn = pymysql.connect(**DB_CONFIG, local_infile=True)
cur = conn.cursor()


# =========================================================
# STEP 1. local_infile 옵션 확인 (꺼져있으면 LOAD 실패함)
# =========================================================
cur.execute("SHOW VARIABLES LIKE 'local_infile';")
print(cur.fetchone())
# 만약 OFF라면 아래 주석 풀고 실행 (관리자 권한 필요할 수 있음)
# cur.execute("SET GLOBAL local_infile = 1;")


# =========================================================
# STEP 2. 테이블 생성 (실제 컬럼 스키마 반영)
#   - module: 설비명 (VARCHAR)
#   - timestamp: Unix epoch ms (BIGINT)
#   - localtime: YYYYMMDDHHMMSS 형태 (BIGINT)
#   - operation: 가동상태 플래그로 추정 (BIGINT)
#   - voltage/current/power 계열: DOUBLE
#   - accumActiveEnergy: 누적 전력량, 정수 절삭값 (BIGINT)
# =========================================================
create_sql = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    module                VARCHAR(50),
    ts                    BIGINT,
    `localtime`             BIGINT,
    operation             BIGINT,
    voltageR              DOUBLE,
    voltageS              DOUBLE,
    voltageT              DOUBLE,
    voltageRS             DOUBLE,
    voltageST             DOUBLE,
    voltageTR             DOUBLE,
    currentR              DOUBLE,
    currentS              DOUBLE,
    currentT              DOUBLE,
    activePower           DOUBLE,
    powerFactorR          DOUBLE,
    powerFactorS          DOUBLE,
    powerFactorT          DOUBLE,
    reactivePowerLagging  DOUBLE,
    accumActiveEnergy     BIGINT,
    INDEX idx_module (module),
    INDEX idx_ts (ts)
) ENGINE=InnoDB;
"""
# 주의: CSV의 "timestamp" 컬럼명이 MySQL 예약어와 겹칠 수 있어 컬럼명을 ts로 매핑함
#       (아래 LOAD 구문에서 순서로 매핑되므로 컬럼명이 달라도 문제 없음)
cur.execute(create_sql)
conn.commit()
print("테이블 생성 완료")


# =========================================================
# STEP 3. LOAD DATA INFILE로 고속 적재
#   - 5GB급이라 pandas to_sql보다 수십~수백 배 빠름
#   - CSV 컬럼 순서가 CREATE TABLE 컬럼 순서와 동일해야 함
#     (DESCRIBE 결과 순서 그대로: module, timestamp, localtime, operation,
#      voltageR, voltageS, voltageT, voltageRS, voltageST, voltageTR,
#      currentR, currentS, currentT, activePower,
#      powerFactorR, powerFactorS, powerFactorT,
#      reactivePowerLagging, accumActiveEnergy)
# =========================================================
load_sql = f"""
LOAD DATA LOCAL INFILE '{CSV_PATH}'
INTO TABLE {TABLE_NAME}
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 ROWS
(module, ts, `localtime`, operation,
 voltageR, voltageS, voltageT, voltageRS, voltageST, voltageTR,
 currentR, currentS, currentT, activePower,
 powerFactorR, powerFactorS, powerFactorT,
 reactivePowerLagging, accumActiveEnergy);
"""
cur.execute(load_sql)
conn.commit()
print("적재 완료")


# =========================================================
# STEP 4. 적재 검증
# =========================================================
cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
print("전체 row 수:", cur.fetchone()[0])

cur.execute(f"""
    SELECT module, COUNT(*) n, AVG(activePower) avg_power,
           MIN(activePower) min_power, MAX(activePower) max_power,
           STDDEV(activePower) std_power
    FROM {TABLE_NAME}
    GROUP BY module
    ORDER BY module;
""")
for row in cur.fetchall():
    print(row)

conn.close()