# %% [markdown]
# # ZeroWatt DAY2 — 피크/최대수요 · 이상탐지 · Fault Injection 검증
#
# 데이터가 균질(합성 추정)하다는 DAY1 결론 위에서, **살아있는 신호**만 골라 분석한다.
# - (a) 순간 최대수요(기본요금 기준) + 피크 이벤트 + 부하율
# - (b) 설비별 자기기준선 대비 이상탐지 (rolling z-score / IQR / IsolationForest)
# - (c) Fault Injection — 정상 데이터에 알려진 이상을 주입해 탐지기 성능(P/R/F1/지연) 정량화
#
# 실행: `CONFIG["use_mock"]=False` 로 두면 factory_db(MySQL)에서 읽는다.
# 오프라인 개발/검증용으로 `use_mock=True` 면 관측 통계에 맞춘 가짜 데이터로 파이프라인 전체를 돌린다.

# %%
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.ensemble import IsolationForest
warnings.filterwarnings("ignore")

# ---- 한글 폰트: 로컬(Mac)은 NanumGothic, 없으면 Noto CJK 자동 폴백 ----
def set_korean_font():
    for cand in ["Nanum Gothic", "AppleGothic", "Malgun Gothic"]:
        if any(cand.replace(" ", "").lower() in f.name.replace(" ", "").lower()
               for f in fm.fontManager.ttflist):
            mpl.rcParams["font.family"] = cand
            mpl.rcParams["axes.unicode_minus"] = False
            return cand
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            mpl.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            mpl.rcParams["axes.unicode_minus"] = False
            return mpl.rcParams["font.family"]
    return "default"
print("font:", set_korean_font())

# %%
# ============================= CONFIG =============================
CONFIG = {
    "use_mock": True,               # True: 가짜데이터로 검증 / False: MySQL factory_db
    "db": {                          # .env 사용 (DB_HOST/DB_USER/DB_PASSWORD/DB_NAME)
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "factory_db"),
    },
    "table": "raw_power_data",
    "power_unit_scale": 0.001,       # raw W -> kW
    "sample_sec": 5,                 # 5초 샘플링
    "demand_window_min": 15,         # 한전 최대수요: 15분 평균
    "zscore_window": 720,            # rolling window (720*5s = 1시간)
    "zscore_thresh": 3.0,
    "iqr_k": 1.5,
    "iforest_contamination": 0.003,  # 관측 꼬리 비율(~0.3%)에 맞춤
    "modules": None,                 # None이면 전체, 리스트면 해당 설비만
    "outdir": "day2_out",
}
os.makedirs(CONFIG["outdir"], exist_ok=True)
FEATURES = ["activePower_kw", "pf_avg", "volt_unbalance", "curr_unbalance", "reactive"]

# %%
# ============================= DATA LOADERS =============================
def _derive(df):
    """원시 컬럼 -> 분석 피처. MySQL/mock 공통."""
    df = df.copy()
    df["activePower_kw"] = df["activePower"] * CONFIG["power_unit_scale"]
    v = df[["voltageR", "voltageS", "voltageT"]]
    i = df[["currentR", "currentS", "currentT"]]
    df["pf_avg"] = df[["powerFactorR", "powerFactorS", "powerFactorT"]].mean(axis=1)
    df["volt_unbalance"] = (v.max(axis=1) - v.min(axis=1)) / v.mean(axis=1) * 100  # %
    df["curr_unbalance"] = (i.max(axis=1) - i.min(axis=1)) / i.mean(axis=1) * 100  # %
    df["reactive"] = df["reactivePowerLagging"]
    # localtime(BIGINT, YYYYMMDDHHMMSS) -> datetime (KST)
    df["dt"] = pd.to_datetime(df["localtime"].astype("int64").astype(str),
                              format="%Y%m%d%H%M%S", errors="coerce")
    return df

def load_from_mysql(module):
    """factory_db에서 한 설비 원시 시계열 로드. localtime 은 예약어라 백틱 필수."""
    import pymysql
    conn = pymysql.connect(**CONFIG["db"])
    q = f"""
        SELECT module, ts, `localtime`, activePower,
               voltageR, voltageS, voltageT,
               currentR, currentS, currentT,
               powerFactorR, powerFactorS, powerFactorT,
               reactivePowerLagging, accumActiveEnergy
        FROM {CONFIG['table']}
        WHERE module = %s
        ORDER BY ts
    """
    df = pd.read_sql(q, conn, params=[module])
    conn.close()
    return _derive(df)

def list_modules_mysql():
    import pymysql
    conn = pymysql.connect(**CONFIG["db"])
    df = pd.read_sql(f"SELECT DISTINCT module FROM {CONFIG['table']}", conn)
    conn.close()
    return df["module"].tolist()

# ---- MOCK: 관측 통계(mean 3010W, std 717W, V215, I17.5, PF92.5)에 맞춤 ----
def make_mock(module, days=5, seed=0):
    rng = np.random.RandomState(seed)
    n = int(days * 24 * 3600 / CONFIG["sample_sec"])
    ts0 = 1733011200000  # 2024-12-01 00:00:00 KST epoch ms
    ts = ts0 + np.arange(n) * CONFIG["sample_sec"] * 1000
    dt = pd.to_datetime(ts, unit="ms")
    lt = dt.strftime("%Y%m%d%H%M%S").astype("int64")
    power = np.clip(rng.normal(3010, 717, n), 400, 5400)         # W
    df = pd.DataFrame({
        "module": module, "ts": ts, "localtime": lt, "activePower": power,
        "voltageR": rng.normal(215, 0.5, n), "voltageS": rng.normal(215, 0.5, n),
        "voltageT": rng.normal(215, 0.5, n),
        "currentR": rng.normal(17.5, 0.4, n), "currentS": rng.normal(17.5, 0.4, n),
        "currentT": rng.normal(17.5, 0.4, n),
        "powerFactorR": rng.normal(92.5, 0.3, n), "powerFactorS": rng.normal(92.5, 0.3, n),
        "powerFactorT": rng.normal(92.5, 0.3, n),
        "reactivePowerLagging": rng.normal(0.602, 0.02, n),
        "accumActiveEnergy": np.cumsum(power * CONFIG["sample_sec"] / 3600),
    })
    return _derive(df)

def load_module(module, **kw):
    return make_mock(module, **kw) if CONFIG["use_mock"] else load_from_mysql(module)

def get_module_list():
    if CONFIG["use_mock"]:
        return ["12(4호기)", "16(호이스트)", "2(L-1전등)"]
    return CONFIG["modules"] or list_modules_mysql()

# %% [markdown]
# ## (a) 최대수요전력 · 피크 이벤트 · 부하율
#
# - **최대수요**: 15분 평균전력의 최댓값 → 한전 기본요금 산정 기준.
# - **부하율(Load Factor)** = 평균수요 / 최대수요. 낮을수록 짧은 피크 때문에 계약전력을 크게 잡는 비효율.
# - **피크 이벤트**: 15분 수요가 rolling μ+3σ 를 넘는 시점.

# %%
def analyze_peak(df, module):
    w = int(CONFIG["demand_window_min"] * 60 / CONFIG["sample_sec"])  # 15분 = 180 샘플
    s = df.set_index("dt")["activePower_kw"]
    demand = s.rolling(w, min_periods=w).mean()          # 15분 이동평균 = 순간수요
    demand15 = demand.resample("15min").max().dropna()   # 15분 구간 대표 수요
    peak = demand15.max()
    avg = demand15.mean()
    lf = avg / peak if peak else np.nan
    mu, sd = demand15.mean(), demand15.std()
    events = demand15[demand15 > mu + 3 * sd]
    return {"module": module, "peak_demand_kw": peak, "avg_demand_kw": avg,
            "load_factor": lf, "n_peak_events": len(events),
            "peak_time": demand15.idxmax(), "demand15": demand15, "events": events}

def plot_peak(res):
    d = res["demand15"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    # 수요 지속곡선 (Load Duration Curve)
    ax[0].plot(np.sort(d.values)[::-1], color="#4C78A8")
    ax[0].axhline(res["peak_demand_kw"], color="red", ls="--", lw=1,
                  label=f"최대수요 {res['peak_demand_kw']:.2f} kW")
    ax[0].axhline(res["avg_demand_kw"], color="green", ls=":", lw=1,
                  label=f"평균수요 {res['avg_demand_kw']:.2f} kW")
    ax[0].set_title(f"{res['module']} 수요 지속곡선 · 부하율 {res['load_factor']*100:.1f}%",
                    fontweight="bold")
    ax[0].set_xlabel("15분 구간 순위"); ax[0].set_ylabel("수요전력 (kW)"); ax[0].legend()
    # 피크 이벤트 타임라인
    ax[1].plot(d.index, d.values, color="#999", lw=0.6)
    if len(res["events"]):
        ax[1].scatter(res["events"].index, res["events"].values, color="red", s=18,
                      zorder=3, label=f"피크 이벤트 {res['n_peak_events']}건")
    ax[1].set_title(f"{res['module']} 15분 수요 · 피크 이벤트(μ+3σ 초과)", fontweight="bold")
    ax[1].set_xlabel("시간"); ax[1].legend()
    for a in ax:
        [a.spines[s].set_visible(False) for s in ["top", "right"]]
    plt.tight_layout()
    plt.savefig(f"{CONFIG['outdir']}/a_peak_{res['module']}.png", dpi=150, bbox_inches="tight")
    plt.close()

# %% [markdown]
# ## (b) 설비별 이상탐지 — 자기기준선 대비
# rolling z-score(단변량) + IQR + IsolationForest(다변량). 라벨 불필요, 설명 가능.

# %%
def detect_zscore(x, w, k):
    mu = x.rolling(w, min_periods=w//2).mean()
    sd = x.rolling(w, min_periods=w//2).std()
    z = (x - mu) / sd
    return (z.abs() > k).fillna(False), z

def detect_iqr(x, k):
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return (x < q1 - k*iqr) | (x > q3 + k*iqr)

def detect_iforest(df, features, contam):
    X = df[features].fillna(df[features].median())
    m = IsolationForest(contamination=contam, random_state=42, n_estimators=200)
    pred = m.fit_predict(X)          # -1 = 이상
    score = -m.score_samples(X)      # 클수록 이상
    return pd.Series(pred == -1, index=df.index), pd.Series(score, index=df.index)

def analyze_anomaly(df, module):
    p = df["activePower_kw"]
    flag_z, z = detect_zscore(p, CONFIG["zscore_window"], CONFIG["zscore_thresh"])
    flag_iqr = detect_iqr(p, CONFIG["iqr_k"])
    flag_if, score_if = detect_iforest(df, FEATURES, CONFIG["iforest_contamination"])
    out = pd.DataFrame({"dt": df["dt"], "power_kw": p,
                        "flag_z": flag_z.values, "flag_iqr": flag_iqr.values,
                        "flag_if": flag_if.values})
    summ = {"module": module, "n": len(df),
            "z_flags": int(flag_z.sum()), "iqr_flags": int(flag_iqr.sum()),
            "if_flags": int(flag_if.sum()),
            "any_flags": int((flag_z.values | flag_iqr.values | flag_if.values).sum())}
    return out, summ

# %% [markdown]
# ## (c) Fault Injection 검증 (★ 핵심 차별화)
# 정상 데이터에 4가지 알려진 이상을 심고, 탐지기가 몇 %를 잡는지 정량화한다.

# %%
FAULTS = ["idle", "low_pf", "phase_imbalance", "spike"]

def inject_faults(df, seed=1):
    """clean df에 이상 구간 주입. ground-truth 라벨과 구간정보 반환."""
    rng = np.random.RandomState(seed)
    d = df.copy().reset_index(drop=True)
    n = len(d)
    label = pd.Series(False, index=d.index)
    ftype = pd.Series("", index=d.index)
    segs = []
    # 각 이상 유형별로 몇 개 구간 삽입
    plan = [("idle", 6, (120, 360)),            # 10~30분 무부하
            ("low_pf", 6, (120, 360)),
            ("phase_imbalance", 6, (120, 360)),
            ("spike", 8, (3, 12))]              # 15~60초 짧은 스파이크
    for name, cnt, (lo, hi) in plan:
        for _ in range(cnt):
            L = rng.randint(lo, hi)
            s = rng.randint(0, n - L)
            idx = np.arange(s, s + L)
            if name == "idle":
                d.loc[idx, "activePower_kw"] *= 0.05        # 5% 로 급락
            elif name == "low_pf":
                d.loc[idx, "pf_avg"] = rng.normal(80, 1, L) # 역률 80%
            elif name == "phase_imbalance":
                d.loc[idx, "curr_unbalance"] += rng.normal(15, 2, L)  # +15%p 불평형
            elif name == "spike":
                d.loc[idx, "activePower_kw"] *= rng.uniform(1.7, 2.2)  # 급등
            label.loc[idx] = True
            ftype.loc[idx] = name
            segs.append((name, idx[0], idx[-1]))
    d["_label"] = label.values
    d["_ftype"] = ftype.values
    return d, segs

def run_detectors_for_eval(d):
    """평가용: 주입된 이상 유형을 커버하도록 다중 탐지기 결합."""
    p = d["activePower_kw"]
    flag_z, _ = detect_zscore(p, CONFIG["zscore_window"], CONFIG["zscore_thresh"])
    flag_pf = d["pf_avg"] < 90.0                                   # 저역률 규칙
    flag_ub = d["curr_unbalance"] > 10.0                          # 전류불평형 규칙(NEMA)
    flag_if, _ = detect_iforest(d, FEATURES, CONFIG["iforest_contamination"]*4)
    pred = (flag_z.values | flag_pf.values | flag_ub.values | flag_if.values)
    return pd.Series(pred, index=d.index)

def prf(label, pred):
    tp = int((label & pred).sum()); fp = int((~label & pred).sum())
    fn = int((label & ~pred).sum())
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return prec, rec, f1

def eval_by_fault(d, pred, segs):
    """유형별 재현율 + 탐지지연(초)."""
    rows = []
    for name in FAULTS:
        seg_n = [s for s in segs if s[0] == name]
        detected, latencies = 0, []
        for _, a, b in seg_n:
            hit = np.where(pred.values[a:b+1])[0]
            if len(hit):
                detected += 1
                latencies.append(hit[0] * CONFIG["sample_sec"])
        rows.append({"fault": name, "n_seg": len(seg_n), "detected": detected,
                     "recall_seg": detected/len(seg_n) if seg_n else 0,
                     "median_latency_s": np.median(latencies) if latencies else np.nan})
    return pd.DataFrame(rows)

# %%
# ============================= RUN ALL =============================
modules = get_module_list()
peak_rows, anom_rows = [], []

for mi, mod in enumerate(modules):
    df = load_module(mod, days=5, seed=mi) if CONFIG["use_mock"] else load_module(mod)
    # (a)
    r = analyze_peak(df, mod); plot_peak(r)
    peak_rows.append({k: r[k] for k in
                      ["module","peak_demand_kw","avg_demand_kw","load_factor",
                       "n_peak_events","peak_time"]})
    # (b)
    out, summ = analyze_anomaly(df, mod); anom_rows.append(summ)

peak_df = pd.DataFrame(peak_rows)
anom_df = pd.DataFrame(anom_rows)
print("\n===== (a) 최대수요 · 부하율 =====")
print(peak_df.to_string(index=False))
print("\n===== (b) 이상탐지 플래그 수 =====")
print(anom_df.to_string(index=False))

# (c) — 대표 설비 1개로 fault injection 검증
base = load_module(modules[0], days=5) if CONFIG["use_mock"] else load_module(modules[0])
inj, segs = inject_faults(base)
pred = run_detectors_for_eval(inj)
P, R, F1 = prf(inj["_label"], pred)
by_fault = eval_by_fault(inj, pred, segs)
print("\n===== (c) Fault Injection 검증 =====")
print(f"전체 포인트 기준  Precision={P:.3f}  Recall={R:.3f}  F1={F1:.3f}")
print(by_fault.to_string(index=False))

# (c) 시각화
fig, ax = plt.subplots(figsize=(13, 4))
seg_view = inj.iloc[:8000]
ax.plot(seg_view.index, seg_view["activePower_kw"], color="#888", lw=0.6, label="전력")
lab = seg_view[seg_view["_label"]]
det = seg_view[pred.iloc[:8000].values]
ax.scatter(lab.index, lab["activePower_kw"], color="orange", s=10, label="주입 이상(정답)", zorder=2)
ax.scatter(det.index, det["activePower_kw"], facecolors="none", edgecolors="red", s=30,
           label="탐지됨", zorder=3)
ax.set_title(f"Fault Injection 검증 (앞 8000 포인트) · Recall={R:.2f} F1={F1:.2f}", fontweight="bold")
ax.set_xlabel("샘플 index"); ax.legend(ncol=3)
[ax.spines[s].set_visible(False) for s in ["top","right"]]
plt.tight_layout()
plt.savefig(f"{CONFIG['outdir']}/c_fault_injection.png", dpi=150, bbox_inches="tight")
plt.close()

# 결과 저장
peak_df.to_csv(f"{CONFIG['outdir']}/a_peak_summary.csv", index=False)
anom_df.to_csv(f"{CONFIG['outdir']}/b_anomaly_summary.csv", index=False)
by_fault.to_csv(f"{CONFIG['outdir']}/c_faultinjection_byfault.csv", index=False)
print(f"\n결과 저장 완료 → {CONFIG['outdir']}/")
