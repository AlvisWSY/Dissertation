# paysim_to_ecom_aug.py
import pandas as pd
import numpy as np
import hashlib, json
from scipy import stats
import uuid

# ---------- 配置区（可调） ----------
INPUT_CSV = "data/PS_20174392719_1491204439457_log.csv"
OUTPUT_CSV = "paysim_augmented.csv"

# mapping概率与阈值（可按实验调整）
REFUND_PROB_IF_SMALL_AMOUNT = 0.05
TRANSFER_TO_PURCHASE_PROB = 0.25
CASHOUT_AS_REFUND_PROB = 0.25
NEW_ADDRESS_BASE_PROB = 0.05
MOBILE_BASE_PROB = 0.8

# item category amount bins
CATEGORY_BINS = [
    (0, 10, "grocery"),
    (10, 50, "beauty"),
    (50, 200, "fashion"),
    (200, 1000, "electronics"),
    (1000, 1e9, "luxury"),
]

# session gap (PaySim step is coarse; treat gap > threshold as new session)
SESSION_GAP_STEP = 24  # if more than 24 steps -> new session

# ---------- 小工具 ----------
def deterministic_hash_to_int(s, mod=10**6):
    h = hashlib.md5(str(s).encode('utf8')).hexdigest()
    return int(h[:8], 16) % mod

def pick_item_category(amount):
    for lo, hi, cat in CATEGORY_BINS:
        if lo <= amount < hi:
            return cat
    return "other"

def random_choice_by_prob(p):
    return np.random.rand() < p

# ---------- 读取 ----------
df = pd.read_csv(INPUT_CSV)

# 确保部分必要列存在
required_cols = ['step','type','amount','nameOrig','nameDest']
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# 保留原始数据副本列
df['orig_type'] = df['type']
df['orig_amount'] = df['amount']

# ---------- 映射 ecom_event ----------
def map_row_to_ecom_event(row):
    t = row['type']
    amt = float(row['amount'])
    # default
    event = None
    meta = {}
    if t in ('PAYMENT','DEBIT'):
        # 大概率为 purchase；极小概率为 refund（模拟友好欺诈情形）
        if amt < 5 and random_choice_by_prob(REFUND_PROB_IF_SMALL_AMOUNT):
            event = 'refund'
            meta['rule'] = 'small_amount_likely_refund'
        else:
            event = 'purchase'
            meta['rule'] = 'payment_as_purchase'
    elif t == 'CASH-IN':
        event = 'topup'
        meta['rule'] = 'cashin_as_topup'
    elif t == 'CASH-OUT':
        # 有部分可能是 refund; 其余视为 withdrawal
        if random_choice_by_prob(CASHOUT_AS_REFUND_PROB):
            event = 'refund'
            meta['rule'] = 'cashout_as_refund_probabilistic'
        else:
            event = 'withdrawal'
            meta['rule'] = 'cashout_as_withdrawal'
    elif t == 'TRANSFER':
        # 条件化：部分作为 purchase（代付），其他作为 peer transfer or payout
        r = np.random.rand()
        if r < TRANSFER_TO_PURCHASE_PROB:
            event = 'purchase'
            meta['rule'] = 'transfer_to_purchase_prob'
        else:
            # 判断是否商户收款：简单 heuristic: nameDest startswith 'M' or 'merchant'
            if str(row['nameDest']).upper().startswith('M') or 'MERCH' in str(row['nameDest']).upper():
                event = 'payout'
                meta['rule'] = 'transfer_to_payout_merchant'
            else:
                event = 'peer_transfer'
                meta['rule'] = 'transfer_peer'
    else:
        event = 'other'
        meta['rule'] = 'unknown_type_passthrough'
    return event, meta

mapped = df.apply(lambda r: map_row_to_ecom_event(r), axis=1)
df['ecom_event'] = mapped.map(lambda x: x[0])
df['meta_reason'] = mapped.map(lambda x: json.dumps(x[1]))

# ---------- 生成 order_id 与 item_category 等（对 purchase） ----------
df['order_id'] = None
df['item_category'] = None
df['device_type'] = None
df['ip_region'] = None
df['account_age_days'] = None
df['is_new_shipping_address'] = False
df['session_id'] = None
df['session_seq'] = None

# 按 user (nameOrig) 排序生成 session_id
df = df.sort_values(['nameOrig','step']).reset_index(drop=True)

# iterate per user to assign session_id and account_age
curr_session_uuid = None
for user, group in df.groupby('nameOrig', sort=False):
    # deterministic account_age: smaller hash -> older/younger distribution
    base = deterministic_hash_to_int(user, mod=3650)  # 0..3649 days
    # map to 1..3650
    account_age = int((base % 3000) + 30)  # safe floor of 30 days
    idxs = group.index.tolist()
    last_step = None
    session_counter = 0
    seq = 0
    for i in idxs:
        step = int(df.at[i,'step'])
        if last_step is None or (step - last_step) > SESSION_GAP_STEP:
            session_counter += 1
            curr_session_uuid = f"{user}_sess_{session_counter}"
            seq = 1
        else:
            seq += 1
        df.at[i, 'session_id'] = curr_session_uuid
        df.at[i, 'session_seq'] = seq
        df.at[i, 'account_age_days'] = account_age
        last_step = step

# device_type & ip_region correlated with deterministic hash + randomness
def pick_device(user, event):
    # mobile common for purchases, less for withdrawals
    base = deterministic_hash_to_int(user, mod=100)
    p_mobile = MOBILE_BASE_PROB if event=='purchase' else 0.6
    # bias by base
    if base % 100 < p_mobile*100:
        return 'mobile'
    else:
        return 'desktop'

def pick_ip_region(user):
    idx = deterministic_hash_to_int(user, mod=100)
    # simple distribution across 5 regions
    regions = ['SG','MY','ID','VN','PH']
    return regions[idx % len(regions)]

for i, row in df.iterrows():
    df.at[i,'device_type'] = pick_device(row['nameOrig'], row['ecom_event'])
    df.at[i,'ip_region'] = pick_ip_region(row['nameOrig'])
    # new shipping address: higher prob if account young
    acct = int(row['account_age_days']) if pd.notnull(row['account_age_days']) else 365
    prob_new_addr = NEW_ADDRESS_BASE_PROB if acct > 365 else min(0.5, NEW_ADDRESS_BASE_PROB + (365-acct)/365*0.5)
    df.at[i,'is_new_shipping_address'] = random_choice_by_prob(prob_new_addr)
    # item category only for purchase
    if row['ecom_event'] == 'purchase':
        df.at[i,'item_category'] = pick_item_category(float(row['amount']))
        df.at[i,'order_id'] = str(uuid.uuid4())
    else:
        df.at[i,'item_category'] = None

# ---------- meta: generator column ----------
df['aug_generator'] = 'rule_based_v1'

# ---------- 简单校验：KS test between original PAYMENT amount and augmented purchases ----------
orig_payments = df[df['orig_type'].isin(['PAYMENT','DEBIT'])]['orig_amount'].dropna().astype(float)
aug_purchases = df[df['ecom_event']=='purchase']['amount'].dropna().astype(float)

# If distributions exist, run ks
ks_stat, ks_p = (None, None)
if len(orig_payments)>0 and len(aug_purchases)>0:
    try:
        ks_stat, ks_p = stats.ks_2samp(orig_payments, aug_purchases)
    except Exception as e:
        print("KS test failed:", e)

print("KS two-sample test between orig PAYMENT amounts and augmented purchases: stat=%.4f p=%.4f" % (ks_stat, ks_p) if ks_stat is not None else "KS not computed")

# ---------- 保存 ----------
df.to_csv(OUTPUT_CSV, index=False)
print("Saved augmented csv to", OUTPUT_CSV)
