# Fraud Detection in E-commerce using ML  
## 数据清理与统一化总结报告（基于 dataset_clean_summary.json）

---

## 🧩 Part 1. 全局概览

| 数据集名称 | 原始特征数（约） | 清洗后特征数 | 新增特征 | 样本规模 (train/test) | 标签正例比例 (train/test) | 缺失率情况 |
|-------------|------------------|---------------|-----------|------------------------|-----------------------------|-------------|
| **IEEE** | 394+ | 81 | `log_amount`, `hour_of_day`, `day_of_week`, `sin_hour`, `cos_hour`, `user_txn_*`, `freq_*`, `V_pca_*` | 100k / 100k | ≈2.7%（原）→ **2–3%（稳定）** | 0% |
| **creditCardPCA** | 31 | 34 | `timestamp`, `hour_of_day`, `day_of_week`, `log_amount` | 100k / 56,962 | 0.16% → **0.17%** | 0% |
| **creditCardTransaction** | 13 | 13 | 无（本身已结构化） | 100k / 100k | 0.99% → **0.4–1.0%** | 0% |
| **col14_behave** | 14 | 15 | `log_amount`, `hour_of_day`, `day_of_week` | 100k / 59,400 | 6.8% → **6.8%** | 0% |
| **col16_raw** | 16 | 14 | `log_amount`, `hour_of_day`, `day_of_week` | 100k / 23,633 | 4.9–5.1% | 0% |
| **counterfeit_products** | 27 | 16 | `timestamp`, `hour_of_day`, `day_of_week`, `log_amount` | 4,000 / 1,000 | 29.4% | 0% |
| **counterfeit_transactions** | 20 | 19 | `timestamp`, `hour_of_day`, `day_of_week`, `log_amount` | 2,400 / 600 | 24.4–24.5% | 0% |

---

## 📊 Part 2. 数据集详细对比

### 1️⃣ IEEE-CIS Fraud Detection
...（此处省略：详细部分同上）...
