# Fraud Detection Data Cleaning Log

This document records the preprocessing and feature standardization decisions for all datasets used in the project **“Fraud Detection in E-commerce using Machine Learning.”**

Each dataset entry includes:
- Original columns (from raw data)
- Unified schema mapping
- Transformation operations
- Rationale behind each step

---

## 🧩 Dataset: col14

**Source:** Simulated e-commerce transaction dataset with 14 columns.  
**Goal:** Standardize feature names and types to match the unified schema for fraud detection.

### 1️⃣ Original Columns

| Original Name | Description |
|----------------|-------------|
| Transaction.Date | Timestamp of the transaction |
| Transaction.Amount | Amount of the transaction |
| Customer.Age | Age of the customer |
| Is.Fraudulent | Fraud label (0 = legitimate, 1 = fraud) |
| Account.Age.Days | Age of the customer account (days) |
| Transaction.Hour | Hour of the transaction (0–23) |
| source | Traffic source (Ads, Direct, Referral...) |
| browser | Browser used during transaction |
| sex | Customer gender |
| Payment.Method | Payment channel (Credit Card, PayPal, etc.) |
| Product.Category | Purchased product type |
| Quantity | Number of items purchased |
| Device.Used | Device used for the transaction |
| Address.Match | Whether billing and shipping addresses match |

---

### 2️⃣ Standardized Schema Mapping

| Original Column | Unified Name | Type | Transformation | Rationale |
|------------------|---------------|------|----------------|------------|
| Transaction.Date | `timestamp` | datetime64 | Parsed with `pd.to_datetime()` | Keeps chronological order |
| Transaction.Amount | `amount` | float64 | Converted to float; additional feature `log_amount = log1p(amount)` | Reduces skewness, improves model stability |
| Customer.Age | `user_age` | int64 | Filled missing with median | Demographic feature |
| Is.Fraudulent | `is_fraud` | int64 | 0/1 label encoding | Binary classification target |
| Account.Age.Days | `account_age_days` | int64 | None | Proxy for account maturity |
| Transaction.Hour | `hour_of_day` | int64 | Derived if missing from timestamp | Captures temporal patterns |
| source | `channel` | category | LabelEncoder per dataset | Reflects marketing/traffic origin |
| browser | `browser` | category | LabelEncoder per dataset | Part of user behavior fingerprint |
| sex | `user_gender` | category | LabelEncoder per dataset | Demographic category |
| Payment.Method | `payment_method` | category | LabelEncoder per dataset | Payment preference |
| Product.Category | `product_category` | category | LabelEncoder per dataset | Product-based risk differentiation |
| Quantity | `quantity` | int64 | None | Order intensity measure |
| Device.Used | `device_type` | category | LabelEncoder per dataset | Indicates platform used |
| Address.Match | `address_match` | bool/int64 | Converted Yes→1, No→0 | Address mismatch is common fraud signal |
| — | `day_of_week` | float64 | Extracted from timestamp | Weekly seasonality feature |
| — | `log_amount` | float64 | Added: `np.log1p(amount)` | Handle long-tailed distribution |

---

### 3️⃣ Processing Summary

```python
# 1. Standardize column names
df.rename(columns={...}, inplace=True)

# 2. Convert datatypes
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['amount'] = df['amount'].astype(float)
df['is_fraud'] = df['is_fraud'].astype(int)
df['address_match'] = df['address_match'].map({'Yes': 1, 'No': 0}).fillna(1)

# 3. Derived features
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour_of_day'] = df['timestamp'].dt.hour
df['log_amount'] = np.log1p(df['amount'])

# 4. Encode categorical features (within dataset)
from sklearn.preprocessing import LabelEncoder
for col in ['channel', 'browser', 'user_gender', 'payment_method', 'product_category', 'device_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
