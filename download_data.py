import kagglehub

# Download latest version
path = kagglehub.dataset_download("kartik2112/fraud-detection")
path = kagglehub.dataset_download("shriyashjagtap/fraudulent-e-commerce-transactions")
path = kagglehub.dataset_download("aimlveera/counterfeit-product-detection-dataset")
path = kagglehub.dataset_download("kevinvagan/fraud-detection-dataset")
path = kagglehub.dataset_download("sahilislam007/e-commerce-customer-analytics-loyalty-vs-fraud")

print("Path to dataset files:", path)