import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('NVDA_cleaned_data.csv')

# Feature Engineering
df['Daily Range'] = df['High'] - df['Low']
df['Average Price'] = (df['Open'] + df['High'] + df['Low']) / 3
df['5-Day Rolling Avg'] = df['Close'].rolling(window=5).mean()

# Drop rows with NaN values resulting from rolling average
df.dropna(inplace=True)

# Select features and target
X = df[['Open', 'High', 'Low', 'Volume', 'Daily Range', 'Average Price', '5-Day Rolling Avg']]
y = df['Close']

# Feature Selection based on correlation
correlation = X.corrwith(y).abs()
print("Feature Correlations with Close Price:")
print(correlation)

# Drop features with low correlation (e.g., less than 0.1)
low_correlation_features = correlation[correlation < 0.5].index
X = X.drop(columns=low_correlation_features)

print(f"Selected Features: {X.columns}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression Model ---
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Evaluate Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - Mean Squared Error: {mse_lr}")
print(f"Linear Regression - R² Score: {r2_lr}")

# --- Random Forest Model ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R² Score: {r2_rf}")

# Plot Actual vs Predicted Prices for Linear Regression
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, color="blue", alpha=0.7, label="Predicted Prices (LR)")
min_val = min(min(y_test), min(y_pred_lr))
max_val = max(max(y_test), max(y_pred_lr))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label="Ideal Fit")
plt.xlabel("Actual Prices")
plt.ylabel("Predicteqd Prices")
plt.title("Linear Regression: Actual vs Predicted Prices")
plt.legend()
plt.grid(True)

# Plot Actual vs Predicted Prices for Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, color="green", alpha=0.7, label="Predicted Prices (RF)")
min_val = min(min(y_test), min(y_pred_rf))
max_val = max(max(y_test), max(y_pred_rf))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label="Ideal Fit")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest: Actual vs Predicted Prices")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
