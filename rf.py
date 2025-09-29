import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/los/LengthOfStay.csv', sep=',', index_col='eid', encoding='utf-8')

# Convert rcount, gender, and facid to numeric
conversions = {
    'rcount': {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5+': 5,
    },
    'gender': {
        'M': 0,
        'F': 1,
    },
    'facid': {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
    }
}

for col, mapping in conversions.items():
    df[col] = df[col].map(mapping)
df = df.dropna()

# Drop date columns
df = df.drop(columns=['vdate', 'discharged'])

# Build the training and testing sets
X = df.drop(columns=['lengthofstay'])
y = df['lengthofstay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on train and test set
y_hats_train = model.predict(X_train)
y_hats_test = model.predict(X_test)

# Add predictions to the original dataframe in a new column
df.loc[y_train.index, 'predicted_lengthofstay'] = y_hats_train
df.loc[y_test.index, 'predicted_lengthofstay'] = y_hats_test

# Save the dataframe with predictions to a new CSV file
df.to_csv('data/los/LengthOfStay_predictions.csv', sep=',', encoding='utf-8')

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_train, y_hats_train, label='Train', alpha=0.7)
plt.scatter(y_test, y_hats_test, label='Test', alpha=0.7)
plt.plot([0, 20], [0, 20], 'k--', lw=2)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('Actual Length of Stay')
plt.ylabel('Predicted Length of Stay')
plt.title('Actual vs Predicted Length of Stay')
plt.legend()
plt.show()

# Evaluate the model on train and test set
mae_train = mean_absolute_error(y_train, y_hats_train)
mae_test = mean_absolute_error(y_test, y_hats_test)
mse_train = mean_squared_error(y_train, y_hats_train)
mse_test = mean_squared_error(y_test, y_hats_test)
rmse_train = mse_train ** 0.5
rmse_test = mse_test ** 0.5

print(f'MAE: {mae_train}, MSE: {mse_train}, RMSE: {rmse_train} (train)')
print(f'MAE: {mae_test}, MSE: {mse_test}, RMSE: {rmse_test} (test)')