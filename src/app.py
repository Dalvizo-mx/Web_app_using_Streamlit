import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import base64
import math
import io  


def load_data():
    csv_file_path = '/workspaces/Web_app_using_Streamlit/Aquifer_Petrignano.csv'
    df = pd.read_csv(csv_file_path)
    return df


df = load_data()


df = df[df.Rainfall_Bastia_Umbra.notna()].reset_index(drop=True)
df = df.drop(['Depth_to_Groundwater_P24', 'Temperature_Petrignano'], axis=1)
df.columns = ['date', 'rainfall', 'depth_to_groundwater', 'temperature', 'drainage_volume', 'river_hydrometry']
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df = df.sort_values(by='date')


df['depth_to_groundwater'] = df['depth_to_groundwater'].fillna(method='ffill')
df['depth_to_groundwater'] = df['depth_to_groundwater'].fillna(method='bfill')


df = df.dropna()


st.sidebar.title("Settings")
test_size_percentage = st.sidebar.slider("Test Size Percentage", min_value=10, max_value=50, value=15, step=5)


univariate_df = df[['date', 'depth_to_groundwater']].copy()
univariate_df.columns = ['ds', 'y']
train_size = int((100 - test_size_percentage) / 100 * len(df))
train = univariate_df.iloc[:train_size, :]
x_valid, y_valid = univariate_df.iloc[train_size:, 0], univariate_df.iloc[train_size:, 1]


model = Prophet()
model.fit(train)
y_pred = model.predict(pd.DataFrame(x_valid))


score_mae = mean_absolute_error(y_valid, y_pred.tail(len(y_valid))['yhat'])
score_rmse = math.sqrt(mean_squared_error(y_valid, y_pred.tail(len(y_valid))['yhat']))


st.write(f"MAE: {score_mae:.2f}")
st.write(f"RMSE: {score_rmse:.2f}")


f, ax = plt.subplots(1, figsize=(15, 6))
model.plot(y_pred, ax=ax)
sns.lineplot(x=x_valid, y=y_valid, ax=ax, color='orange', label='Ground truth')
ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Depth to Groundwater', fontsize=14)


buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
plt.close()

st.write(df)
st.image(f"data:image/png;base64,{plot_data}")
