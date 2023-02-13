# Machine learning model that predicts future annual sales

import pandas as pd
import numpy as np
import datetime

import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import app as app

ventas= pd.read_csv(r'C:\Users\Lina_\OneDrive\Escritorio\PrediccionVentas\Datasets\VentasMensuales.csv', sep=';')

ventas.head(10)

ventas.info()

def transformData(data):
    # We create a copy of the original dataset 
    monthly_data = data.copy()

    # Drop the day indicator from the date column
    monthly_data.fecha = monthly_data.fecha.apply(lambda x: str(x)[:-3])

    # Sum sales per month 
    monthly_data = monthly_data.groupby('fecha')['ventas'].sum().reset_index()

    # We change the data type to date
    monthly_data.fecha = pd.to_datetime(monthly_data.fecha)
    
    # Divide by 1000 to put the '.' in the right place
    monthly_data.ventas /= 1000

    # We eliminate sales data that is only from one day, since it generates noise in the prediction
    monthly_data.drop(monthly_data[(monthly_data['fecha'] == '2018-09-01')].index, inplace=True)
    monthly_data.drop(monthly_data[(monthly_data['fecha'] == '2016-12-01')].index, inplace=True)

    return monthly_data

ventasM = transformData(ventas)

#We graph the data frame information to visualize the behavior of the data
plt.figure(figsize=(25,10))
sns.lineplot(x = 'fecha', y = 'ventas', data = ventasM)
plt.show()

def sales_per_day():
    fig, ax = plt.subplots(figsize=(7,4))
    plt.hist(ventasM.ventas, color='mediumblue')
    
    return ax.set(xlabel = "Sales Per day",
           ylabel = "Count",
           title = "Distrobution of Sales Per Day")
    
print(sales_per_day())

# Average monthly sells

# Overall
avg_monthly_sells = ventasM.ventas.mean()
print(f"Overall average monthly ventas: ${avg_monthly_sells}")

# Last 12 months (this will be the forecasted ventas)
avg_monthly_sells_12month = ventasM.ventas[-12:].mean()
print(f"Last 12 months average monthly ventas: ${avg_monthly_sells_12month}")


def time_plot(data, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(x_col, y_col, data=data, ax=ax, color='mediumblue', label='Total Sales')
    
    second = data.groupby(data.fecha.dt.year)[y_col].mean().reset_index()
    second.fecha = pd.to_datetime(second.fecha, format='%Y')
    sns.lineplot((second.fecha + datetime.timedelta(7*365/12)), y_col, data=second, ax=ax, color='red', label='Mean Sales')   
    
    return ax.set(xlabel = "fecha",
                  ylabel = "ventas",
                  title = title)
    
    
print(time_plot(ventasM, 'fecha', 'ventas', 'Monthly Sales Before Diff Transformation'))

#We create a copy dataframe where we will create a column that calculates the difference in monthly sales
def get_diff(data):
    data['diferenciaVentas'] =  data.ventas.diff()
    data = data.dropna()
    return data

stationary_df = get_diff(ventasM)

print(time_plot(stationary_df, 'fecha', 'diferenciaVentas', 'Monthly Sales After Diff Transformation'))

# Observing lags
def plots(data, lags=None):
    
    # Convert dataframe to datetime index
    dt_data = data.set_index('fecha').drop('ventas', axis=1)
    dt_data.dropna(axis=0)
    
    layout = (1, 3)
    raw  = plt.subplot2grid(layout, (0, 0))
    acf  = plt.subplot2grid(layout, (0, 1))
    pacf = plt.subplot2grid(layout, (0, 2))
    
    dt_data.plot(ax=raw, figsize=(12, 5), color='mediumblue')
    smt.graphics.plot_acf(dt_data, lags=lags, ax=acf, color='mediumblue')
    smt.graphics.plot_pacf(dt_data, lags=lags, ax=pacf, color='mediumblue')
    sns.despine()
    plt.tight_layout()

print(plots(stationary_df, lags=9))


#We create a datafrae that will store the retrospective period of the model which is 4 months
#create dataframe for transformation from time series to supervised
def generate_supervised(data):

    supervised_df = data.copy()
    #create column for each lag
    for i in range(1, 5):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['diferenciaVentas'].shift(i)

    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)

    return supervised_df

modelo = generate_supervised(stationary_df)


def tts(data):
    data = data.drop(['ventas','fecha'],axis=1)
    train, test = data[0:-7].values, data[-7:].values
    
    return train, test

train, test = tts(modelo)

print(train.shape)
print(test.shape)

def scale_data(train_set, test_set):
    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    
    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)
    
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()
    
    return X_train, y_train, X_test, y_test, scaler

X_train, y_train, X_test, y_test, scaler_object = scale_data(train, test)


def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):  
    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
    
    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    
    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index],x_test[index]],axis=1))
        
    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    
    #inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
    
    return pred_test_set_inverted

def load_original_df():
    #load in original dataframe without scaling applied
    original_df = pd.read_csv(r'C:\Users\Lina_\OneDrive\Escritorio\PrediccionVentas\Datasets\VentasMensuales2017.csv', sep = ';')
    original_df.fecha = original_df.fecha.apply(lambda x: str(x)[:-3])
    original_df = original_df.groupby('fecha')['ventas'].sum().reset_index()
    original_df.fecha = pd.to_datetime(original_df.fecha)
    original_df.ventas /= 1000
    # We eliminate sales data that is only from one day, since it generates noise in the prediction
    original_df.drop(original_df[(original_df['fecha'] == '2018-09-01')].index, inplace=True)
    original_df.drop(original_df[(original_df['fecha'] == '2016-12-01')].index, inplace=True)
    return original_df

def predict_df(unscaled_predictions, original_df):
    #create dataframe that shows the predicted sales
    result_list = []
    sales_date = list(original_df[-10:].fecha)
    act_sales = list(original_df[-10:].ventas)
    
    for index in range(0,len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_sales[index])
        result_dict['fecha'] = sales_date[index+1]
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)

    return df_result

'''
model_scores = {}

def get_scores(unscaled_df, original_df, model_name):
    rmse = np.sqrt(mean_squared_error(original_df.ventas[-7:], unscaled_df.pred_value[-7:]))
    mae = mean_absolute_error(original_df.ventas[-7:], unscaled_df.pred_value[-7:])
    r2 = r2_score(original_df.ventas[-7:], unscaled_df.pred_value[-7:])
    model_scores[model_name] = [rmse, mae, r2]

    return {f"RMSE: {rmse}",'\n' f"MAE: {mae}", '\n' f"R2 Score: {r2}"}
'''

'''
def plot_results(results, original_df, model_name):

       fig, ax = plt.subplots(figsize=(15,5))
       sns.lineplot(original_df.fecha, original_df.ventas, data=original_df, ax=ax, 
                     label='Original', color='mediumblue')
       sns.lineplot(results.fecha, results.pred_value, data=results, ax=ax, 
                     label='Predicted', color='Red')

       print('-----Results-----')
       print(results)

       return ax.set(xlabel = "fecha",
              ylabel = "ventas",
              title = f"{model_name} Sales Forecasting Prediction"), ax.legend()
'''


def run_model(train_data, test_data, model, model_name):
    
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    mod = model
    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)
    
    # Undo scaling to compare predictions against original data
    original_df = load_original_df()
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)

    return(unscaled_df)


res = run_model(train, test, LinearRegression(), 'LinearRegression')

