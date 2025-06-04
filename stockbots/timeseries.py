from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt

from stockbots.data_loader import load_final_time_df


final_time_df = load_final_time_df()

#Here we will use the auto_arima function to complete steps 1-5, as this function streamlines the process of finding the most adequate ARIMA model.
#It determines the best ARIMA model for our time series data by balancing the model complexity with the fit quality, typically assessed using AIC
#or BIC metrics. Essentially, the auto_arima function automates the process of ARIMA modeling which otherwise would require manual iteration and 
#testing of different combinations of these hyperparameters.


best_arima_model = auto_arima(final_time_df["S&P 500 Index Price"], start_p = 1, start_q = 1, max_p = 5, max_q = 5, m = 12, start_P = 0, 
                              seasonal = True, d = None, D = 1, trace = True, error_action = "ignore", suppress_warnings = True, stepwise = True)

best_arima_model.summary()



#Finding the best ARIMA model for the S&P 500 Index WITHOUT the seasonal componenet

best_arima_model_X = auto_arima(final_time_df["S&P 500 Index Price"], start_p = 1, start_1 = 1, max_p = 5, max_q = 5,
                                d = None, trace = True, error_action = "ignore", suppress_warnings = True, stepwise = True)


#Finding the best ARIMA model for the Apple time series (which is an individual stock unlike the S&P500 Index)

best_arima_model = auto_arima(final_time_df["AAPL"], start_p = 1, start_q = 1, max_p = 5, max_q = 5, m = 12, start_P = 0, 
                              seasonal = True, d = None, D = 1, trace = True, error_action = "ignore", suppress_warnings = True, stepwise = True)

best_arima_model.summary()




#We will now conduct an out of sample forecast to evaluate the models' potential effectiveness in the real-world. We will forecast both the 
#S&P 500 Index series and Apple stock series two years in the future.

start = len(final_time_df)
end = (len(final_time_df - 1) + 12) 

final_ARIMA_Model_SP500 = SARIMAX(final_time_df["S&P 500 Index Price"], order = (0, 1, 1), seasonal_order = (1, 1, 1, 12)).fit()
final_ARIMA_Model_Apple = SARIMAX(final_time_df["AAPL"], order = (0, 1, 0), seasonal_order = (0, 1, 1, 12)).fit()
final_ARIMA_Model_SP500_Y = SARIMAX(final_time_df["S&P 500 Index Price"], order = (0, 1, 1), seasonal_order = (1, 1, 0, 12)).fit()

SP500_forecast = final_ARIMA_Model_SP500.predict(start, end + 24, typ = "levels").rename("Forecast")
Apple_forecast = final_ARIMA_Model_Apple.predict(start, end + 24, typ = "levels").rename("Forecast")
SP500_forecast_Y = final_ARIMA_Model_SP500_Y.predict(start, end + 24, typ = "levels").rename("Forecast")


#SP500 Forecast Visualization (Out of Sample)

SP500_forecast.plot(legend = True)
final_time_df["S&P 500 Index Price"].plot(legend = True)
plt.show()




#Apple Forevast Visualization (Out of Sample)

Apple_forecast.plot(legend = True)
final_time_df["AAPL"].plot(legend = True)
plt.show()



#S&P 500 Index Forecast Visualization (Out of Sample)

SP500_forecast_Y.plot(legend = True)
final_time_df["S&P 500 Index Price"].plot(legend = True)
plt.show()


