import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_final_time_df
from data_loader import load_fundamentals_df
import numpy as np

#Visualization 1: 
#Computing the autocorrelation function for each of our 7 companies' time series data with a lag of 6 (half a year)
from statsmodels.graphics.tsaplots import plot_acf

final_time_df = load_final_time_df()
fundamentals_df = load_fundamentals_df()

magnificent_seven = ["AAPL", "MSFT", "META", "TSLA", "NVDA", "AMZN", "GOOGL"]

for company in magnificent_seven:
    fig, ax = plt.subplots(figsize = (12, 4))
    plot_acf(final_time_df[company], lags = 6, alpha = 0.05, ax = ax)
    ax.set_title(f"Autocorrelation for {company}")
    plt.show()



#Visualization 2: Stock prices from 2010 to 2022 for the 7 companies in the "Magnificent 7".
for stock in magnificent_seven:
    plt.subplot(2, 1, 1)
    plt.plot(final_time_df[stock], label = stock + "Stock Price")
    plt.legend(loc = "best", fontsize = "small")
    plt.show()



#Visualization 3: Monthly Returns for each stock in the "Magnificent 7"
for company in magnificent_seven:
    final_time_df[f"{company}_Monthly_Returns"] = final_time_df[company].pct_change(periods = 1).mul(100)
    final_time_df[f"{company}_Monthly_Returns"].plot(title = f"{company} Monthly Returns")
    plt.ylabel("Monthly Returns [%]")
    plt.show()


#Visualization 1

from matplotlib.ticker import FuncFormatter

plt.figure(figsize = (10, 6))
plt.scatter(x = fundamentals_df["Market Cap"], y = fundamentals_df["Net Income"], s = np.array(fundamentals_df["Price"]) * 2, alpha = 0.8)

plt.xscale('log')
plt.xlabel("Market Cap [In Billions]")
plt.ylabel("Net Income [In Billions]")
plt.title("Comparing Company Valuation to Profitability for S&P 500 companies")

def billions_formatter(x, pos):
    return '%1.1fB' % (x * 1e-9)


formatterB = FuncFormatter(billions_formatter)
plt.gca().xaxis.set_major_formatter(formatterB)
plt.gca().yaxis.set_major_formatter(formatterB)

plt.grid(True)

plt.show()

#Visualization 2

plt.figure(figsize = (10, 6))
plt.hist(fundamentals_df["Price/Earnings"], bins = 50, color = "blue", alpha = 0.8)
plt.xlabel("P/E Ratio")
plt.ylabel("Frequency")
plt.title("Histogram of P/E ratios for the S&P 500")

plt.xlim(-40, 100)
plt.grid(True)
plt.show()



#Visualization 3

sector_market_cap = fundamentals_df.groupby("Sector")["Market Cap"].sum()
sector_market_cap.plot(kind = "bar")

plt.title("Total Market Cap by Sector")
plt.xlabel("Sector")
plt.ylabel("Total Market Cap ($)")
plt.xticks(rotation = 45, ha = "right")

def trillions_formatter(x, pos):
    return '%1.1fT' % (x * 1e-12)
 
formatterT = FuncFormatter(trillions_formatter)
plt.gca().yaxis.set_major_formatter(formatterT)
plt.tight_layout()
plt.show()



#Visualization 4:

fundamentals_df.boxplot(column = "Price/Earnings", by = "Sector", figsize = (12, 6), vert = False)
plt.title("P/E Ratios by Sector")
plt.xlabel("Price/Earnings Ratio")
plt.xlim(-150, 150)
plt.ylabel("Sector")
plt.show()





#Visualization 5: 

fundamentals_df.boxplot(column = "Dividend Yield", by = "Sector", vert = False, figsize = (12, 6))
plt.title("Dividend Yield By Sector")
plt.xlabel("Dividend Yield [In %]")
plt.ylabel("Sector")
plt.show()


#Beta Value Visualization

sector_beta = fundamentals_df.groupby("Sector")["Beta"].mean()
sector_beta.plot(kind = "bar")

plt.title("Average Beta Value For Each Sector in S&P500")
plt.ylabel("Average Beta (12 year span from 2010 - 2022)")
plt.xlabel("Sector")
plt.xticks(rotation = 45, ha = "right")

plt.show()



