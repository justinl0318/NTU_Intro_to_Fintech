import yfinance as yf

# Define the stock symbol and date range
stock_symbol = "0050.TW"
start_date = "2019-01-02"
end_date = "2023-12-22"  # Replace with today's date or any desired end date

# Download historical data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Save the data as an Excel file
csv_file_name = "0050.TW-short.csv"
stock_data.to_csv(csv_file_name)

