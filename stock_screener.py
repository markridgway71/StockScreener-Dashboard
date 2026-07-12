import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

# Sidebar inputs - change these in the browser, no code edits needed
ticker = st.sidebar.text_input("Ticker", value="CAP.PA")
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)


def stockAnalysis(ticker, period):

    # Download data
    df = yf.download(ticker, period=period, interval="1d")

    # Flatten multi-level column headers if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if df.empty:
        st.error("No data returned for " + ticker + ". Check the ticker symbol.")
        return

    st.subheader("Data Preview")
    st.dataframe(df.tail(10))

    with st.expander("Summary Statistics"):
        st.dataframe(df.describe())

    # stock_info = yf.Ticker(ticker).info
    # trailing_pe = stock_info.get("trailingPE")
    # forward_pe = stock_info.get("forwardPE")

    # col1, col2 = st.columns(2)
    # col1.metric("Trailing P/E", f"{trailing_pe:.1f}" if trailing_pe is not None else "n/a")
    # col2.metric("Forward P/E", f"{forward_pe:.1f}" if forward_pe is not None else "n/a")

    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    avg_rsi = df['RSI'].mean()
    pct_from_avg_rsi = ((df['RSI'] - avg_rsi) / avg_rsi) * 100

    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(pct_from_avg_rsi, label='% from ' + period + ' Avg RSI', color='purple', alpha=0.7)
    plt.axhline(y=0, color='orange', linestyle='--', label=period + ' Avg (0%)')
    plt.fill_between(pct_from_avg_rsi.index, pct_from_avg_rsi, 0,
                     where=(pct_from_avg_rsi >= 0), color='green', alpha=0.2, label='Above Avg')
    plt.fill_between(pct_from_avg_rsi.index, pct_from_avg_rsi, 0,
                     where=(pct_from_avg_rsi < 0), color='red', alpha=0.2, label='Below Avg')
    plt.title(ticker + " RSI vs " + period + " Average (%)")
    plt.ylabel("% Difference")
    plt.xlabel("Date")
    plt.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    # Calculate Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=30).std() * (252 ** 0.5)

    avg_Volatility = df['Volatility'].mean()
    pct_from_avg_Volatility = ((df['Volatility'] - avg_Volatility) / avg_Volatility) * 100

    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(pct_from_avg_Volatility, label='% from ' + period + ' Avg Volatility', color='purple', alpha=0.7)
    plt.axhline(y=0, color='orange', linestyle='--', label=period + ' Avg (0%)')
    plt.fill_between(pct_from_avg_Volatility.index, pct_from_avg_Volatility, 0,
                     where=(pct_from_avg_Volatility >= 0), color='green', alpha=0.2, label='Above Avg')
    plt.fill_between(pct_from_avg_Volatility.index, pct_from_avg_Volatility, 0,
                     where=(pct_from_avg_Volatility < 0), color='red', alpha=0.2, label='Below Avg')
    plt.title(ticker + " Volatility vs " + period + " Average (%)")
    plt.ylabel("% Difference")
    plt.xlabel("Date")
    plt.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    # Daily Returns +/- 2 Standard Deviations
    df['Daily_Return_Pct'] = df['Close'].pct_change() * 100
    avg_return = df['Daily_Return_Pct'].mean()
    std_return = df['Daily_Return_Pct'].std()
    upper_band = avg_return + (2 * std_return)
    lower_band = avg_return - (2 * std_return)

    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(df['Daily_Return_Pct'], color='steelblue', alpha=0.7, linewidth=1, label='Daily Return %')
    plt.axhline(y=avg_return, color='orange', linestyle='--', label=f'Mean: {avg_return:.2f}%')
    plt.axhline(y=upper_band, color='red', linestyle='--', label=f'+2 StDev: {upper_band:.2f}%')
    plt.axhline(y=lower_band, color='green', linestyle='--', label=f'-2 StDev: {lower_band:.2f}%')
    plt.fill_between(df['Daily_Return_Pct'].index, upper_band, lower_band,
                     alpha=0.1, color='yellow', label='Normal Range')
    plt.title(ticker + " Daily Returns +/- 2 Standard Deviations (" + period + ")")
    plt.ylabel("Daily Return %")
    plt.xlabel("Date")
    plt.legend()
    st.pyplot(fig3)
    plt.close(fig3)

    

    # --- Price as % of 1-year high ---

#   Rolling 1-year high (252 trading days)
    df["52w_high"] = df["Close"].rolling(window=252).max()

    # Close price expressed as % of that high
    df["pct_of_high"] = df["Close"] / df["52w_high"] * 100

    # Plot it
    fig_high, ax_high = plt.subplots(figsize=(12, 6))
    ax_high.plot(df.index, df["pct_of_high"], color="steelblue", linewidth=1.5)
    ax_high.axhline(100, color="green", linestyle="--", linewidth=1, label="At 1-year high (100%)")
    ax_high.set_title("CAP.PA — Close Price as % of 1-Year High")
    ax_high.set_ylabel("% of 52-week high")
    ax_high.set_xlabel("Date")
    ax_high.legend()
    ax_high.grid(True, alpha=0.3)

    st.pyplot(fig_high)
    plt.close(fig_high)

    

    # Price with +/- 2 Standard Deviation Bands
    avg_price = df['Close'].mean()
    std_price = df['Close'].std()
    upper_band_price = avg_price + (2 * std_price)
    lower_band_price = avg_price - (2 * std_price)

    fig5 = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], color='steelblue', alpha=0.7, linewidth=1, label='Daily Close')
    plt.axhline(y=avg_price, color='orange', linestyle='--', label=f'Mean: {avg_price:.2f}')
    plt.axhline(y=upper_band_price, color='red', linestyle='--', label=f'+2 StDev: {upper_band_price:.2f}')
    plt.axhline(y=lower_band_price, color='green', linestyle='--', label=f'-2 StDev: {lower_band_price:.2f}')
    plt.fill_between(df['Close'].index, upper_band_price, lower_band_price,
                     alpha=0.1, color='yellow', label='Normal Range')
    plt.title(ticker + " Price +/- 2 Standard Deviations (" + period + ")")
    plt.ylabel("Price (€)")
    plt.xlabel("Date")
    plt.legend()
    st.pyplot(fig5)
    plt.close(fig5)


# Run the analysis - note: NOT indented, so it sits outside the function
stockAnalysis(ticker, period)