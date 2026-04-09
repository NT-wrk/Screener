import streamlit as st
import yfinance as yf
from finvizfinance.quote import finvizfinance
import pandas as pd

# --- Watchlist ---
st.title ("Stock Screener")

# Watchlist im Session State speichern
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAOI","TSEM","AXTI","SIVE","AMBA","LITE","MTSI","NVDA","AMD","INTC"]

# Ticker zur Watchlist hinzufügen
new_ticker = st.text_input("Add a ticker to the watchlist:")
if st.button("Add to Watchlist"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker.upper())
        st.success(f"{new_ticker.upper()} added to watchlist.")
    elif new_ticker in st.session_state.watchlist:
        st.warning(f"{new_ticker.upper()} is already in the watchlist.")
    else:
        st.error("Please enter a valid ticker.")

# Ticker aus der Watchlist entfernen
ticker_to_remove = st.selectbox("Remove a ticker from the watchlist:", st.session_state.watchlist)
if st.button("Remove from Watchlist"):
    if ticker_to_remove in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker_to_remove)
        st.success(f"{ticker_to_remove} removed from watchlist.")
    else:
        st.error("Please select a valid ticker to remove.")
    
st.divider()

# --- Stock Analysis ---

def analyze(symbol):
    try:
        fvz = finvizfinance(symbol).ticker_fundament()
        yf_ticker = yf.Ticker(symbol)
        hist = yf_ticker.history(period="45d")

        # volume ratio
        avg_volume = hist['Volume'][:-2].mean()
        current_volume = hist['Volume'][-2]
        volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0

        # short float (Finviz)
        short_float = fvz.get('Short Float', 'N/A')
        # short float (Yahoo Finance)
        short_float_yf = yf_ticker.info.get('shortPercentOfFloat', 'N/A')

        # P/S und P/E Ratio (Finviz)
        ps_ratio = fvz.get('P/S', 'N/A')
        pe_ratio = fvz.get('P/E', 'N/A')

        # Analyst PT (Yahoo Finance)
        analyst_pt = yf_ticker.info.get('targetMeanPrice', 'N/A')
        current = yf_ticker.info.get('currentPrice', 'N/A')
        upside = (analyst_pt - current) / current * 100 if isinstance(analyst_pt, (int, float)) and isinstance(current, (int, float)) else 'N/A'

        return {
            'symbol': symbol,
            'current_price': current,
            'P/S': ps_ratio,
            'P/E': pe_ratio,
            'Short Float (Finviz)': short_float,
            'Short Float (Yahoo Finance)': short_float_yf,
            'Volume Ratio': round(volume_ratio, 2),
            'Analyst PT': analyst_pt,
            'Upside %': round(upside, 2) if upside != 'N/A' else 'N/A'
        }       
    except Exception as e:
            st.error(f"Error analyzing {symbol}: {e}")
            return None
    
# Alle Ticker in der Watchlist analysieren
if st.button("Analyze Watchlist"):
    results = []
    for ticker in st.session_state.watchlist:
        result = analyze(ticker)
        if result:
            results.append(result)
    # Ergebnisse in einem DataFrame anzeigen
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)    
