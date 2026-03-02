import pandas as pd
import numpy as np
import os

# Configuration
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
ETFS = {
    'XLK': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xlk.us.txt"),
    'XLE': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xle.us.txt"),
    'XLF': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xlf.us.txt"),
    'XLV': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xlv.us.txt"),
    'XLI': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xli.us.txt")
}
FEDFUNDS_PATH = os.path.join(RAW_DATA_DIR, "FEDFUNDS.csv")
EVENTS_PATH = os.path.join(RAW_DATA_DIR, "events.csv")

def load_stooq_data(filepath, symbol):
    """Loads and cleans Stooq historical OHLCV data."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    # Clean column names
    df.columns = [c.lower().strip('<>') for c in df.columns]
    # Convert date and set as index
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # Calculate daily log returns
    df[f'{symbol}_log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Keep only relevant columns
    cols_to_keep = ['close', 'vol', f'{symbol}_log_ret']
    df = df[cols_to_keep]
    df.columns = [f'{symbol}_{c}' if c != f'{symbol}_log_ret' else c for c in df.columns]
    
    return df

def preprocess_all_data():
    """Main pipeline for data integration."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 1. Load all ETFs
    etf_dfs = []
    for symbol, path in ETFS.items():
        df = load_stooq_data(path, symbol)
        if df is not None:
            etf_dfs.append(df)
    
    if not etf_dfs:
        print("Error: No ETF data loaded.")
        return
    
    # Merge all ETFs on date
    master_df = etf_dfs[0]
    for df in etf_dfs[1:]:
        master_df = master_df.join(df, how='outer')
    
    # 2. Integrate FEDFUNDS
    if os.path.exists(FEDFUNDS_PATH):
        fed_df = pd.read_csv(FEDFUNDS_PATH)
        fed_df['observation_date'] = pd.to_datetime(fed_df['observation_date'])
        fed_df.set_index('observation_date', inplace=True)
        fed_df.sort_index(inplace=True)
        
        # Align with master_df dates (forward fill monthly rate to daily)
        master_df = master_df.join(fed_df, how='left')
        master_df['FEDFUNDS'] = master_df['FEDFUNDS'].ffill()
    else:
        print(f"Warning: FEDFUNDS not found at {FEDFUNDS_PATH}")

    # 3. Integrate Events (as a marker column)
    if os.path.exists(EVENTS_PATH):
        events_df = pd.read_csv(EVENTS_PATH)
        events_df['date'] = pd.to_datetime(events_df['date'])
        # Create event marker columns
        events_map = events_df.set_index('date')['event'].to_dict()
        master_df['event_marker'] = master_df.index.map(lambda d: events_map.get(d, np.nan))
    
    # Final cleanup
    master_df.dropna(how='all', inplace=True) # Drop rows where all ETF data is missing
    
    # Save output
    output_file = os.path.join(PROCESSED_DATA_DIR, "master_sector_data.csv")
    master_df.to_csv(output_file)
    print(f"Successfully saved master dataset to {output_file}")
    return master_df

if __name__ == "__main__":
    preprocess_all_data()
