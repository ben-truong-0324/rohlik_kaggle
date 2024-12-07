import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import time
from rohlik_sales.config import *
import rohlik_sales.plots as plots

# from config import *
# import rohlik_sales.data_plots
import pickle



def get_sp500_data(dataset, do_scaling, do_pca, do_panda):
    if "sp500" in DATASET_SELECTION:
        # Load the data into a Pandas DataFrame
        sp500_df = dataset
        sp500_df.columns = sp500_df.iloc[0]
        sp500_df = sp500_df[1:]
        sp500_df = sp500_df.dropna(subset=['Close'])
        sp500_df['Close'] = pd.to_numeric(sp500_df['Close'], errors='coerce')
        sp500_df['Volume'] = pd.to_numeric(sp500_df['Volume'], errors='coerce')
        
        # Calculate Moving Averages
        sp500_df['MA_10d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=10).mean()) / x)
        sp500_df['MA_25d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=25).mean()) / x)
        sp500_df['MA_50d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=50).mean()) / x)
        sp500_df['MA_200d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=200).mean()) / x)

        tolerance = 1e-5
        sp500_df['MA_vol_10d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=10).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_25d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=25).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_50d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=50).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_200d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=200).mean()) / (x.rolling(window=10).mean() + tolerance))
        
        # Calculate MACD
        def calculate_macd(df, short_window, long_window, signal_window, tolerance=1e-4):
            short_ema = sp500_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=short_window, adjust=False).mean())
            long_ema = sp500_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=long_window, adjust=False).mean())
            macd = short_ema - long_ema
            signal_line = macd.ewm(span=signal_window, adjust=False).mean()
            macd = macd.clip(lower=tolerance) 
            return (macd - signal_line)/macd
        # sp500_df['macd'], sp500_df['signal_line'] = calculate_macd(sp500_df, short_window, long_window, signal_window)
        sp500_df['macd_signal_ratio'] = calculate_macd(sp500_df, short_window, long_window, signal_window)

        
        # Calculate Bollinger Bands

        def calculate_bollinger_ratio(df, window_size, tolerance=1e-8):
            bollinger_mid = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=window_size).mean())
            bollinger_std = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=window_size).std())
            bollinger_std = bollinger_std.clip(lower=tolerance)  # Clip the standard deviation to a minimum value
            return (df['Close'] - bollinger_mid) / (bollinger_std * 2)

        sp500_df['bollinger_band_ratio'] = calculate_bollinger_ratio(sp500_df, window_size)
       
        # Calculate RSI
        def calculate_rsi(df, period):
            
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            exponentially_weighted_moving_average_gain = gain.ewm(com=period - 1, adjust=False).mean()
            exponentially_weighted_moving_average_loss = loss.ewm(com=period - 1, adjust=False).mean()
            rs = exponentially_weighted_moving_average_gain / exponentially_weighted_moving_average_loss
            return 100 - (100 / (1 + rs))
        sp500_df['rsi'] = calculate_rsi(sp500_df, rsi_period)
       

        # Calculate ROC
        sp500_df['roc'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: ((x - x.shift(roc_period)) /  (x.shift(roc_period) + tolerance) ).clip(lower=tolerance)  * 100)

        # Calculate DoD Delta
        sp500_df['dod_delta'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: np.where(x < x.shift(-1), 1, 0))
        sp500_df['dod5_delta'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: np.where(x < x.shift(-5), 1, 0))

        # Drop NaN values
        sp500_df.dropna(inplace=True)
        columns_to_drop = ['Open', 'Close', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Ticker']
        sp500_df.drop(columns=columns_to_drop, inplace=True)
        print("Columns:", sp500_df.columns)
        print("Shape:", sp500_df.shape)
        print(sp500_df.head(20))

        with open(SP500_PROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(sp500_df, f)
        print(f"Results saved to {SP500_PROCESSED_DATA_PATH}")

        def print_df_stats(df):
            print("Shape:", df.shape)
            print("Columns:", df.columns)
            print("Data Types:", df.dtypes)

            for col in df.columns:
                print(f"\nColumn: {col}")
                print("Unique Values:", df[col].nunique())
                if pd.api.types.is_numeric_dtype(df[col]):
                    print("Min:", df[col].min())
                    print("Max:", df[col].max())
                    print("Median:", df[col].median())
                    print("Example Values:", df[col].sample(3).tolist())
                else:
                    print("Example Values:", df[col].sample(3).tolist())

        

        print_df_stats(sp500_df)



    else: 
        raise ValueError("Invalid dataset specified. Check config.py")
          
    
    #probably need to deprecate
    if do_scaling:
        scaling=StandardScaler()
        scaling.fit(X_df) 
        # X_df=scaling.transform(X_df)
        X_df_scaled = scaling.transform(X_df)  # Scaled data is a NumPy array
        X_df = pd.DataFrame(X_df_scaled, )
        print("Scaling implemneted")
    if do_pca:    
        pca = PCA(n_components=.95)  # get enough for 95% of var explained
        pca.fit(X_df)
        X_df_pca = pca.fit_transform(X_df)  # PCA transforms into NumPy array
        X_df = pd.DataFrame(X_df_pca, index=X_df.index, 
                        columns=[f'PC{i+1}' for i in range(X_df_pca.shape[1])])
        print("PCA implemneted")
    if do_panda:
        pass

    return sp500_df



def get_data():
    print(f"Getting data for {DATASET_SELECTION}")
    if "kaggle_rohlik_sales" in DATASET_SELECTION:
        # Load the DataFrame from the pickle file
        try:
            calendar = pd.read_csv(CALENDAR_PATH)
            inventory = pd.read_csv(INVENTORY_PATH)
            sales_train = pd.read_csv(TRAIN_PATH)
            print("Accessed .csv in data folder")

            # Extract product and category components
            try:
                # Debugging: Check for None or missing values in the relevant columns
                print("Checking for None or missing values in 'name' and 'L4_category_name_en'")
                print(inventory[['name', 'L4_category_name_en']].isnull().sum())
                
                # Debugging: Display a sample of rows with missing or irregular data
                missing_data = inventory[inventory['name'].isnull() | inventory['L4_category_name_en'].isnull()]
                if not missing_data.empty:
                    print("Rows with missing data in 'name' or 'L4_category_name_en':")
                    print(missing_data)
                
                # Splitting the columns
                print("Splitting 'name' into 'product_name' and 'product_num'")
                inventory['product_name'] = inventory['name'].str.rsplit('_', 1).str[0]
                inventory['product_num'] = inventory['name'].str.rsplit('_', 1).str[1]

                
                print("Splitting 'L4_category_name_en' into 'cat_name' and 'cat_num'")
                inventory['cat_name'] = inventory['L4_category_name_en'].str.rsplit('_L4_', 1).str[0]
                inventory['cat_num'] = inventory['L4_category_name_en'].str.rsplit('_L4_', 1).str[1]
            except Exception as e:
                print("An error occurred during the splitting process.")
                print(f"Error: {e}")
                print("Displaying first few rows of inventory for debugging:")
                print(inventory.head())


            inventory = inventory[['unique_id', 'warehouse', 'product_name', 'product_num', 'cat_name', 'cat_num']]
            sales_train = sales_train.merge(inventory, on=['unique_id', 'warehouse'], how='left')
            print("processed and merged inventory data")
            sales_train = sales_train.merge(calendar[['date', 'holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays']],
                                            on='date', how='left')
            print("merged calendar data")
            sales_train['sales_whole'] = sales_train.apply(
                lambda row: row['sales'] / row['availability'] if row['availability'] < 1.0 else row['sales'], axis=1
            )
            print("transformed sales_whole for y")
            print(sales_train.columns)
            Y_df = sales_train['sales_whole']  # Target variable
            X_df = sales_train.drop(columns=['sales', 'availability', 'sales_whole'])

            print(f"X shape: {X_df.shape}")
            print(f"Columns of the X_df: {X_df.columns.tolist()}")
            print(f"Y shape: {Y_df.shape}")
            # return X, Y
        except FileNotFoundError:
            print(f"Error: The file '{TRAIN_PATH}' was not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    else: 
        print("#"*18)
        raise ValueError("Invalid dataset specified. Check config.py")


    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)  # Convert to DataFrame
    if Y_df.ndim == 1:
        # If it's 1D, convert to Pandas Series
        Y_df = pd.Series(Y_df)
    else:
        # If it's 2D, convert to Pandas DataFrame
        Y_df = pd.DataFrame(Y_df)
    return X_df, Y_df

def graph_raw_data(X_df, Y_df):
    raw_data_outpath =OUTPUT_DIR_RAW_DATA_A3
    # Check if Y_df is multi-label (2D) or single-label (1D)
    if Y_df.ndim == 1:  # Single-label
        if not os.path.exists(f'{raw_data_outpath}/feature_heatmap.png'):
            # Plot class imbalance, feature violin, heatmap, etc.
            plots.graph_class_imbalance(Y_df, 
                                             f'{raw_data_outpath}/class_imbalance.png')
            plots.graph_feature_violin(X_df, Y_df, 
                                             f'{raw_data_outpath}/feature_violin.png')
            plots.graph_feature_heatmap(X_df, Y_df,
                                             f'{raw_data_outpath}/feature_heatmap.png')
            plots.graph_feature_histogram(X_df, 
                                             f'{raw_data_outpath}/feature_histogram.png')
            plots.graph_feature_correlation(X_df, Y_df,
                                             f'{raw_data_outpath}/feature_correlation.png')
            plots.graph_feature_cdf(X_df, 
                                             f'{raw_data_outpath}/feature_cdf.png')
    else:  # Multi-label
        if not os.path.exists(f'{raw_data_outpath}/feature_heatmap.png'):
            # Handle multi-label plotting differently if necessary
            pass



def inspect_pickle_content(pkl_path):
    """
    Inspect contents of a pickle file, showing structure and samples
    """
    print(f"\nInspecting pickle file: {pkl_path}")
    print("=" * 80)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    print("\n1. Basic Info:")
    print(f"Type of loaded data: {type(data)}")
    print(f"Is empty? {not bool(data)}")
    
    if isinstance(data, dict):
        print(f"\n2. Dictionary Structure:")
        print(f"Number of top-level keys: {len(data)}")
        print("\nTop-level keys:")
        for key in list(data.keys())[:5]:  # First 5 keys
            print(f"- {key} ({type(key)})")
            
        # Sample a random key for deeper inspection
        if data:
            sample_key = next(iter(data))
            print(f"\n3. Sample value for key '{sample_key}':")
            sample_value = data[sample_key]
            print(f"Type: {type(sample_value)}")
            
            # If the value is also a dictionary, show its structure
            if isinstance(sample_value, dict):
                print("Nested dictionary structure:")
                for k, v in list(sample_value.items())[:3]:  # First 3 items
                    print(f"- {k}: {type(v)}")
                    if isinstance(v, (dict, list)):
                        print(f"  Length: {len(v)}")
                    try:
                        print(f"  Sample: {str(v)[:100]}...")  # First 100 chars
                    except:
                        print("  Sample: [Cannot display sample]")
            
            # If it's a list or array, show some info
            elif isinstance(sample_value, (list, np.ndarray)):
                print(f"Length: {len(sample_value)}")
                print("First few elements:")
                print(sample_value[:3])
    
    # If it's not a dictionary, show appropriate info
    else:
        if isinstance(data, (list, np.ndarray)):
            print(f"\nArray/List Info:")
            print(f"Length: {len(data)}")
            print("First few elements:")
            print(data[:3])
        else:
            print("\nData sample:")
            try:
                print(str(data)[:200])  # First 200 chars
            except:
                print("[Cannot display data sample]")

if __name__ == "__main__":
    ###################
    pass

