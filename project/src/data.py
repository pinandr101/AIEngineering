"""Загрузка и агрегация данных по категориям."""
import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def aggregate_sales_by_category(df: pd.DataFrame) -> dict:
    """
    Возвращает словарь {category: DataFrame с колонками date, sales}.
    """
    categories = df['Category'].unique()
    result = {}
    for cat in categories:
        cat_df = df[df['Category'] == cat].copy()
        daily = cat_df.groupby('Date')['Units Sold'].sum().reset_index()
        daily.columns = ['date', 'sales']
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.sort_values('date').reset_index(drop=True)
        result[cat] = daily
    return result
