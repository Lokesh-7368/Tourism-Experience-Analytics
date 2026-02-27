# src/data_cleaning.py

import pandas as pd
import numpy as np
import os

BASE_PATH = 'data/raw/Tourism_Dataset/Tourism Dataset/'

def load_all_tables():
    """Load all Excel tables into DataFrames"""
    tables = {}

    tables['transaction'] = pd.read_excel(BASE_PATH + 'Transaction.xlsx')
    tables['user']        = pd.read_excel(BASE_PATH + 'User.xlsx')
    tables['city']        = pd.read_excel(BASE_PATH + 'City.xlsx')
    tables['item']        = pd.read_excel(
        BASE_PATH + 'Additional_Data_for_Attraction_Sites/Updated_Item.xlsx'
    )
    tables['type']        = pd.read_excel(BASE_PATH + 'Type.xlsx')
    tables['mode']        = pd.read_excel(BASE_PATH + 'Mode.xlsx')
    tables['continent']   = pd.read_excel(BASE_PATH + 'Continent.xlsx')
    tables['country']     = pd.read_excel(BASE_PATH + 'Country.xlsx')
    tables['region']      = pd.read_excel(BASE_PATH + 'Region.xlsx')

    return tables


def clean_transaction(df):
    """Clean transaction data"""
    print(f'Transaction original shape: {df.shape}')

    df = df.drop_duplicates(subset=['TransactionId'])
    df = df[df['VisitMode'] != 0]
    df = df[df['Rating'].between(1, 5)]
    df = df[df['VisitMonth'].between(1, 12)]
    df = df[df['VisitYear'].between(2010, 2024)]

    print(f'Transaction cleaned shape: {df.shape}')
    return df


def clean_user(df):
    """Clean user data"""
    print(f'User original shape: {df.shape}')

    df['CityId'] = df['CityId'].fillna(0).astype(int)
    df = df[df['ContinentId'] != 0]
    df = df.drop_duplicates(subset=['UserId'])

    print(f'User cleaned shape: {df.shape}')
    return df


def clean_city(df):
    """Clean city data"""
    df = df[df['CityId'] != 0]
    df['CityName'] = df['CityName'].str.strip()
    df = df[df['CityName'] != '-']
    return df


def merge_all(tables):
    """Merge all tables into one master DataFrame"""

    txn   = tables['transaction']
    user  = tables['user']
    item  = tables['item']
    cont  = tables['continent']
    reg   = tables['region']
    cntry = tables['country']
    city  = tables['city']
    atype = tables['type']
    mode  = tables['mode']

    df = txn.merge(user, on='UserId', how='left')
    df = df.merge(item, on='AttractionId', how='left')
    df = df.merge(cont, on='ContinentId', how='left')
    df = df.merge(reg[['RegionId', 'Region']], on='RegionId', how='left')
    df = df.merge(cntry[['CountryId', 'Country']], on='CountryId', how='left')

    city_renamed = city.rename(columns={'CityName': 'UserCityName'})
    df = df.merge(city_renamed[['CityId', 'UserCityName']], on='CityId', how='left')

    df = df.merge(atype, on='AttractionTypeId', how='left')

    mode_renamed = mode.rename(
        columns={'VisitModeId': 'VisitMode', 'VisitMode': 'VisitModeName'}
    )
    df = df.merge(mode_renamed, on='VisitMode', how='left')

    print(f'Master DataFrame shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')

    return df


if __name__ == '__main__':
    tables = load_all_tables()

    tables['transaction'] = clean_transaction(tables['transaction'])
    tables['user']        = clean_user(tables['user'])
    tables['city']        = clean_city(tables['city'])

    master = merge_all(tables)

    os.makedirs('data/processed', exist_ok=True)
    master.to_csv('data/processed/master_cleaned.csv', index=False)

    print('\n✅ Cleaned data saved to data/processed/master_cleaned.csv')
    print(master.head(3).to_string())