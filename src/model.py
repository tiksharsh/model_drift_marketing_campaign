import os
import sys
import random

import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_metadata.proto.v0 as schema_pb2
import tensorflow.python.lib.io as file_io

from dataclasses import dataclass


@dataclass

class ModelDriftConfig:

    train_df_path = os.path.join("data/raw","marketing_campaign.csv")
    col = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome','Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

class ModelDrift:

    def __init__(self):
        self.drift_config = ModelDriftConfig()

    def data_read(self):
        pass

    def data_split(self):
        # parase pandas datetime type on Dt_Customer
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format="%d-%m-%Y")

        # get the minimum and maximum dates in your date column
        min_date = df['Dt_Customer'].min()
        max_date = df['Dt_Customer'].max()

        # print out the range of dates
        print('Date Range: {} - {}'.format(min_date, max_date))
        # set the date column as the index
        df.set_index('Dt_Customer', inplace=True)

        # group the DataFrame by yearly frequency
        years = df.groupby(pd.Grouper(freq = 'Y'))

        # print out the data for each quarter
        datasets = []
        for i, (year,data) in enumerate(years):
        datasets.append(data)
        pass

    def get_data_yearwise(self):
        # get the 2012 costumers and drop the index
        early_data = datasets[0]
        early_data = early_data.reset_index(drop=True)
        # get the 2014 costumers and drop the index
        latest_data = datasets[-1]
        latest_data = latest_data.reset_index(drop=True)

    def future_marital_status(self, v):
        '''
        Increase the number of married customers by 20%
        '''

        if v == "Married":
            return v
        else:
            coin_toss = random.uniform(0,1)
            if coin_toss > 0.2:
            return v
            return "Married"

        # increase income by 20-30% for 2014 costumers
    latest_data['Income'] = latest_data['Income'].apply(lambda v: v * (1 + random.uniform(.2, .3)))

        # increase the number of married costumers in 2014 by 20%
    latest_data['Marital_Status'] = latest_data['Marital_Status'].apply(future_marital_status)
