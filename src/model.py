
# https://medium.com/geekculture/automated-data-drift-detection-for-machine-learning-pipelines-5c95aca45e1c

import os
import sys
import random
import pandas as pd

import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_metadata.proto.v0 as schema_pb2
import tensorflow.python.lib.io as file_io

from dataclasses import dataclass

from tensorflow.python.lib.io import file_io
from google.protobuf import text_format

# sys.path.append("src")

from logger import logging
from exception import CustomException


@dataclass

class ModelDriftConfig:

    train_df_path = os.path.join("data/raw","marketing_campaign.csv")
    col = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome','Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

    directory = 'schema-stats'
    if not os.path.exists(directory):
        os.makedirs(directory)

    stats_path = os.path.join(directory, 'schema.txtpb')   
    schema_path = os.path.join(directory,"data_stats.txtpb")


    logging.info(f"Training file path \n {train_df_path}")
    logging.info('-'*80)


        

class ModelDrift:

    def __init__(self):
        self.drift_config = ModelDriftConfig()

    def data_read(self):
        
        df = pd.read_csv(self.drift_config.train_df_path, sep='\t')
        df = df[self.drift_config.col]
        
        logging.info(f"Raw data \n {df.head()}")
        logging.info('-'*80)

        return df

    def data_split(self):
        
        df = self.data_read()
        # parase pandas datetime type on Dt_Customer
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format="%d-%m-%Y")

        # get the minimum and maximum dates in your date column
        min_date = df['Dt_Customer'].min()
        max_date = df['Dt_Customer'].max()

        # logging.info out the range of dates
        logging.info('Date Range: {} - {}'.format(min_date, max_date))
        # set the date column as the index
        df.set_index('Dt_Customer', inplace=True)

        # group the DataFrame by yearly frequency
        years = df.groupby(pd.Grouper(freq = 'Y'))

        # logging.info out the data for each quarter
        datasets = []
        for i, (year,data) in enumerate(years):
            datasets.append(data)

        # get the 2012 costumers and drop the index
        early_data = datasets[0]
        early_data = early_data.reset_index(drop=True)
        # get the 2014 costumers and drop the index
        latest_data = datasets[-1]
        latest_data = latest_data.reset_index(drop=True)

        # increase income by 20-30% for 2014 costumers
        latest_data['Income'] = latest_data['Income'].apply(lambda v: v * (1 + random.uniform(.2, .3)))

        # increase the number of married costumers in 2014 by 20%
        latest_data['Marital_Status'] = latest_data['Marital_Status'].apply(self.future_marital_status)

        logging.info(f"early_data \n {early_data.head()}")
        logging.info('-'*80)
        logging.info(f"latest_data \n {latest_data.head()}")
        logging.info('-'*80)
        
        return early_data, latest_data
    

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
    
    
    def visualize_stats(self, early_data, latest_data):
        import tensorflow as tf
        import tensorflow_data_validation as tfdv
        import pandas as pd
        # generate statistics for 2012 
        early_data_stats = tfdv.generate_statistics_from_dataframe(early_data)
        # generate statistics for 2014
        latest_data_stats = tfdv.generate_statistics_from_dataframe(latest_data)
        # visualize statistics
        # tfdv.visualize_statistics(early_data_stats)

        logging.info(f"visualize_statistics for early_data_stats \n {tfdv.visualize_statistics(early_data_stats)}")
        logging.info('-'*80)

        return early_data_stats, latest_data_stats
    
    
    def visualizing_schema(self, early_data_stats):

        # infer schema from the statistics 

        early_data_schema = tfdv.infer_schema(statistics = early_data_stats)

        # save schema to file 
        tfdv.write_schema_text(early_data_schema, self.drift_config.schema_path)
        # save statistics to file
        tfdv.write_stats_text(early_data_stats, self.drift_config.stats_path)

    
        #display the schema
        tfdv.display_schema(schema=early_data_schema)

        logging.info(f"display_schema for early_data_schema \n {tfdv.display_schema(schema=early_data_schema)}")
        logging.info('-'*80)

        return early_data_schema


    def modifying_data(self, early_data_schema):

        from tensorflow_metadata.proto.v0 import schema_pb2
        # add range contraint to the year birth feature 
        tfdv.set_domain(early_data_schema, 'Year_Birth', schema_pb2.IntDomain(name="Year_Birth", min=1900, max=2015))
        # make income feature required
        tfdv.get_feature(early_data_schema, "Income").presence.min_fraction = 1
    
        # dispaly the new schema
        tfdv.display_schema(schema=early_data_schema)

        logging.info(f"display_schema for MODIFIED early_data_schema \n {tfdv.display_schema(schema=early_data_schema)}")
        logging.info('-'*80)
    
    def visualizing_anamolies(self, early_data_schema, early_data_stats, latest_data_stats):

        income = tfdv.get_feature(early_data_schema, 'Income')
        income.drift_comparator.jensen_shannon_divergence.threshold = 0.1

        martial_status = tfdv.get_feature(early_data_schema, 'Marital_Status')
        martial_status.drift_comparator.infinity_norm.threshold = 0.1
        martial_status.distribution_constraints.min_domain_mass = 0.9

        # overlay both statistics on top of each other.
        tfdv.visualize_statistics(lhs_statistics=early_data_stats, rhs_statistics=latest_data_stats,
                          lhs_name='2012', rhs_name='2014')
        
        logging.info(f"visualizing anamolies early_data_schema \n {tfdv.visualize_statistics(lhs_statistics=early_data_stats, rhs_statistics=latest_data_stats,lhs_name='2012', rhs_name='2014')}")
        logging.info('-'*80)
        
    def detect_anomalies(self, latest_data_stats, early_data_schema, early_data_stats):

        drift_anomalies = tfdv.validate_statistics(latest_data_stats, schema=early_data_schema, previous_statistics= early_data_stats)

        # display the differences between the two datasets
        tfdv.display_anomalies(drift_anomalies)

        logging.info(f"display_schema for drift_anomalies \n {tfdv.display_anomalies(drift_anomalies)}")
        logging.info('-'*80)

    def detect_data_drift(self, new_data: pd.DataFrame, schema_path:str, previous_stats_path: str) -> bool:
        """
        compare new data statistics with baseline data.
        """

        previous_stats = tfdv.load_stats_text(previous_stats_path)
        schema = tfdv.load_schema_text(schema_path)
        options = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)
        new_stats = tfdv.generate_statistics_from_dataframe(new_data, stats_options=options)
        drift_anomalies = tfdv.validate_statistics(new_stats, schema=schema, previous_statistics=previous_stats)
        drift_detected = len(drift_anomalies.anomaly_info) > 0

        logging.info('-'*80)
        logging.info(f"drift_detected \n {drift_detected}")
        logging.info('-'*80)
        logging.info(f"drift_anomalies \n {drift_anomalies}")
        logging.info('-'*80)

        return drift_detected, drift_anomalies
    
    def action_on_data_drift(self, latest_data):


        drift_detected, drift_anomalies = self.detect_data_drift(latest_data, self.drift_config.schema_path, self.drift_config.stats_path )

        if drift_detected:
            logging.info("stop pipeline")
        else:
            logging.info("proceed")
    


        



if __name__ == "__main__":

    obj = ModelDrift()
    early_data, latest_data = obj.data_split()
    early_data_stats_obj, latest_data_stats_obj = obj.visualize_stats(early_data, latest_data)
    early_data_schema_obj = obj.visualizing_schema(early_data_stats_obj)
    obj.modifying_data(early_data_schema_obj) 
    obj.visualizing_anamolies(early_data_schema_obj, early_data_stats_obj, latest_data_stats_obj)
    obj.detect_anomalies(latest_data_stats_obj, early_data_schema_obj, early_data_stats_obj)
    obj.action_on_data_drift(latest_data)
        
