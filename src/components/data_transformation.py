import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

class DataCleaning(object):
    def __init__(self):
        pass

    def initiate_data_cleaning(self, train_path, test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        #train_df = train_path
        #test_df = test_path

        train_df = train_df.rename(columns={'price_rupiah': 'price'})
        train_df['price'] = train_df['price'].str.replace('Rp', '').str.replace(',', '').astype(int)
        train_df['diameter'] = train_df['diameter'].str.extract('(\d+\.?\d*)').astype(float)

        test_df = test_df.rename(columns={'price_rupiah': 'price'})
        test_df['price'] = test_df['price'].str.replace('Rp', '').str.replace(',', '').astype(int)
        test_df['diameter'] = test_df['diameter'].str.extract('(\d+\.?\d*)').astype(float)
        
        return train_df, test_df

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            
            numerical_columns = ["diameter"]
            categorical_columns = [
                'company',
                'topping',
                'variant',
                'size',
                'extra_sauce',
                'extra_cheese',
                'extra_mushrooms'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )
   
            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                    
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df= train_path
            test_df= test_path

            logging.info(f"Read train and test data completed")

            logging.info(f"Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df).toarray()

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)