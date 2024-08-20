
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.exception import CustomException



# Fixture para crear un mock de DataIngestionConfig
@pytest.fixture
def mock_ingestion_config(mocker):
    mock_config = MagicMock()
    mock_config.train_data_path = 'artifacts/train.csv'
    mock_config.test_data_path = 'artifacts/test.csv'
    mock_config.raw_data_path = 'artifacts/data.csv'
    mocker.patch('src.components.data_ingestion.DataIngestionConfig', return_value=mock_config)
    return mock_config

def test_initiate_data_ingestion(mocker, mock_ingestion_config):
    # Configurar mocks
    mock_df = pd.DataFrame({'column1': [1, 2], 'column2': [3, 4]})
    mock_read_csv = mocker.patch('src.components.data_ingestion.pd.read_csv', return_value=mock_df)
    mock_makedirs = mocker.patch('src.components.data_ingestion.os.makedirs')
    mock_train_test_split = mocker.patch('src.components.data_ingestion.train_test_split', return_value=(mock_df, mock_df))

    # Instanciar la clase
    data_ingestion = DataIngestion()
    
    # Llamar al método que se está probando
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    # Aserciones
    mock_read_csv.assert_called_once_with(os.path.normpath('notebook/data/pizza_v2.csv'))
    mock_makedirs.assert_called_once_with(os.path.dirname(mock_ingestion_config.train_data_path), exist_ok=True)
    mock_train_test_split.assert_called_once_with(mock_df, test_size=0.2, random_state=42)

    # Verificar rutas de archivo devueltas
    assert train_path == mock_ingestion_config.train_data_path
    assert test_path == mock_ingestion_config.test_data_path

def test_initiate_data_ingestion_exception(mocker):
    # Configurar mock para lanzar excepción al leer el archivo CSV
    mocker.patch('src.components.data_ingestion.pd.read_csv', side_effect=Exception('File not found'))
    
    # Instanciar la clase
    data_ingestion = DataIngestion()
    
    # Verificar que se lanza CustomException
    with pytest.raises(CustomException):
        data_ingestion.initiate_data_ingestion()