import pytest
import requests
from src.exception import CustomException
import sys


def test_app_running():
    url = "http://127.0.0.1:5000"

    try:
        response = requests.get(url, timeout=5)
        assert response.status_code == 200

    except Exception as e:
        raise CustomException(e,sys)
