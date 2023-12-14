import pytest
from Model import IrisModel, IrisSpecies


def test_model_initialization():
    new_model = IrisModel()
    assert 'iris_model.pkl' in new_model.model_fname_
