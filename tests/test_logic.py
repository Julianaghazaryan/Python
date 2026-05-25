import pytest
import pandas as pd
from unittest.mock import patch
from src.processor import process_chunk, run_pipeline

@pytest.mark.parametrize("input_hours, input_cgpa", [
    (4.5, 3.2),
    (6.0, 3.8),
    (2.1, 2.4)
])
def test_process_chunk_types(input_hours, input_cgpa):
    test_df = pd.DataFrame({
        "Study_Hours_Per_Day": [input_hours],
        "Final_CGPA": [input_cgpa]
    })
    
    result = process_chunk(test_df)
    
    assert result["count"] == 1
    assert isinstance(result["total_cgpa"], float)
    assert isinstance(result["total_hours"], float)

# Ուղղված թեստ, որը երբեք չի կախվի
@patch("src.processor.Pool")
@patch("src.processor.pd.read_csv")
def test_run_pipeline_mocked(mock_read_csv, mock_pool):
    fake_summary = [{
        "count": 3,
        "total_cgpa": 9.7,
        "total_hours": 13.0,
        "sum_xx": 57.5,
        "sum_xy": 42.5
    }]
    
    # Ձևացնում ենք, թե Pool.map-ը հետ է տվել մեր կազմակերպած սումմաները
    mock_pool.return_value.__enter__.return_value.map.return_value = fake_summary
    mock_read_csv.return_value = [pd.DataFrame()] # Կեղծ chunk iterator
    
    results = run_pipeline("fake_path.csv")
    
    assert results["total_students_processed"] == 3
    assert "ml_model_insights" in results
    assert results["analysis_meta"]["status"] == "statistical_and_ml_integration_complete"
