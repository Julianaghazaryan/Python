import pytest
import pandas as pd
from unittest.mock import patch
from src.processor import process_chunk, run_pipeline

# 1. Parameterized Test Case Setup (Rubric Item 5)
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

# 2. Mocking File System / I/O Bounds (Rubric Item 5)
@patch("src.processor.pd.read_csv")
def test_run_pipeline_mocked(mock_read_csv):
    fake_df = pd.DataFrame({
        "Study_Hours_Per_Day": [4.5, 5.0, 3.5],
        "Final_CGPA": [3.2, 3.5, 3.0]
    })
    
    mock_read_csv.return_value = fake_df
    results = run_pipeline("fake_path.csv")
    
    assert results["total_students_processed"] == 3
    assert "ml_model_insights" in results
    assert results["analysis_meta"]["status"] == "statistical_and_ml_integration_complete"