from unittest.mock import patch

from dist_xgboost.train import main


@patch("dist_xgboost.train.log_run_to_mlflow")
@patch("dist_xgboost.train.save_preprocessor")
def test_main_execution(mock_save_preprocessor, mock_log_run):
    # Run the main function with mocked artifact saving
    main()

    # Verify mocks were called
    mock_save_preprocessor.assert_called_once()
    mock_log_run.assert_called_once()
