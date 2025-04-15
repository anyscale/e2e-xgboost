from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from dist_xgboost.infer import main


# Option 1: Mock the load_model_and_preprocessor function
# this correctly patches the load_model_and_preprocessor function in the main function,
# but not in the Validator.__init__ method because Ray gets the actual function and not the patched one.
def mock_load_model_and_preprocessor():
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform_batch.side_effect = lambda x: x
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda x: np.random.random(size=(len(x),))
    return mock_preprocessor, mock_model


# # Option 2: Mock the Validator.__init__ method
# # This does not work. Ray seems to pickle the original class without the patched init
# def mock_validator_init(self, *args, **kwargs):
#     print("#"*50)
#     print("Using mock_validator_init")
#     print("#"*50)
#     self.model = MagicMock()
#     self.model.predict.side_effect = lambda x: np.random.random(size=(len(x),))


# Option 3: Mock the entire Validator class
# This does not work because when the Validator class is serialized,
# pickle cannot find the `test_infer` module:
# ModuleNotFoundError: No module named 'test_infer'
class MockValidator:
    def __init__(self):
        # Set up the mock model without calling load_model_and_preprocessor
        self.model = MagicMock()
        self.model.predict.side_effect = lambda x: np.random.random(size=(len(x),))

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Similar to the original __call__ but using our mock model
        target = batch.pop("target")
        predictions = self.model.predict(batch)
        return pd.DataFrame({"prediction": predictions, "target": target})


@patch("dist_xgboost.infer.load_model_and_preprocessor", mock_load_model_and_preprocessor)
@patch("dist_xgboost.infer.Validator", MockValidator)
# @patch("dist_xgboost.infer.Validator.__init__", mock_validator_init)
def test_infer_main():
    main()
