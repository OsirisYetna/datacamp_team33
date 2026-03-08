# How to participate

You should submit a `submission.py` file that contains a `get_model()` function. This function must return an **untrained** instance of your model class.

Your model class (e.g., `RFMorganBaseline`) should be compatible with the scikit-learn API. At a minimum, it must implement the following methods:

* `__init__(self, ...)`: Initializes the model hyperparameters.
* `fit(self, X, y)`: Trains the model on the provided data.
* `predict_proba(self, X)`: Returns the predicted probabilities for the test data.
* `predict(self, X)`: Returns the class predictions for the test data.

*(Note: Your model will receive raw SMILES strings as the `X` input, so ensure your class handles the conversion to features, such as Morgan Fingerprints, internally before passing them to the base classifier.)*

This model will then be used with the platform's evaluation loop: your `submission.py` will be imported, and your model will be instantiated, trained, and tested directly on Codalab.

See the "Seed" page for a complete example of a valid `submission.py` file.

See the "Timeline" page for additional information about the phases of this competition.