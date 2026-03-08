# Seed:

```
class RFMorganBaseline(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, radius=2, nBits=1024):
        """
        Baseline model converting SMILES to Morgan Fingerprints
        and training a Random Forest.
        """
        self.n_estimators = n_estimators
        self.radius = radius
        self.nBits = nBits
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=42,
            n_jobs=-1
        )
        # Initialize the Morgan fingerprint generator
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.nBits)
        
    def _smiles_to_fps(self, X):
        fps = []
        for smiles in X:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Calculate Morgan Fingerprint (equivalent to ECFP4 if radius=2)
                fp = self.mfpgen.GetFingerprintAsNumPy(mol)
                fps.append(fp)
            else:
                # Handle invalid smiles
                fps.append(np.zeros(self.nBits))
        return np.array(fps)

    def fit(self, X, y):
        """
        Training of the model
        """
        print("Converting SMILES to Morgan Fingerprints...")
        X_fp = self._smiles_to_fps(X)
        
        print(f"Training Random Forest with {self.n_estimators} estimators...")
        self.model.fit(X_fp, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        """
        Probability prediction
        """
        X_fp = self._smiles_to_fps(X)
        return self.model.predict_proba(X_fp)

    def predict(self, X):
        """
        Class prediction
        """
        X_fp = self._smiles_to_fps(X)
        return self.model.predict(X_fp)

def get_model():
    """
    Returns the baseline model.
    """
    return RFMorganBaseline(n_estimators=100, radius=2, nBits=1024)
```