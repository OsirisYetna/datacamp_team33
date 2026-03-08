## Summary
This challenge focuses on **binary classification of molecules** represented by SMILES strings. The goal is to predict whether a compound is an **active inhibitor** (label `1`) or **inactive** (label `0`) against **Beta-secretase 1 (BACE1)**, a key therapeutic target for Alzheimer’s disease.

## Why it matters
Accurate prediction of BACE1 inhibitors can accelerate early-stage drug discovery by reducing costly wet‑lab screening and prioritizing the most promising candidates.

## Task
Given a molecule’s **SMILES** string, build a model that outputs the probability of being an active inhibitor. The final prediction is a binary label derived from these probabilities.

### Out‑of‑distribution Split
This is **not a random split**. The data is split by **molecular weight (MW)**:

- **Training set:** smaller molecules with **MW < 592 Da**
- **Test set:** larger molecules with **MW ≥ 592 Da**

Your model must **generalize to heavier molecules** it has never seen, making this a true out‑of‑distribution (OOD) generalization task.

## Evaluation metric
Submissions are evaluated using **Cohen’s Kappa** score. This metric is robust to class imbalance and accounts for agreement occurring by chance, which makes it more appropriate than raw accuracy for this dataset.

## Baseline model
A baseline is provided in the notebook using:

- **Morgan fingerprints** (ECFP-like) from RDKit
- **Random forest** classifier

This baseline converts SMILES to fixed-length fingerprints, then trains a Random Forest to predict labels. The socre is approximately 0.31, TRY TO BEAT IT!

## Submission format
Your submission should output predictions for each test molecule in the required format (as described in the competition instructions). Typically, this is a CSV file with one prediction per test instance.

## Tips for better performance
- Try molecular featurization beyond Morgan fingerprints (e.g., RDKit descriptors, graph neural networks)
- Use probability calibration to improve decision thresholds
- Consider model robustness to the OOD split (e.g., domain adaptation or regularization)

---

If you want, I can also add a shorter one‑page summary or adapt this into the official `README.md`.
