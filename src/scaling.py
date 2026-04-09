import pickle
from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def save_scaler(scaler, path="models/scaler.pkl"):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path="models/scaler.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def verify_scaling(X_scaled):
    import numpy as np

    means = X_scaled.mean(axis=0)
    stds = X_scaled.std(axis=0)
    print("Mean (approx 0):", means[:5])
    print("Std (approx 1):", stds[:5])

    assert np.allclose(means, 0, atol=1e-1)
    assert np.allclose(stds, 1, atol=1e-1)
    print("Scaling verified")