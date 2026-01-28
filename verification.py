import h5py
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Import the feature extractor from your previous script
from ml import extract_features 

def verify_model(hdf5_path, model_sys_path, model_dia_path, scaler_path):
    # 1. Load the Test Data
    # We will use the end of the HDF5 file (data the model hasn't seen)
    with h5py.File(hdf5_path, 'r') as hf:
        ppg_test = hf['ppg'][-500:] # Last 500 segments
        labels_test = hf['label'][-500:]
    
    y_true_sys = labels_test[:, 0]
    y_true_dia = labels_test[:, 1]

    # 2. Load Models and Scaler
    gp_sys = joblib.load(model_sys_path)
    gp_dia = joblib.load(model_dia_path)
    scaler = joblib.load(scaler_path)

    # 3. Extract and Scale Features
    print("Extracting features from test set...")
    X_test = np.array([extract_features(sig) for sig in ppg_test])
    
    # Filter out any bad segments
    valid_mask = ~np.all(X_test == 0, axis=1)
    X_test = X_test[valid_mask]
    y_true_sys = y_true_sys[valid_mask]
    y_true_dia = y_true_dia[valid_mask]

    X_test_scaled = scaler.transform(X_test)

    # 4. Predict
    print("Generating predictions...")
    y_pred_sys, sigma_sys = gp_sys.predict(X_test_scaled, return_std=True)
    y_pred_dia, sigma_dia = gp_dia.predict(X_test_scaled, return_std=True)

    # 5. Calculate Metrics
    mae_sys = mean_absolute_error(y_true_sys, y_pred_sys)
    mae_dia = mean_absolute_error(y_true_dia, y_pred_dia)
    
    print("\n" + "="*30)
    print("VERIFICATION RESULTS")
    print("="*30)
    print(f"Systolic MAE:  {mae_sys:.2f} mmHg")
    print(f"Diastolic MAE: {mae_dia:.2f} mmHg")
    print(f"Avg Uncertainty (Sigma): {np.mean(sigma_sys):.2f}")
    
    # Check against AAMI Standard (MAE <= 5mmHg)
    if mae_sys <= 5:
        print("Status: Passes AAMI Clinical Standard for Systolic")
    else:
        print("Status: Fails AAMI Standard (Requires more data/better features)")
    print("="*30)

    # 6. Visualization: Correlation Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_true_sys, y_pred_sys, alpha=0.5, color='blue')
    plt.plot([y_true_sys.min(), y_true_sys.max()], [y_true_sys.min(), y_true_sys.max()], 'r--')
    plt.title(f"Systolic Correlation (MAE: {mae_sys:.2f})")
    plt.xlabel("Actual BP (mmHg)")
    plt.ylabel("Predicted BP (mmHg)")

    plt.subplot(1, 2, 2)
    plt.scatter(y_true_dia, y_pred_dia, alpha=0.5, color='green')
    plt.plot([y_true_dia.min(), y_true_dia.max()], [y_true_dia.min(), y_true_dia.max()], 'r--')
    plt.title(f"Diastolic Correlation (MAE: {mae_dia:.2f})")
    plt.xlabel("Actual BP (mmHg)")
    plt.ylabel("Predicted BP (mmHg)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_model(
        'vitaldb_research.h5', 
        'gp_sys_model.pkt', 
        'gp_dia_model.pkt', 
        'scaler.pkt'
    )