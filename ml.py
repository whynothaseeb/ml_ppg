import h5py
import numpy as np
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from scipy.integrate import simpson # More accurate than auc for signal area

# 1. ADVANCED FEATURE EXTRACTOR
def extract_features(ppg_segment):
    """
    Extracts morphological features: Area under curve, Pulse Width, and HR.
    """
    # 0-1 Normalization for consistent feature extraction
    ppg = (ppg_segment - np.min(ppg_segment)) / (np.max(ppg_segment) - np.min(ppg_segment) + 1e-8)
    
    # distance=15 assumes 30fps (min 0.5s between beats)
    peaks, _ = find_peaks(ppg, distance=15, height=0.5)
    
    if len(peaks) < 2:
        return np.zeros(8) 

    # --- Time-based Features ---
    intervals = np.diff(peaks)
    hr_mean = np.mean(intervals)
    hr_var = np.std(intervals)
    
    # --- Morphological Features (Shape Analysis) ---
    # Analyze the first complete heart-beat pulse in the segment
    start, end = peaks[0], peaks[1]
    single_pulse = ppg[start:end]
    
    # Feature 3: Pulse Area (Integral) 
    # Represents the volume of the pulse; smaller areas often correlate with higher BP
    pulse_area = simpson(y=single_pulse)
    
    # Feature 4 & 5: Pulse Width at 50% and 75% height
    def get_width(pulse, height_perc):
        threshold = height_perc * np.max(pulse)
        width_points = np.where(pulse > threshold)[0]
        return len(width_points) if len(width_points) > 0 else 0

    width_50 = get_width(single_pulse, 0.5)
    width_75 = get_width(single_pulse, 0.75)
    
    # Feature 6: Peak-to-Area Ratio
    peak_area_ratio = np.max(single_pulse) / (pulse_area + 1e-8)
    
    # Feature 7: Signal Skewness
    skew = np.mean(((single_pulse - np.mean(single_pulse)) / (np.std(single_pulse) + 1e-8))**3)
    
    # Feature 8: Pulse Rate Count
    pulse_count = len(peaks)

    return np.array([hr_mean, hr_var, pulse_area, width_50, width_75, peak_area_ratio, skew, pulse_count])



# 2. UPDATED TRAINING FUNCTION
def train_gpr_advanced(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        ppg_data = hf['ppg'][:]
        labels = hf['label'][:]
    
    print("Extracting advanced morphological features...")
    X = np.array([extract_features(sig) for sig in ppg_data])
    y_sys = labels[:, 0]
    y_dia = labels[:, 1]

    # Clean data (remove failed extractions)
    valid_mask = ~np.all(X == 0, axis=1)
    X, y_sys, y_dia = X[valid_mask], y_sys[valid_mask], y_dia[valid_mask]

    # Standardize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkt')

    # Optimized Kernel for medical regression
    # WhiteKernel accounts for the noise typical in PPG sensors
    kernel = C(100.0, (1e-1, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1)

    gp_sys = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    gp_dia = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)

    train_limit = min(4000, len(X_scaled))
    
    print(f"Fitting Systolic GPR on {train_limit} samples...")
    gp_sys.fit(X_scaled[:train_limit], y_sys[:train_limit]) 
    
    print(f"Fitting Diastolic GPR on {train_limit} samples...")
    gp_dia.fit(X_scaled[:train_limit], y_dia[:train_limit])

    joblib.dump(gp_sys, 'gp_sys_model.pkt')
    joblib.dump(gp_dia, 'gp_dia_model.pkt')
    print("Advanced GPR Models and Scaler saved successfully.")

if __name__ == "__main__":
    train_gpr_advanced('vitaldb_research.h5')