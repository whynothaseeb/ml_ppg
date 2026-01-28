import vitaldb
import numpy as np
import h5py
import os

def extract_all_to_hdf5(filename='vitaldb_research.h5', target_fps=30, duration_sec=20):
    samples_needed = target_fps * duration_sec
    tracks = ['SNUADC/PLETH', 'SNUADC/ART']
    
    # 1. Get ALL case IDs that contain both PPG and Arterial lines
    all_case_ids = vitaldb.find_cases(tracks)
    print(f"Total matching cases found in VitalDB: {len(all_case_ids)}")
    
    # Initialize HDF5 file with resizable datasets
    with h5py.File(filename, 'w') as hf:
        # We create empty datasets that can grow (maxshape=None)
        hf.create_dataset("ppg", shape=(0, samples_needed), maxshape=(None, samples_needed), 
                          dtype='float32', compression="gzip", chunks=True)
        hf.create_dataset("label", shape=(0, 2), maxshape=(None, 2), 
                          dtype='float32', compression="gzip", chunks=True)
        hf.create_dataset("case_id", shape=(0,), maxshape=(None,), 
                          dtype='int32', compression="gzip", chunks=True)
        
        hf.attrs['fps'] = target_fps
        hf.attrs['duration'] = duration_sec

        total_segments_saved = 0

        for count, cid in enumerate(all_case_ids):
            try:
                # Download and convert to numpy
                vf = vitaldb.VitalFile(cid, tracks)
                data = vf.to_numpy(tracks, interval=1/target_fps)
                
                if data is None or data.shape[1] < 2:
                    continue

                # Clean NaNs
                mask = ~np.isnan(data).any(axis=1)
                data = data[mask]
                
                if len(data) < samples_needed:
                    continue

                case_ppg, case_labels, case_meta = [], [], []

                for i in range(0, len(data) - samples_needed, samples_needed):
                    segment = data[i : i + samples_needed]
                    ppg = segment[:, 0]
                    art = segment[:, 1]
                    
                    # Quality Control: Signal must have enough amplitude
                    if np.max(ppg) - np.min(ppg) < 0.05: 
                        continue
                    
                    # Ground Truth
                    sys_val = np.max(art)
                    dia_val = np.min(art)
                    
                    # Physiological Filtering: Remove errors/outliers
                    if not (30 < dia_val < 130 and 60 < sys_val < 220): 
                        continue

                    # Z-score Normalization (Critical for LSTM)
                    ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
                    
                    case_ppg.append(ppg_norm)
                    case_labels.append([sys_val, dia_val])
                    case_meta.append(cid)

                # If we found valid segments, append them to the HDF5 file
                if len(case_ppg) > 0:
                    num_new = len(case_ppg)
                    
                    # Resize datasets to fit new data
                    hf["ppg"].resize((total_segments_saved + num_new), axis=0)
                    hf["label"].resize((total_segments_saved + num_new), axis=0)
                    hf["case_id"].resize((total_segments_saved + num_new), axis=0)

                    # Write data
                    hf["ppg"][total_segments_saved:] = np.array(case_ppg)
                    hf["label"][total_segments_saved:] = np.array(case_labels)
                    hf["case_id"][total_segments_saved:] = np.array(case_meta)

                    total_segments_saved += num_new
                
                print(f"[{count+1}/{len(all_case_ids)}] Case {cid}: Saved {len(case_ppg)} segments. Total: {total_segments_saved}")
                    
            except Exception as e:
                print(f"Error in Case {cid}: {e}")

    print(f"\nFINISH! Total segments extracted: {total_segments_saved}")

if __name__ == "__main__":
    extract_all_to_hdf5()