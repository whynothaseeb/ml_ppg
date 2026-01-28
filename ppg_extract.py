import vitaldb
import numpy as np
import h5py

def extract_to_hdf5(max_cases=200, filename='vitaldb_research.h5', target_fps=30, duration_sec=20):
    samples_needed = target_fps * duration_sec
    tracks = ['SNUADC/PLETH', 'SNUADC/ART']
    
    # Get the list of cases that have these tracks
    case_ids = vitaldb.find_cases(tracks)
    case_ids = case_ids[:max_cases]
    
    all_ppg = []
    all_labels = [] 
    all_meta = []   

    print(f"Found {len(case_ids)} matching cases. Starting extraction...")

    for cid in case_ids:
        try:
            # Use the method that worked in your first script
            vf = vitaldb.VitalFile(cid, tracks)
            data = vf.to_numpy(tracks, interval=1/target_fps)
            
            if data is None or data.shape[1] < 2:
                continue

            # Remove rows with NaNs
            mask = ~np.isnan(data).any(axis=1)
            data = data[mask]
            
            if len(data) < samples_needed:
                continue

            for i in range(0, len(data) - samples_needed, samples_needed):
                segment = data[i : i + samples_needed]
                ppg = segment[:, 0]
                art = segment[:, 1]
                
                # Quality Control
                if np.max(ppg) - np.min(ppg) < 0.05: 
                    continue
                
                # Ground Truth
                sys_val = np.max(art)
                dia_val = np.min(art)
                
                # Physiological Filtering
                if not (40 < dia_val < 120 and 70 < sys_val < 190): 
                    continue

                # Z-score Normalization
                # (ppg - mean) / std makes the data zero-centered for the AI model
                ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
                
                all_ppg.append(ppg_norm)
                all_labels.append([sys_val, dia_val])
                all_meta.append(cid)

            print(f"Case {cid} processed. Total segments: {len(all_ppg)}")
                
        except Exception as e:
            print(f"Error in Case {cid}: {e}")

    # Save to HDF5
    if len(all_ppg) > 0:
        all_ppg_arr = np.array(all_ppg).astype('float32')
        all_labels_arr = np.array(all_labels).astype('float32')
        all_meta_arr = np.array(all_meta).astype('int32')

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("ppg", data=all_ppg_arr, compression="gzip", chunks=True)
            hf.create_dataset("label", data=all_labels_arr, compression="gzip", chunks=True)
            hf.create_dataset("case_id", data=all_meta_arr, compression="gzip", chunks=True)
            
            hf.attrs['fps'] = target_fps
            hf.attrs['duration'] = duration_sec

        print(f"\nSuccess! Saved {len(all_ppg_arr)} segments to {filename}")
    else:
        print("No valid segments found.")

if __name__ == "__main__":
    extract_to_hdf5(max_cases=100)