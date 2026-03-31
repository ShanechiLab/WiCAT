import os
import re
import json
from pathlib import Path
import pynwb
                                  
import torch
import numpy         
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import cv2
from scipy.signal import savgol_filter, butter, filtfilt                                                
from wicat.data.base_dataset import BaseImagingDataset
from wicat.data import musall_utils                                             


class KondoWidefieldDataset(BaseImagingDataset):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    @property
    def experiment_type(self):
        return 'StimLeverPull'                                                                                             

    
    def _get_and_parse_session_info(self):
        """
        Scans the filesystem once to find all NWB files, parses their subject
        and session names, and caches the result for other methods to use.
        Filters specifically for days 2, 9, 14, and 15.
        """
                                                           
        if hasattr(self, "_session_info_list"):
            return self._session_info_list

        self.logger.info("Scanning and parsing session info from filesystem...")
        self._session_info_list = []

        cache_path = os.path.join(self.config.save_dir, "raw", "session_info_cache.json")
        
        if os.path.exists(cache_path):
            self.logger.info(f"📂 Loading session info from cache file: {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    self._session_info_list = json.load(f)
                self.logger.info(f"✅ Loaded {len(self._session_info_list)} sessions from cache.")
                return self._session_info_list
            except Exception as e:
                self.logger.error(f"Failed to load cache file: {e}. Falling back to filesystem scan.")

        root_dir = Path(self.config.raw_data_dir)
        
                                                               
        session_pattern = re.compile(r"ses-((?:\d{4}-\d{2}-\d{2})-(?:task|resting-state|sensory-stim)-day\d+)")

                                 
        ALLOWED_DAYS = list(numpy.arange(1,16))                 

        for subject_dir in root_dir.glob('sub-*'):
            if not subject_dir.is_dir():
                continue
            
            original_subject = subject_dir.name
                                                               
            parsed_subject = original_subject.replace('#', '')

            for nwb_path in subject_dir.glob('*.nwb'):
                original_session_stem = nwb_path.stem
                match = session_pattern.search(original_session_stem)
                
                if match:
                                                                 
                    parsed_session = match.group(1)
                    
                                                
                                                                                  
                    day_search = re.search(r'day(\d+)', parsed_session)
                    
                    if day_search:
                        day_num = int(day_search.group(1))
                        
                        if day_num not in ALLOWED_DAYS:
                                                                                                                   
                            continue
                    else:
                                                                           
                        continue
                                              

                    self._session_info_list.append({
                        "parsed_subject": parsed_subject,
                        "parsed_session": parsed_session,
                        "original_subject": original_subject,
                        "original_session_stem": original_session_stem,
                    })
        
        self.logger.info(f"Found and parsed {len(self._session_info_list)} total sessions (Filtered for days {ALLOWED_DAYS}).")

        return self._session_info_list

    @property
    def download_command(self):
        """Data is local, so no download is needed."""
        self.logger.info("Kondo NWB data is local. Skipping download.")
        return []

                                                                     

    @property
    def available_sessions(self):
        """
        Returns session info in the dictionary format required by the base class,
        using the parsed names for subjects and sessions.
        """
        sessions_dict = {}
        for info in self._get_and_parse_session_info():
            subject = info["parsed_subject"]
            if subject not in sessions_dict:
                sessions_dict[subject] = []
            sessions_dict[subject].append(info["parsed_session"])
        
                       
                                                              
                                                                
                                                               
           

                                                                                    
        
        return sessions_dict
        

    def get_raw_data_file_path(self, subject, session):
        """
        Constructs the full path to a raw .nwb file using the original, un-parsed names.
        This is a change from the base class implementation.
        """
        return os.path.join(self.config.raw_data_dir, subject, f"{session}.nwb")

    def process_raw_data(self):
        """
        Orchestrates Stage 1a processing using the parsed and original names
        from our session info list.
        """
                                                                                                                                                                                        
        session_info_list = self._get_and_parse_session_info()
        for info in tqdm(session_info_list[::-1], desc="Processing Raw Sessions (Stage 1a)"):         
                                                              
            self.process_single_session_raw_data(
                subject=info["original_subject"], 
                session=info["original_session_stem"]
            )

    def _extract_behavioral_data(self, nwbfile, view: str):
        """
        Safely extracts behavioral data for a given view ('body', 'face', 'eye').
        Checks for the existence of the data interface and builds a DataFrame
        from individual series to avoid errors with empty/malformed data.
        Returns a pandas DataFrame or None if the data is not found.
        """
        parent_name = f"{view}_video_keypoints"
        data_dict = {}
        
                                                                                                                                    
        if 'downsampled' in nwbfile.processing and parent_name in nwbfile.processing['downsampled'].data_interfaces:
            interface = nwbfile.processing['downsampled'].get_data_interface(parent_name)
            
            if not interface.pose_estimation_series:
                return None

            timestamps = None
            for keypoint in interface.pose_estimation_series.keys():
                series = interface.get_pose_estimation_series(keypoint)
                if series.data is None or series.data.shape[0] == 0:
                    continue 
                
                if timestamps is None:
                                                                                                  
                    timestamps = series.timestamps[:]
                    
                data_dict[f"{keypoint}_x"] = series.data[:, 0]
                data_dict[f"{keypoint}_y"] = series.data[:, 1]
            
            if not data_dict or timestamps is None:
                return None
            
            return pd.DataFrame(data_dict, index=pd.Index(timestamps, name='time'))
        
        return None
        
    def _extract_sensory_data(self, nwbfile):
        """
        Safely extracts all downsampled sensory/environmental data.
        Returns a pandas DataFrame or None if the data is not found.
        """
        sensory_keys = [
            'CO2_level', 'air_pressure', 'humidity', 'lever', 'lick_rate', 'motion', 
            'reward', 'room_temp', 'state_lever', 'state_task', 'tone'
        ]
        data_dict = {}
        timestamps = None

        if 'downsampled' in nwbfile.processing:
            interfaces = nwbfile.processing['downsampled'].data_interfaces
            
            for key in sensory_keys:
                if key in interfaces and isinstance(interfaces[key], pynwb.base.TimeSeries):
                    ts = interfaces[key]
                    if ts.data is None or ts.data.shape[0] == 0:
                        continue
                    
                    if timestamps is None:
                        timestamps = ts.timestamps[:]
                    
                                                                                                
                    if len(ts.timestamps) != len(timestamps):
                        self.logger.warning(f"Timestamp mismatch for {key}. Re-interpolating.")
                                                                   
                        data_dict[key] = numpy.interp(timestamps, ts.timestamps, ts.data)
                    else:
                        data_dict[key] = ts.data[:]
                else:
                    self.logger.warning(f"Sensory key '{key}' not found or is not a TimeSeries.")
            
            if not data_dict or timestamps is None:
                return None, None
            
            df = pd.DataFrame(data_dict, index=pd.Index(timestamps, name='time'))
            return df, df.columns.tolist()
            
        return None, None

    def process_single_session_raw_data(self, subject, session, save_data=True):
        """
        Stage 1a: Loads raw NWB data using pynwb, preprocesses, and saves an intermediate file.
        """
        processed_raw_path = self.get_processed_raw_data_file_path(subject, session)
        if os.path.exists(processed_raw_path) and not self.config.force_reprocess_stage1:
                                                                                        
            return

        raw_nwb_path = self.get_raw_data_file_path(subject, session)
        self.logger.debug(f"Loading NWB: {raw_nwb_path}")

        allen_mask = None
        if self.config.get('align_to_allen_atlas', False):
            mask_path = os.path.join(self.config.raw_data_dir, "allen_mask_512.png")
            if not os.path.exists(mask_path):
                self.logger.error(f"align_to_allen_atlas is True, but mask not found at {mask_path}!")
                return
            allen_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                          
            allen_mask = (allen_mask > 0).astype(numpy.float32) 
            self.logger.debug(f"Loaded Allen atlas mask with shape {allen_mask.shape}")

        try:
            with pynwb.NWBHDF5IO(raw_nwb_path, mode='r', load_namespaces=True) as io:
                nwbfile = io.read()

                behavior_dfs = []
                for view in ['body', 'face', 'eye']:
                    df = self._extract_behavioral_data(nwbfile, view)
                    if df is not None and not df.empty:
                        behavior_dfs.append(df)

                combined_behavior_df = pd.concat(behavior_dfs, axis=1)
                continuous_behavior = combined_behavior_df.values.astype(numpy.float32)
                continuous_behavior_labels = combined_behavior_df.columns.tolist()

                sensory_df, sensory_labels = self._extract_sensory_data(nwbfile)
                if sensory_df is None:
                    self.logger.warning(f"No sensory data found for {raw_nwb_path}. Skipping session.")
                    return
                continuous_sensory = sensory_df.values.astype(numpy.float32)

                if self.config.z_score_kinem:
                    mean = numpy.nanmean(continuous_behavior, axis=0)
                    std = numpy.nanstd(continuous_behavior, axis=0)
                    std[std < 1e-8] = 1.0
                    continuous_behavior = (continuous_behavior - mean) / std
                    continuous_behavior = numpy.nan_to_num(continuous_behavior)

                                                              
                    mean_sen = numpy.nanmean(continuous_sensory, axis=0)
                    std_sen = numpy.nanstd(continuous_sensory, axis=0)
                    std_sen[std_sen < 1e-8] = 1.0
                    continuous_sensory = (continuous_sensory - mean_sen) / std_sen
                    continuous_sensory = numpy.nan_to_num(continuous_sensory)

                                               
                trials_df = nwbfile.trials.to_dataframe() if nwbfile.trials is not None else pd.DataFrame()


                                              
                self.logger.debug("Extracting imaging data...")
                blue_data = nwbfile.acquisition['widefield_blue'].data[:]
                uv_data = nwbfile.acquisition['widefield_UV'].data[:] if self.config.hemodynamic_correction else None
                
                atlas_transform = nwbfile.analysis['atlas_to_data_transform']['affine_matrix'].data[:]
                                                                
                timestamps = nwbfile.acquisition['widefield_blue'].timestamps[:]

                                                             
                preprocessed_imaging = musall_utils.preprocess_raw_session_imaging(
                    blue_data=blue_data,
                    uv_data=uv_data,
                    config=self.config,allen_mask=allen_mask,
                    atlas_transform=atlas_transform,
                )
                                                
                save_dict = {
                    'preprocessed_imaging': torch.from_numpy(preprocessed_imaging),
                    'continuous_behavior': torch.from_numpy(continuous_behavior),
                    'continuous_behavior_labels': continuous_behavior_labels,
                    'continuous_sensory': torch.from_numpy(continuous_sensory),
                    'sensory_labels': sensory_labels, 
                    'binary_behavior_labels': ['trial_outcome', 'pull_onset'],
                    'trials_df': trials_df.to_dict("list"),
                    'timestamps': timestamps,                                   
                    'fs': self.config.fs,
                }
            if save_data:
                os.makedirs(os.path.dirname(processed_raw_path), exist_ok=True)
                torch.save(save_dict, processed_raw_path)
                self.logger.debug(f"Saved Stage 1a data to: {processed_raw_path}")
                              
        except Exception as e:
            self.logger.error(f"Failed to process {raw_nwb_path}: {e}")
            return


    def _apply_preprocessing(self, data):
        """
        Applies Interpolation -> Bandpass Filter -> Z-Score.
        """
        if not self.config.apply_kinem_preprocessing:
            return data
                             
        if numpy.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.interpolate(method='linear', limit_direction='both', axis=0)
            df = df.fillna(method='bfill').fillna(method='ffill')
            df = df.fillna(0.0)
            data = df.values

                                           
        nyq = 0.5 * self.config.fs
        low = self.config.bp_low / nyq
        high = self.config.bp_high / nyq
        b, a = butter(self.config.filter_order, [low, high], btype='band')
        data_filt = filtfilt(b, a, data, axis=0)

                           
        mean_val = numpy.nanmean(data_filt, axis=0)
        std_val = numpy.nanstd(data_filt, axis=0)
        std_val[std_val < 1e-8] = 1.0
        data_z = (data_filt - mean_val) / std_val
        
        return numpy.nan_to_num(data_z)

    def _is_bad_chunk(self, chunk):
        """
        Checks if a specific chunk is bad based on Max Amplitude or Low Variance.
        Returns True if bad (should be dropped).
        """
                                                 
        if numpy.max(numpy.abs(chunk)) > self.config.abs_thresh_max:
            return True
        
                                                
        chunk_vars = numpy.var(chunk, axis=0)
        if numpy.any(chunk_vars < self.config.var_thresh_min):
            return True
            
        return False

    def process_single_session_segments(self, subject, session):
        """
        Stage 1b: Segments data using parsed names for metadata and robust frame indexing.
        Note: The 'subject' and 'session' arguments are the PARSED names.
        """
                                                                      
        session_info = next(
            info for info in self._get_and_parse_session_info() 
            if info["parsed_subject"] == subject and info["parsed_session"] == session
        )
        original_subject = session_info["original_subject"]
        original_session_stem = session_info["original_session_stem"]
        
        processed_raw_path = self.get_processed_raw_data_file_path(original_subject, original_session_stem)
        if not os.path.exists(processed_raw_path):
            self.logger.warning(f"Stage 1a file not found for {original_subject}/{original_session_stem}, skipping segmentation.")
            return
        

        try:
            data = torch.load(processed_raw_path, weights_only=False)
                              
        except (ModuleNotFoundError, ImportError):
                                                                                             
            import sys
                                
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            data = torch.load(processed_raw_path, weights_only=False)


        imaging_data = data['preprocessed_imaging'].numpy()
                                            
        kinem_inds_to_keep = list(range(24,33)) + [2,34,10,12,13,14,15,17,18,43,44]
                                                                       
        if data['continuous_behavior'].shape[1] <= 101:
            self.logger.warning(f"Kinematic data has insufficient dimensions ({data['continuous_behavior'].shape[1]}) for {original_subject}/{original_session_stem}, skipping segmentation.")
            return


        kinem_data = data['continuous_behavior'].numpy()[:, kinem_inds_to_keep]
        kinem_labels = [data['continuous_behavior_labels'][i] for i in kinem_inds_to_keep]

        self.logger.info(f"Preprocessing behavior (BP {self.config.bp_low}-{self.config.bp_high}Hz) for {session}...")
        kinem_data = self._apply_preprocessing(kinem_data)

        trials_df = pd.DataFrame(data['trials_df'])
        fs = data['fs']
        metadata_rows = []
        chunk_len = int(self.config.segment_length * fs)
        valid_chunks = []

        if 'task' in session:
            if trials_df.shape[0] == 0:
                self.logger.warning(f"No trial data found for session {subject}_{session}. Skipping segmentation.")
                return
            if self.config.trial_align:
                pre_frames = int(self.config.pre_event_time * fs)
                post_frames = chunk_len - pre_frames

                for seg_id, trial in trials_df.iterrows():
                    start_frame = int(trial['start_time'] * fs)
                    
                    if start_frame - pre_frames < 0 or start_frame + post_frames > imaging_data.shape[0]:
                        continue

                    img_chunk = imaging_data[start_frame - pre_frames : start_frame + post_frames]
                    kinem_chunk = kinem_data[start_frame - pre_frames : start_frame + post_frames]

                                           
                    if self._is_bad_chunk(kinem_chunk):
                        continue

                                                            
                    outcome_map = {'success': 1.0, 'failure': -1.0, 'ignore': 0.0}
                    outcome = outcome_map.get(trial.get('trial_outcome'), 0.0)
                    binary_beh = numpy.array([outcome, trial['pull_onset']], dtype=numpy.float32)

                    valid_chunks.append({
                        'img': img_chunk,
                        'kinem': kinem_chunk,
                        'binary_beh': binary_beh,
                        'seg_id': seg_id
                    })

        elif 'resting-state' in session:
            overlap = int(self.config.segment_overlap * fs)
            step = chunk_len - overlap
            seg_id = 0
            
            for start_frame in range(0, imaging_data.shape[0] - chunk_len + 1, step):
                end_frame = start_frame + chunk_len
                img_chunk = imaging_data[start_frame:end_frame]
                kinem_chunk = kinem_data[start_frame:end_frame]
                
                if self._is_bad_chunk(kinem_chunk):
                    continue

                valid_chunks.append({
                    'img': img_chunk,
                    'kinem': kinem_chunk,
                    'binary_beh': numpy.array([], dtype=numpy.float32),
                    'seg_id': seg_id
                })
                seg_id += 1
        else:
            self.logger.warning(f"Unknown session type: {session}. Skipping.")
            return

        if not valid_chunks:
            self.logger.warning(f"No valid segments after filtering for session {subject}_{session}.")
            return

                                                                      
        self.logger.info(f"Applying final z-scoring to {len(valid_chunks)} valid kinematic chunks...")
        all_kinem = numpy.concatenate([chunk['kinem'] for chunk in valid_chunks], axis=0)
        final_mean = numpy.mean(all_kinem, axis=0)
        final_std = numpy.std(all_kinem, axis=0)
        final_std[final_std < 1e-8] = 1.0                          

                                                                
        for chunk in valid_chunks:
                                   
            kinem_chunk_final = (chunk['kinem'] - final_mean) / final_std
            kinem_chunk_final = numpy.nan_to_num(kinem_chunk_final)

                             
            processed_img = musall_utils.preprocess_imaging_chunk(
                chunk['img'], do_zscore=self.config.z_score_segments
            )
            
            data_dict = {
                'imaging': torch.from_numpy(processed_img).unsqueeze(1),
                'kinem': torch.from_numpy(kinem_chunk_final).to(torch.float32),
                'binary_beh': torch.from_numpy(chunk['binary_beh']),
                'subject': subject,
                'session': f"{subject}_{session}",
            }
            
            path, fname = self.save_segment_data(data_dict, subject, session, chunk['seg_id'])
            meta_row = self._create_metadata_row(
                subject, session, path, fname, kinem_labels,
                list(data_dict['imaging'].shape[2:]) + [data_dict['imaging'].shape[0]],
                data_dict['kinem'].shape[1],
                data_dict['binary_beh'].numel(),
                self.experiment_type
            )
            metadata_rows.append(meta_row)
 
        if not metadata_rows:
            self.logger.warning(f"No segments were created for session {subject}_{session}.")
            return

                                                    
                          
                                                    
        session_df = pd.DataFrame(metadata_rows)
        session_df = session_df.sample(frac=1, random_state=42).reset_index(drop=True)

        if 'resting-state' in session:
                                                                    
            val_count = int(len(session_df) * 0.2)
            session_df.loc[:len(session_df)-val_count, 'split'] = 'train'
            session_df.loc[len(session_df)-val_count:, 'split'] = 'val'
        else:
                                     
            val_count = int(len(session_df) * self.config.val_ratio)
            test_count = int(len(session_df) * self.config.test_ratio)
            train_count = len(session_df) - val_count - test_count
            
            session_df.loc[:train_count, 'split'] = 'train'
            session_df.loc[train_count: train_count + val_count, 'split'] = 'val'
            session_df.loc[train_count + val_count:, 'split'] = 'test'

        self.metadata.concat(new_metadata_df=session_df)
        self.logger.info(f"Processed {len(session_df)} segments for {session}")
        

    def _create_metadata_row(self, subject, session, path, fname, kinem_labels, d_imaging, d_kinem, d_binary_beh, experiment_type):
        """Helper function to create a single metadata dictionary."""
        return {
            "subject": subject,
            "session": session,
            "subject_session": f"{subject}_{session}",
            "experiment_name": experiment_type,
            "d_imaging": d_imaging,                                                          
            "d_kinem": d_kinem,                               
            "d_binary_beh": d_binary_beh,                                                           
            "kinem_channel_names": kinem_labels,                       
            "input_channel_names": "",
            "path": path,
            "split": "train",
            "segment_filename": fname,
            "segments_processing_str": self.segments_processing_str,
        }

