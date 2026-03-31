                   
import os
from typing import Optional

import torch
import numpy as np
import pandas as pd
from einops import rearrange
from omegaconf import DictConfig

from wicat.data.base_dataset import BaseImagingDataset
from wicat.data.metadata import Metadata                                                
from wicat.data import musall_utils                                   

import requests
from bs4 import BeautifulSoup                                         

class MusallImagingDataset(BaseImagingDataset):                                   
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)                                


    @property
    def experiment_type(self):
        return "StimLeverLick"                                              

    @property
    def available_sessions(self):
        """
        Return a dictionary of available sessions.
        Keys: subject IDs, Values: list of session IDs.
        Adapt this to your data organization.
        """
                                                                            
        return {
            "mSM30": ["10-Oct-2017", "12-Oct-2017"],
            "mSM34": ["01-Dec-2017", "02-Dec-2017"],
            "mSM36": ["05-Dec-2017", "07-Dec-2017"],
            "mSM43": ["21-Nov-2017", "23-Nov-2017"],
            "mSM44": ["21-Nov-2017", "29-Nov-2017"],
            "mSM46": ["01-Dec-2017", "13-Jan-2018"],
            "mSM49": ["12-Mar-2018", "19-Dec-2017"],
            "mSM53": ["14-Mar-2018", "21-Mar-2018"],
            "mSM55": ["13-Feb-2018", "16-Feb-2018"],
            "mSM56": ["22-Feb-2018", "27-Feb-2018"],
            "mSM57": ["02-Feb-2018", "08-Feb-2018"],
            "mSM65": ["08-Sep-2018", "10-Sep-2018"],
            "mSM66": ["08-Sep-2018", "10-Sep-2018"],
        }

    @property
    def download_command(self):
        """
        Downloads Widefield imaging data from labshare.cshl.edu, filtering mouse and session folders,
        including BehaviorVideo subfolders, skips existing files, uses correct URL construction,
        and includes specific combined segments files from BehaviorVideo subfolders.
        """
        base_url = "https://labshare.cshl.edu//shares/library/repository/38599/Widefield/"
        raw_data_dir = self.config.raw_data_dir
        download_commands = []
        unwanted_folders = ['Name', 'Last modified', 'Size', 'Description',
                            'Parent Directory']                           
        combined_segment_files = ["motionSVD_CombinedSegments.mat",
                                  "SVD_CombinedSegments.mat"]                               

        try:
            response = requests.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

                                       
            mouse_folders = [
                folder.text.strip()
                for folder in soup.find_all('a')
                if folder.text.strip()
                   and folder.text.strip() not in unwanted_folders
                   and not folder.text.strip().endswith('.mat')
                   and not folder.text.strip() == '../'
                   and folder.text.strip().startswith('mSM')
            ]

            for mouse_folder in mouse_folders:
                mouse_url = os.path.join(base_url, mouse_folder)
                mouse_dir = os.path.join(raw_data_dir, mouse_folder)
                os.makedirs(mouse_dir, exist_ok=True)

                session_response = requests.get(mouse_url)
                session_response.raise_for_status()
                session_soup = BeautifulSoup(session_response.content, 'html.parser')

                                             
                session_folders = [
                    folder.text.strip()
                    for folder in session_soup.find_all('a')
                    if folder.text.strip()
                       and folder.text.strip() not in unwanted_folders
                       and not folder.text.strip().endswith('.mat')
                       and not folder.text.strip() == '../'
                       and folder.text.strip().endswith('/')
                ]

                for session_folder in session_folders:
                    session_folder = session_folder.rstrip('/')
                    session_url = os.path.join(mouse_url, session_folder)
                    session_dir = os.path.join(mouse_dir, session_folder)
                    os.makedirs(session_dir, exist_ok=True)

                    file_response = requests.get(session_url)
                    file_response.raise_for_status()
                    file_soup = BeautifulSoup(file_response.content, 'html.parser')
                    mat_files = [folder.text.strip() for folder in file_soup.find_all('a') if
                                                                         
                                 folder.text.strip().endswith('.mat')]

                    for mat_file in mat_files:
                        file_url = session_url.rstrip('/') + '/' + mat_file                                
                        file_path = os.path.join(session_dir, mat_file)

                                                                      
                        if not os.path.exists(file_path):                        
                            download_command = f'wget -q -O "{file_path}" "{file_url}"'
                            download_commands.append(download_command)
                        else:
                            self.logger.info(f"File already exists: {file_path}, skipping download.")

                                                                
                    behavior_video_folder = "BehaviorVideo"                                       
                    behavior_video_url = session_url.rstrip('/') + '/' + behavior_video_folder
                    behavior_video_dir = os.path.join(session_dir, behavior_video_folder)

                    try:                                           
                        behavior_video_response = requests.get(behavior_video_url)
                        behavior_video_response.raise_for_status()                                                 
                        behavior_video_soup = BeautifulSoup(behavior_video_response.content, 'html.parser')
                        behavior_video_mat_files = [folder.text.strip() for folder in behavior_video_soup.find_all('a')
                                                    if                                          
                                                    folder.text.strip().endswith('.mat')]
                        os.makedirs(behavior_video_dir, exist_ok=True)                                                

                        for behavior_video_mat_file in behavior_video_mat_files:
                            behavior_video_file_url = behavior_video_url.rstrip(
                                '/') + '/' + behavior_video_mat_file                               
                            behavior_video_file_path = os.path.join(behavior_video_dir,
                                                                    behavior_video_mat_file)              

                                                                                        
                            if not os.path.exists(behavior_video_file_path):                        
                                behavior_video_download_command = f'wget -q -O "{behavior_video_file_path}" "{behavior_video_file_url}"'
                                download_commands.append(behavior_video_download_command)
                            else:
                                self.logger.info(
                                    f"BehaviorVideo file already exists: {behavior_video_file_path}, skipping download.")

                                                                                                         
                        for combined_file in combined_segment_files:
                            behavior_video_combined_file_url = behavior_video_url.rstrip(
                                '/') + '/' + combined_file                                                        
                            behavior_video_combined_file_path = os.path.join(behavior_video_dir,
                                                                             combined_file)                                                               

                            if not os.path.exists(behavior_video_combined_file_path):
                                behavior_video_combined_download_command = f'wget -q -O "{behavior_video_combined_file_path}" "{behavior_video_combined_file_url}"'
                                download_commands.append(behavior_video_combined_download_command)
                            else:
                                self.logger.info(
                                    f"BehaviorVideo combined segments file already exists: {behavior_video_combined_file_path}, skipping download.")


                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 404:
                            self.logger.info(f"BehaviorVideo folder not found in session {session_folder}, skipping.")
                        else:
                            self.logger.error(f"Error accessing BehaviorVideo folder in session {session_folder}: {e}")
                    except requests.exceptions.RequestException as e:
                        self.logger.error(f"Error processing BehaviorVideo folder in session {session_folder}: {e}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading data: {e}")
            return ["echo 'Download failed. Check error logs.'"]

        return download_commands

    def get_raw_data_file_path(self, subject, session):
        """
        Construct the path to the raw .mat file for a given session.
        Adapt the path construction to your data organization.
        """
        cPath = self.config.raw_data_base_dir                                     
        paradigm = self.config.paradigm                                                 
        dtype = self.config.imaging_dtype                                                    
        animal = subject                            
        rec = session                            

        file_path = os.path.join(cPath, animal, rec)                
        bhv_files = [f for f in os.listdir(file_path) if f.startswith(f"{animal}_{paradigm}_") and f.endswith(".mat")]
        if bhv_files:
            bhv_filename = bhv_files[0]                                      
        else:
            raise FileNotFoundError(f"No behavior file found in {file_path}")
        full_file_path = os.path.join(file_path, bhv_filename)                        
        return cPath                                                                      


    def process_raw_data(self):
        """
        Orchestrates raw data processing for all sessions.
        Calls process_single_session_raw_data for each session.
        """
        for subject in self.available_sessions.keys():
            sessions_count = len(self.available_sessions[subject])
            if sessions_count:
                self.logger.info(f"Raw data processing for subject {subject} starts.")
                for i, session in enumerate(self.available_sessions[subject]):
                    self.logger.info(f"Processing session {session} ({i+1}/{sessions_count})...")
                    self.process_single_session_raw_data(subject=subject, session=session)


    def process_single_session_raw_data(self, subject, session, save_data=True):
        """
        Processes raw data for a single session: Loads .mat, extracts data using
        extract_time_series_data, and saves processed data (optional).
        """
        processed_raw_data_path = self.get_processed_raw_data_file_path(subject=subject, session=session)
        if os.path.exists(processed_raw_data_path):
            return
        file_path = self.get_raw_data_file_path(subject=subject, session=session)
        self.logger.info(f"Loading raw data file: {file_path}")

        extracted_data = musall_utils.extract_time_series_data(                   
            cPath = self.config.raw_data_base_dir,                                    
            Animal = subject,
            Paradigm = self.config.paradigm,
            Rec = session,
            dType = self.config.imaging_dtype,
            use_prehc_pcs=self.config.use_500_pcs,
        )
        combined_data = musall_utils.process_combined_data(extracted_data, self.config.imaging_dtype)


        U = extracted_data['U']
        Vc = combined_data['Vc']

        if self.config.align_to_allen_atlas:
            trans_params = extracted_data['opts']['transParams']                                
            for i in range(U.shape[2]):
                U[:, :, i] = musall_utils.align_allen_transform_image(U[:, :, i], trans_params)
            allen_mask = musall_utils.loadmask(os.path.join(self.config.raw_data_base_dir, 'allenDorsalMapSM.mat'))
            assert U.shape[0] == allen_mask.shape[0], "U and mask dimensions do not match"
            U = U[:, 24:564]                                       
            allen_mask = allen_mask[:540, 24:564]                  
            if self.config.mask_out_nonallen:
                U = U * allen_mask[..., None]
        else:
            allen_mask = None
            U = U[:, 40:580]              

        if self.config.resize_dims:
            U = musall_utils.spatial_downsample(U, self.config.resize_dims)
            allen_mask = musall_utils.spatial_downsample(allen_mask, self.config.resize_dims).astype('bool') if allen_mask is not None else None


                                                 
        if self.config.z_score_imaging:
            U_flat = U.reshape(-1, U.shape[2])
            non_nan_ind = ~np.isnan(U_flat[:,0])
            U_flat = U_flat[non_nan_ind]
            y_rec = U_flat @ Vc.reshape(-1, Vc.shape[2]).T
            y_std = y_rec.std() * non_nan_ind.mean()
            del y_rec
            Vc = Vc / y_std

                                             
        z = combined_data['behavior_cont']
        z = (z - z.mean(axis=(0,1), keepdims=True)) / z.std(axis=(0,1), keepdims=True)

                                         
        z_bin = combined_data['behavior_binary']
        if self.config.z_score_binary_behavior:
            z_bin = (z_bin - z_bin.mean(axis=(0,1), keepdims=True)) / z_bin.std(axis=(0,1), keepdims=True)

                           
                    
                       
                          
                                                      
                                                       
                  
          

        save_dict = dict(
            vc=torch.from_numpy(Vc),
            z_cont=torch.from_numpy(z),
            z_bin=torch.from_numpy(z_bin),
            z_vid1_pc=torch.from_numpy(combined_data['vidR_aligned']),
            z_vid2_pc=torch.from_numpy(combined_data['moveR_aligned']),
            U=torch.from_numpy(U),
            trials=torch.from_numpy(combined_data['all_trials']),
            z_cont_keys=combined_data['continuous_keys'],                    
            z_bin_keys=combined_data['binary_keys'],
        )

                                                                                  
        if save_data:
            self.logger.info(f"Saving processed raw data to: {processed_raw_data_path}")
            torch.save(save_dict, processed_raw_data_path)                               
        return extracted_data                                                           


    def process_single_session_segments(self, subject, session):
        """
        Processes segments for a single session: Loads processed raw data,
        segments it using process_imaging_video_data, process_continuous_bhv_data,
        process_event_timeseries_data, creates metadata, and saves segments.
        """
        processed_raw_data_path = self.get_processed_raw_data_file_path(subject=subject, session=session)
        self.logger.info(f"Loading processed raw data from: {processed_raw_data_path}")
        data = torch.load(processed_raw_data_path)

        U = data["U"].to(torch.float32)
        vc = data["vc"].to(torch.float32)
        z_cont, z_bin = data["z_cont"].to(torch.float32), data["z_bin"]

        indices_to_keep = [i for i in range(z_cont.shape[-1]) if i != 1] 
        z_cont = z_cont[..., indices_to_keep]
        
        if self.config.trial_align:
            segmented_vc = vc                                       
            segmented_z_cont = z_cont                                      
            segmented_z_bin = z_bin                                    

        else:
            trial_lens = 189
            segmented_vc = torch.split(vc, trial_lens)
            segmented_z_cont = torch.split(z_cont, trial_lens)
            segmented_z_bin = torch.split(z_bin, trial_lens)


                                                    
        metadata_df = []                                         
        num_segments = len(segmented_vc)

        for i_segment in range(num_segments):

            inputs = torch.nan_to_num((U @ segmented_vc[i_segment].clone().T).permute(2, 0, 1), nan=0.0).unsqueeze(1)                                                                                      
 
            data_dict = dict(
                imaging=inputs,                                    
                kinem=segmented_z_cont[i_segment].clone(),                                  
                vc=segmented_vc[i_segment].clone(),
                z_bin=segmented_z_bin[i_segment].clone(),
            )


            segment_path, segment_filename = self.save_segment_data(
                data_dict=data_dict,
                subject=subject,
                session=session,
                segment_id=i_segment,
            )

            meta_row = dict(
                subject=subject,
                session=session,
                subject_session=f"{subject}_{session}",
                experiment_name=self.experiment_type,
                d_imaging=str(list(U.shape)),
                d_binary_beh=segmented_z_bin.shape[2],
                d_kinem=int(segmented_z_cont.shape[2]),
                imaging_channel_names=str(U.shape[2]),
                imaging_modality=self.config.imaging_dtype,
                path=segment_path,
                split="train",
                segment_filename=segment_filename,
                segments_processing_str=self.segments_processing_str,
            )
            metadata_df.append(meta_row)


        metadata_df = pd.DataFrame(metadata_df)                                        

                                                                                       
        if not metadata_df.empty:                                                     
            metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)          
            val_segments_count = int(len(metadata_df) * self.config.val_ratio)
            test_segments_count = int(len(metadata_df) * self.config.test_ratio)

            metadata_df.loc[:val_segments_count, 'split'] = 'val'                     
            metadata_df.loc[val_segments_count:val_segments_count + test_segments_count, 'split'] = 'test'                      

        self.metadata.concat(new_metadata_df=metadata_df)                                           




