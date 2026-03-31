                 
 
import hashlib
import os
import subprocess
import sys
from typing import Optional
import random
import json 
import torchvision.transforms.functional as TF                      

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import DictConfig
from scipy.signal import decimate
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset

from wicat.data.metadata import Metadata
from wicat.utility.utils import init_logger


class BaseImagingCoreDataset(Dataset):
    def __init__(self, config: DictConfig, **kwargs):
        self.config = config
        self.logger = init_logger(name=self.__class__.__name__)

        self.segments_processing_str, self.segments_processing_hash_str = (
            self.get_segments_processing_hash(
                segment_length=self.config.segment_length,
                segment_from_existing_data=self.config.segment_from_existing_data,
                existing_data_segment_length=self.config.existing_data_segment_length,
            )
        )
                           
        os.makedirs(self.raw_data_dir, exist_ok=True)
        if not self._is_downloaded():
            self.logger.info("Raw dataset is not downloaded, download starts.")
            if self.download_data:
                self._mark_as_downloaded()
                self.logger.info("Download complete.")
            else:
                self.logger.error("Error downloading the dataset.")
                sys.exit(1)
        else:
            self.logger.info("Raw dataset is downloaded.")

                                                     
        os.makedirs(self.processed_raw_data_dir, exist_ok=True)
        if not self._is_raw_data_processed() or self.config.force_reprocess_stage1:
            self.logger.info(
                "Processed raw dataset do not exist (i.e., images (and/or ) are not extracted) or reprocessing is enabled, processing starts."
            )
            self.process_raw_data()
            self.logger.info("Raw data processing complete.")
        else:
            self.logger.info(
                "Processed raw data exists (i.e., images (and/or ) are extracted.)"
            )

                                                        
        os.makedirs(self.processed_segments_data_dir, exist_ok=True)

                                  
        self.metadata = self.initialize_or_load_metadata()
        if not self._is_segments_processed() or self.config.force_reprocess_stage2:
            self.logger.info(
                (
                    f"Processed segments with processing string: '{self.segments_processing_str}' and hash '{self.segments_processing_hash_str}' "
                    " do not exist, or reprocessing is enabled, processing starts (paths or hashed string may have changed, check paths inside metadata)."
                )
            )

                                                            
            self.metadata.clear()

                                      
            self.process_segments()
            self.logger.info("Processing of segments is complete.")
        else:
            self.logger.info(
                f"Processed segments exists (with processing string '{self.segments_processing_str} and hash {self.segments_processing_hash_str}')."
            )

    @property
    def column_names(self):
        return [
            "subject",
            "session",
            "subject_session",
            "experiment_name",
            "d_image",
            "d_kinem",
            "image_channel_names",
            "path",
            "split",
            "segment_filename",
            "segments_processing_str",
        ]

    def initialize_or_load_metadata(self) -> Metadata:
        if os.path.exists(self.metadata_path):
            metadata = Metadata(load_path=self.metadata_path)
        else:
            metadata_df = pd.DataFrame(columns=self.column_names)
            metadata = Metadata(metadata_df=metadata_df)
        return metadata

    @property
    def available_sessions(self):
        """
        return a list of available sessions in format of {subject}-{session}
        these will be used for
        """
        raise NotImplementedError

    @property
    def experiment_type(self):
        raise NotImplementedError

    @property
    def download_command(self):
        raise NotImplementedError

    @property
    def raw_data_dir(self):
        return os.path.join(self.config.save_dir, "raw")

    @property
    def processed_raw_data_dir(self):
        """
        filename for processed raw data, i.e., image extraction
        """
        z_score_image_name, z_score_kinem_name = self.get_z_scoring_names()
        multi_unit_name = "_multiUnit" if self.config.extract_multi_unit else ""
        return os.path.join(
            self.config.save_dir,
            f"processed_raw_data{multi_unit_name}_{self.config.delta:.0f}ms_{z_score_image_name}_{z_score_kinem_name}{self.fr_threshold_name}",
        )

    @property
    def processed_segments_data_dir(self):
        """
        data dir for constructing the segmented trials from extracted images
        """
        return os.path.join(
            self.config.save_dir,
            f"processed_segments_{self.segments_processing_hash_str}",
        )

    @property
    def metadata_path(self):
        return os.path.join(
            self.config.save_dir,
            f"metadata_{self.segments_processing_hash_str}.csv",
        )

    @property
    def processed_segment_pattern(self):
        return "subject_session_segid"

    @property
    def raw_data_downloaded_indicator_file_path(self):
        return os.path.join(self.raw_data_dir, "download_complete.done")

    @property
    def fr_threshold_name(self):
        if self.config.fr_threshold > 0:
            return f"_frThrs{self.config.fr_threshold:.0f}Hz"
        else:
            return ""

    def get_segment_path(self, **kwargs):
        if self.processed_segment_pattern == "subject_session_segid":
            segment_filename = (
                f"{kwargs['subject']}_{kwargs['session']}_{kwargs['segment_id']}.pt"
            )
            return (
                os.path.join(self.processed_segments_data_dir, segment_filename),
                segment_filename,
            )
        else:
            raise NotImplementedError(
                f"Only 'subject_session_segid' pattern is supported for saving segment files."
            )

    def get_segment_id_from_path(self, path):
        if self.processed_segment_pattern == "subject_session_segid":
            segment_filename = os.path.split(path)[-1]
            segment_id = int(segment_filename[:-3].split("_")[-1])
            return segment_id
        else:
            raise NotImplementedError(
                f"Only 'subject_session_segid' pattern is supported for saving segment files, segment_id cannot be obtained."
            )

    def _is_downloaded(self):
        self.logger.debug(
            f"Raw data downloaded indicator file path: {self.raw_data_downloaded_indicator_file_path}"
        )
        return os.path.exists(self.raw_data_downloaded_indicator_file_path)

    def _mark_as_downloaded(self):
        open(self.raw_data_downloaded_indicator_file_path, "w").close()

    def download_data(self):
        success = []
        for dc in self.download_command:
            self.logger.info(f"Downloading: {dc}.")

            split_command = dc.split(" ")

                                                                                             
                                                              
                                            
            proc = subprocess.run(split_command)
            success.append(proc.returncode)
        return all([s == 0 for s in success])              

    def get_raw_data_file_path(self, **kwargs):
        raise NotImplementedError

    def get_z_scoring_names(self):
        if self.config.z_score_image:
            z_score_image_name = "zScimage"
        else:
            z_score_image_name = "nozScimage"

        if self.config.z_score_kinem:
            z_score_kinem_name = "zScKinem"
        else:
            z_score_kinem_name = "nozScKinem"
        return z_score_image_name, z_score_kinem_name

    def get_processed_raw_data_file_path(self, subject, session):
        filename = f"{subject}_{session}.pt"
        return os.path.join(self.processed_raw_data_dir, filename)

    def _is_raw_data_processed(self):
        if not os.path.exists(self.processed_raw_data_dir):
            return False

        files_exist = []
        for subject in self.available_sessions.keys():
            for session in self.available_sessions[subject]:
                path = self.get_processed_raw_data_file_path(
                    subject=subject, session=session
                )
                files_exist.append(os.path.exists(path))
        return np.array(files_exist).all()

    def process_raw_data(self):
        """
        should call process_single_session_raw_data
        """
        raise NotImplementedError

    def get_channel_inds_above_fr_threshold(self, image):
        t_all = image.shape[0] * self.config.delta / 1000              
        fr_hz = image.sum(0) / t_all
        keep_bool = fr_hz >= self.config.fr_threshold
        keep_inds = torch.arange(image.shape[1])[keep_bool]
        return keep_inds

    def process_single_session_raw_data(
        self, file_path, subject, session, save_data=True
    ):
        raise NotImplementedError

    def _is_segments_processed(self):
        if not os.path.exists(self.processed_segments_data_dir) or not os.path.exists(
            self.metadata_path
        ):
            return False

        if len(self.metadata):
            paths_exists = self.metadata.apply_fn_on_all_rows("path", os.path.exists)
            return paths_exists.all()
        else:
            return False

    def get_segments_processing_hash(
        self,
        segment_length,
        segment_from_existing_data: Optional[bool] = False,
        existing_data_segment_length: Optional[int] = None,
    ):
        """
        returns a tuple where the key is the processing str, value is the hashed key.
        actual str can be found in metadata.

        this part can be overwritten by each dataset class based on specific settings
        """
        z_score_image_name, z_score_kinem_name = self.get_z_scoring_names()

        if self.config.d_image_multiple > 0:
            d_image_str = f"d_image_mul{self.config.d_image_multiple}"
        else:
            d_image_str = f"d_image{self.config.d_image}"

        trial_align_str = f"_trial_align{self.config.trial_align}"
        if self.config.trial_align:
            segment_length_str = ""
        else:
            segment_length_str = f"_segment_length{segment_length}"

        processing_str = (
            f"delta{self.config.delta:.0f}ms_{d_image_str}{self.fr_threshold_name}{trial_align_str}"
            f"{segment_length_str}_val_ratio{self.config.val_ratio:.1e}_test_ratio{self.config.test_ratio:.1e}"
            f"_{z_score_image_name}_{z_score_kinem_name}{self.fr_threshold_name}"
        )

        if not self.config.sort_units:
            processing_str += "_noSortUnits"

        if self.config.extract_multi_unit:
            processing_str += "_multiUnit"

        if segment_from_existing_data:
            assert (
                existing_data_segment_length is not None
            ), "Segments are asked to be created from an existing segmented data, but segment length of the existing data is None."
            processing_str += f"_from_segment_length{existing_data_segment_length}"

        hash_str = hashlib.sha256(bytes(processing_str, "utf-8")).hexdigest()[:5]
        return processing_str, hash_str

    def process_segments(self):
        if not self.config.segment_from_existing_data:
            for subject in self.available_sessions.keys():
                sessions_count = len(self.available_sessions[subject])
                if sessions_count:                          
                    self.logger.info(
                        f"Segment processing for subject {subject} starts."
                    )
                    for i, session in enumerate(self.available_sessions[subject]):
                        self.logger.info(
                            f"Processing session {session} ({i+1}/{sessions_count})..."
                        )
                        self.process_single_session_segments(
                            subject=subject, session=session
                        )
        else:
            if not self.config.perturb_existing_segments:
                self.process_segments_from_existing_data()
            else:
                self.process_segments_with_perturbation_from_existing_data()
                       
        self.metadata.save(self.metadata_path)

    def save_segment_data(self, data_dict, subject, session, segment_id):
                                                                                        
        segment_path, segment_filename = self.get_segment_path(
            subject=subject, session=session, segment_id=segment_id
        )
        torch.save(data_dict, segment_path)
        return segment_path, segment_filename

    def process_segments_with_perturbation_from_existing_data(self):
                         
        raise NotImplementedError
    
    def process_segments_from_existing_data(self):
        _, existing_segments_processing_hash_str = self.get_segments_processing_hash(
            segment_length=self.config.existing_data_segment_length,
            segment_from_existing_data=False,
        )

        existing_metadata_path = os.path.join(
            self.config.save_dir,
            f"metadata_{existing_segments_processing_hash_str}.csv",
        )
        existing_metadata = Metadata(load_path=existing_metadata_path)

        self.logger.info(
            f"Processing {self.config.segment_length} second segments from {self.config.existing_data_segment_length} second segments with metadata_{existing_segments_processing_hash_str}.csv..."
        )

        num_splits = (
            self.config.existing_data_segment_length // self.config.segment_length
        )
        new_metadata_df = []

        for row_id in range(len(existing_metadata)):
            row = existing_metadata.get_row_by_index(row_id)
            existing_segment_data = torch.load(row.path)
            segment_id = self.get_segment_id_from_path(path=row.path)

                                                         
            new_segment_data = {}
            for name in existing_segment_data.keys():
                existing_num_steps = existing_segment_data[name].shape[0]
                new_num_steps_in_segment = int(
                    self.config.segment_length / (self.config.delta / 1000)
                )
                num_discard_steps = int(existing_num_steps % new_num_steps_in_segment)
                new_data = rearrange(
                    existing_segment_data[name][
                        : (existing_num_steps - num_discard_steps), ...
                    ],
                    "(b t) n -> b t n",
                    t=new_num_steps_in_segment,
                )
                new_segment_data[name] = new_data

                                                              
            for split_id in range(num_splits):
                new_segment_id = segment_id * num_splits + split_id
                data_dict = {
                    k: v[split_id].clone() for k, v in new_segment_data.items()
                }
                new_segment_path, new_segment_filename = self.save_segment_data(
                    data_dict=data_dict,
                    subject=row.subject,
                    session=row.session,
                    segment_id=new_segment_id,
                )

                new_meta_row = row.to_dict()
                new_meta_row["path"] = new_segment_path
                new_meta_row["segment_filename"] = new_segment_filename
                new_meta_row["segments_processing_str"] = self.segments_processing_str
                new_metadata_df.append(new_meta_row)

        new_metadata_df = pd.DataFrame(new_metadata_df)
        self.metadata.concat(new_metadata_df=new_metadata_df)


    def process_single_session_segments(self, subject, session):
        raise NotImplementedError


class BaseNeurotaskDataset(BaseImagingCoreDataset):
    def get_segments_processing_hash(
        self,
        segment_length,
        segment_from_existing_data: Optional[bool] = False,
        existing_data_segment_length: Optional[int] = None,
    ):
        """
        returns a tuple where the key is the processing str, value is the hashed key.
        actual str can be found in metadata.

        this part can be overwritten by each dataset class based on specific settings
        """

        assert (
            not self.config.extract_multi_unit
        ), "Multi-unit images are not allowed for Neurotask datasets, neuron channel names are not available."

        z_score_image_name, z_score_kinem_name = self.get_z_scoring_names()

        if self.config.d_image_multiple > 0:
            d_image_str = f"d_image_mul{self.config.d_image_multiple}"
        else:
            d_image_str = f"d_image{self.config.d_image}"

        trial_align_str = f"trial_align{self.config.trial_align}"
        if self.config.trial_align:
            trial_align_str = f"_event{self.config.event_align}_minOffset{self.config.event_align_min_offset:.1e}_maxOffset{self.config.event_align_max_offset:.1e}"

        chunk_seg_str = f"chunk_segment{self.config.chunk_segments}"
        if not self.config.chunk_segments:
            chunk_seg_str += f"_minTrialSegmentLength{self.config.min_trial_segment_length:.1e}_maxTrialSegmentLength{self.config.max_trial_segment_length:.1e}"

        processing_str = (
            f"delta{self.config.delta:.0f}ms_{d_image_str}{self.fr_threshold_name}_{chunk_seg_str}_{trial_align_str}"
            f"_segment_length{segment_length}_val_ratio{self.config.val_ratio:.1e}_test_ratio{self.config.test_ratio:.1e}"
            f"_{z_score_image_name}_{z_score_kinem_name}{self.fr_threshold_name}"
        )

        if not self.config.sort_units:
            processing_str += "_noSortUnits"

        if segment_from_existing_data:
            assert (
                existing_data_segment_length is not None
            ), "Segments are asked to be created from an existing segmented data, but segment length of the existing data is None."
            processing_str += f"_from_segment_length{existing_data_segment_length}"

        hash_str = hashlib.sha256(bytes(processing_str, "utf-8")).hexdigest()[:5]
        return processing_str, hash_str

    @property
    def download_command(self):
        return [
            f"kaggle datasets download -d carolinafilipe/neurotask-multi-tasks-benchmark-dataset -p {self.raw_data_dir}",
            f"unzip {self.raw_data_dir}/neurotask-multi-tasks-benchmark-dataset.zip -d {self.raw_data_dir}",
        ]

    @property
    def neurotask_dataset_identifier(self):
        """
        Neurotask dataset do not identify animals and sessions by unique strings such as an
        animal name or date. We will be using this identifier to assign unique names to subjects
        in each separate dataset, and sessions will be left as they are.

        Therefore, make sure that subject names in imageAvailableSessions match <neurotask_dataset_identifier><subject_id> pattern.
        """
        raise NotImplementedError

    @staticmethod
    def rebin(dataset1, prev_bin_size, new_bin_size, reset=True):
        """
        Taken from: https://github.com/catniplab/NeuroTask/blob/8157454ffdc80f3c4067fee1c1dfca9b76bb9277/api_neurotask.py

        Rebin the given dataset to a new bin size.

        Parameters:
        dataset1 (pd.DataFrame): The dataset to rebin.
        prev_bin_size (int): The previous bin size.
        new_bin_size (int): The new bin size.
        reset (bool): Whether to reset the index and drop specific columns.

        Returns:
        pd.DataFrame: The rebinned dataset.
        """

                                        
        d = dataset1.reset_index()

                                
        bin_size = new_bin_size // prev_bin_size

                                                                             
        grouped = d.groupby(["session", "trial_id", d.index // bin_size])
        agg_functions = {}

                                                                        
        def safe_decimate(x, bin_size):
            if len(x) <= 27:
                return np.mean(x)
            return decimate(x, bin_size, ftype="iir", zero_phase=True).mean()

                                      
        for col in dataset1.columns:
            if col.startswith("Neuron"):
                agg_functions[col] = "sum"
            elif (
                col.startswith("force")
                or col.startswith("hand")
                or col.startswith("finger")
                or col.startswith("cursor")
            ):
                agg_functions[col] = lambda x: safe_decimate(x, bin_size)
            else:
                agg_functions[col] = "max"

                                                           
        data_bin = grouped.agg(agg_functions)

                     
        if reset:
            del data_bin["session"]
            del data_bin["trial_id"]
            data_bin = data_bin.reset_index()
            del data_bin["level_2"]

        return data_bin

    @staticmethod
    def align_trial(df, start_event, bin_size, offset_min=None, offset_max=None):
        """
        Taken from: https://github.com/catniplab/NeuroTask/blob/8157454ffdc80f3c4067fee1c1dfca9b76bb9277/api_neurotask.py

        Align trials in a DataFrame based on a start event and bin size.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            start_event (str): The column name indicating the start event.
            bin_size (int): The bin size of the data.
            offset_min (int, optional): The minimum offset for backward filling. Must be <= 0.
            offset_max (int, optional): The maximum offset for forward filling. Must be >= 0.

        Returns:
            pd.DataFrame: The DataFrame with aligned trials.
        """

        df[start_event] = df[start_event].replace(False, np.nan)
        df["ev"] = df[start_event]

        if offset_min:
            assert offset_min <= 0, "offset_min must be less than or equal to 0"
            offset_min = -offset_min // bin_size
            try:
                df["ev"] = df["ev"].bfill(limit=offset_min).infer_objects(copy=False)
            except TypeError:
                df["ev"] = df["ev"].bfill(limit=offset_min).infer_objects()
        if offset_max:
            assert offset_max >= 0, "offset_max must be greater than or equal to 0"
            offset_max = offset_max // bin_size
            try:
                df["ev"] = df["ev"].ffill(limit=offset_max).infer_objects(copy=None)
            except TypeError:
                df["ev"] = df["ev"].bfill(limit=offset_min).infer_objects()
        else:
            df["ev"] = df["ev"].ffill()

        df = df[(df["ev"] == 1)]
        del df["ev"]

        return df

    @property
    def raw_data_filenames(self):
        raise NotImplementedError

    @property
    def vel_columns(self):
        raise NotImplementedError

    def get_raw_data_file_path(self, filename):
        return os.path.join(self.raw_data_dir, filename)

    @property
    def processed_raw_data_dir(self):
        """
        filename for processed raw data, i.e., image extraction
        """
        z_score_image_name, z_score_kinem_name = self.get_z_scoring_names()
        return os.path.join(
            self.config.save_dir,
            f"processed_raw_data_{self.config.delta:.0f}ms_{z_score_image_name}_{z_score_kinem_name}{self.fr_threshold_name}",
            self.neurotask_dataset_identifier,
        )

    @property
    def processed_segments_data_dir(self):
        """
        data dir for constructing the segmented trials from extracted images
        """
        return os.path.join(
            self.config.save_dir,
            f"processed_segments_{self.segments_processing_hash_str}",
            self.neurotask_dataset_identifier,
        )

    @property
    def metadata_path(self):
        return os.path.join(
            self.config.save_dir,
            f"metadata_{self.segments_processing_hash_str}_{self.neurotask_dataset_identifier}.csv",
        )

    def process_raw_data(self):
        for i, filename in enumerate(self.raw_data_filenames):
            self.logger.info(
                f"Raw data processing (i.e., image and/or  extraction) for file {filename} starts."
            )
            raw_file_path = self.get_raw_data_file_path(filename=filename)
            self.logger.info(
                f"Processing file {filename} ({i+1}/{len(self.raw_data_filenames)})..."
            )
            self.process_single_session_raw_data(file_path=raw_file_path)

    def process_single_session_raw_data(self, file_path):
        df = pd.read_parquet(file_path)
        raw_delta = int(file_path.split("_")[1])
        grouped_df = df.groupby(["animal", "session"])
        for i, ((df_subject, df_session), group) in enumerate(grouped_df):
            self.logger.info(
                f"Processing session ({i+1}/{len(grouped_df)}) inside the file..."
            )

                                                                                                                    
                                                                                   
            group = group[group.result.isin(self.config.include_trial_results)]

                                        
            group = group.dropna(axis=1, how="all")

                                                 
            if self.config.delta > raw_delta:
                group = self.rebin(
                    group, prev_bin_size=raw_delta, new_bin_size=self.config.delta
                ).reset_index(drop=True)

                               
            if self.config.z_score_kinem:
                cursor_vel = torch.tensor(group[self.vel_columns].to_numpy())

                cursor_vel_std = cursor_vel.std(dim=0)
                cursor_vel_std[cursor_vel_std == 0] = 1

                cursor_vel = (
                    cursor_vel - cursor_vel.mean(dim=0)[None, :]
                ) / cursor_vel_std[None, :]

                group[self.vel_columns] = cursor_vel

            if self.config.z_score_image:
                image_cols = [i for i in group.columns if i.startswith("Neuron")]
                image = torch.tensor(group[image_cols].to_numpy())

                data_std = image.std(dim=0)
                data_std[data_std == 0] = 1
                image = (image - image.mean(dim=0)[None, :]) / data_std[None, :]
                group[image_cols] = image

            subject = list(self.available_sessions.keys())[df_subject - 1]
            session = f"{self.experiment_type}_{df_session}"
            save_path = self.get_processed_raw_data_file_path(
                subject=subject, session=session
            )
            torch.save(group, save_path)

    def process_single_session_segments(self, subject, session):
        processed_raw_data_path = self.get_processed_raw_data_file_path(
            subject=subject, session=session
        )
        data_df = torch.load(processed_raw_data_path)

                                          
        image = data_df[
            [i for i in data_df.columns if i.startswith("Neuron")]
        ].to_numpy()
        vel = data_df[self.vel_columns].to_numpy()

                                                           
        all_image = torch.tensor(image, dtype=torch.float32)
        all_vel = torch.tensor(vel, dtype=torch.float32)
        if self.config.sort_units:
            all_image, sorted_image_inds = self.sort_images_on_r2(
                image=all_image, kinem=all_vel
            )
        else:
            sorted_image_inds = None

        if self.config.chunk_segments:
                                                                                                     
                                                        
            num_steps = all_image.shape[0]

            num_steps_in_segment = int(
                self.config.segment_length / (self.config.delta / 1000)
            )
                                                                                    
                                                                  
            num_discard_steps = int(num_steps % num_steps_in_segment)

            segmented_image = rearrange(
                all_image[:-num_discard_steps, :],
                "(b t) n -> b t n",
                t=num_steps_in_segment,
            )
            segmented_vel = rearrange(
                all_vel[:-num_discard_steps, :],
                "(b t) n -> b t n",
                t=num_steps_in_segment,
            )
            segmented_image = list(segmented_image)
            segmented_vel = list(segmented_vel)
        else:
                                                                       
                                                             
            if self.config.trial_align:
                assert (
                    self.config.event_align_max_offset
                    - self.config.event_align_min_offset
                    >= self.config.min_trial_segment_length
                ), (
                    f"Difference between config.event_align_min_offset: {self.config.event_align_min_offset} and config.event_align_max_offset{self.config.event_align_max_offset}"
                    f"is smaller than config.min_trial_segment_length: {self.config.min_trial_segment_length}!"
                )
                try:
                    data_df = self.align_trial(
                        data_df,
                        self.config.event_align,                       
                        bin_size=self.config.delta,
                        offset_min=(
                            self.config.event_align_min_offset
                            / (self.config.delta / 1000)
                        ),                      
                        offset_max=(
                            self.config.event_align_max_offset
                            / (self.config.delta / 1000)
                        ),                     
                    )
                except Exception as e:
                    raise f"{e}, please check {self.config.event_align} exists in the dataframe."

            trial_ids = data_df["trial_id"].unique()
            segmented_image, segmented_vel = [], []
            for tr in trial_ids:
                data_df_trial = data_df[data_df["trial_id"] == tr]
                image = data_df_trial[
                    [i for i in data_df_trial.columns if i.startswith("Neuron")]
                ].to_numpy()
                vel = data_df_trial[self.vel_columns].to_numpy()

                if sorted_image_inds is not None:
                    image = image[..., sorted_image_inds]

                image = torch.tensor(image, dtype=torch.float32)
                vel = torch.tensor(vel, dtype=torch.float32)

                num_steps = image.shape[0]
                max_trial_segment_steps = int(
                    self.config.max_trial_segment_length / (self.config.delta / 1000)
                )
                min_trial_segment_steps = int(
                    self.config.min_trial_segment_length / (self.config.delta / 1000)
                )
                num_discard_steps = int(num_steps % max_trial_segment_steps)

                if num_discard_steps >= min_trial_segment_steps:
                    segmented_image.append(image[-num_discard_steps:, :])
                    segmented_vel.append(vel[-num_discard_steps:, :])

                if num_steps >= max_trial_segment_steps:
                    image = rearrange(
                        image[:-num_discard_steps, :],
                        "(b t) n -> b t n",
                        t=max_trial_segment_steps,
                    )
                    vel = rearrange(
                        vel[:-num_discard_steps, :],
                        "(b t) n -> b t n",
                        t=max_trial_segment_steps,
                    )
                    for j in range(image.shape[0]):
                        segmented_image.append(image[j, :, :])
                        segmented_vel.append(vel[j, :, :])

                                        
        metadata_df = []
        num_segments = len(segmented_image)
        for i in range(num_segments):
            image, vel = segmented_image[i], segmented_vel[i]

                                    
            d_image = image.shape[-1]
            if self.config.d_image_multiple > 0:
                d_image_keep = d_image - (d_image % self.config.d_image_multiple)
            else:
                d_image_keep = self.config.d_image
            image = image[..., :d_image_keep]

                              
            data_dict = dict(image=image.clone(), kinem=vel.clone())
            segment_path, segment_filename = self.save_segment_data(
                data_dict=data_dict,
                subject=subject,
                session=session,
                segment_id=i,
            )

                                                                  
            meta_row = dict(
                subject=subject,
                session=session,
                subject_session=f"{subject}_{session}",
                experiment_name=self.experiment_type,
                d_image=image.shape[-1],
                d_kinem=vel.shape[-1],
                image_channel_names=[],
                path=segment_path,
                split="train",
                segment_filename=segment_filename,
                segments_processing_str=self.segments_processing_str,
            )
            metadata_df.append(meta_row)

                                                                 
        metadata_df = pd.DataFrame(metadata_df)
        metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)

        val_size = int(self.config.val_ratio * len(metadata_df))
        test_size = int(self.config.test_ratio * len(metadata_df))

        metadata_df.loc[:val_size, "split"] = "val"
        metadata_df.loc[val_size : (val_size + test_size), "split"] = "test"

                                                         
        self.metadata.concat(new_metadata_df=metadata_df)


class BaseMultimodalDataset(BaseImagingCoreDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

    @property
    def column_names(self):
        return super().column_names + [
            "d_",
            "_channel_names",
        ]

    @property
    def _processing_str(self):
        if self.config.z_score_:
            z_score__name = "zSc"
        else:
            z_score__name = "nozSc"
        return f"_lpCut{self.config._lp_cutoff:.1e}Hz_hpCut{self.config._hp_cutoff:.1e}Hz_{z_score__name}"

    @property
    def processed_raw_data_dir(self):
        """
        filename for processed raw data, i.e., image extraction
        """
        return super().processed_raw_data_dir + f"_{self._processing_str}"

    def get_segments_processing_hash(
        self,
        segment_length,
        segment_from_existing_data: Optional[bool] = False,
        existing_data_segment_length: Optional[int] = None,
    ):
        processing_str, _ = super().get_segments_processing_hash(
            segment_length=segment_length,
            segment_from_existing_data=segment_from_existing_data,
            existing_data_segment_length=existing_data_segment_length,
        )

        if self.config.d__multiple > 0:
            d__str = f"d__mul{self.config.d__multiple}"
        else:
            d__str = f"d_{self.config.d_}"

        processing_str = f"{processing_str}_{self._processing_str}_{d__str}"
        hash_str = hashlib.sha256(bytes(processing_str, "utf-8")).hexdigest()[:5]
        return processing_str, hash_str




import hashlib
from typing import Optional
import os                                  

class BaseImagingDataset(BaseImagingCoreDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)


    @property
    def column_names(self):
        return [
            "subject",
            "session",
            "subject_session",
            "experiment_name",
            "d_imaging",
            "d_binary_beh",
            "d_kinem",
            "imaging_modality",
            "path",
            "split",
            "segment_filename",
            "segments_processing_str",
        ]

    @property
    def download_data(self):
        success = []
        for dc in self.download_command:
                                                                                             
                                                              
                                            
            proc = subprocess.run(dc, shell=True)
            success.append(proc.returncode)
        return all([s == 0 for s in success])              

    @property
    def imaging_processing_str(self):
                                                                                     
        imaging_type_name = self.config.imaging_dtype                          
        return f"{imaging_type_name}Imaging"                 


    def get_z_scoring_names(self):
        if self.config.z_score_imaging:
            z_score_imaging_name = "_zScImg"
        else:
            z_score_imaging_name = ""

        if self.config.z_score_binary_behavior:
            z_score_kinem_name = "_zScBinBeh"
        else:
            z_score_kinem_name = ""
        return z_score_imaging_name, z_score_kinem_name

    @property
    def get_spatial_align_name(self):
        return f"_spatialAlign" if self.config.align_to_allen_atlas else ""

    @property
    def get_resize_name(self):
        return f"_{str(self.config.resize_dims[0])}x{str(self.config.resize_dims[1])}" if self.config.resize_dims else ""

    @property
    def get_mask_out_nonallen_name(self):
        return f"_MnAllen" if self.config.mask_out_nonallen else ""


    @property
    def processed_raw_data_dir(self):
        """
        filename for processed raw data, i.e., image extraction
        """
        z_score_image_name, z_score_beh_name = self.get_z_scoring_names()

        pca_str = ""
        if self.config.get('use_pca', False):
            pca_str = f"_pca{self.config.get('num_pcs', 'default')}"

        return os.path.join(
            self.config.save_dir,
            f"processed_raw{z_score_image_name}{z_score_beh_name}"
            f"{self.get_spatial_align_name}"
            f"{self.get_mask_out_nonallen_name}"
            f"{self.get_resize_name}"
            f"{pca_str}"
        )

    @property
    def processed_segments_data_dir(self):
        """
        data dir for constructing the segmented trials from extracted images
        """
        return os.path.join(
            self.config.save_dir,
            f"processed_segments_{self.segments_processing_hash_str}",
        )


    def get_segments_processing_hash(
        self,
        segment_length,
        segment_from_existing_data: Optional[bool] = False,
        existing_data_segment_length: Optional[int] = None,
    ):
        """
        Returns a tuple where the key is the processing str, value is the hashed key,
        specific for imaging data processing.
        """

                                                
                                                                              
               
                                                                 

        trial_align_str = f"_trial_align{self.config.trial_align}"
        if self.config.trial_align:
            segment_length_str = ""
        else:
            segment_length_str = f"_segment_length{segment_length}"                                  

        pca_str = ""
        if self.config.get('use_pca', False):
            pca_str = f"_pca{self.config.get('num_pcs', 'default')}"

        processing_str = (
            f"imaging_{trial_align_str}"                          
            f"{segment_length_str}_val_ratio{self.config.val_ratio:.1e}_test_ratio{self.config.test_ratio:.1e}"
            f"_{self.get_z_scoring_names()[0]}"                            
            f"_{self.get_z_scoring_names()[1]}"                            
            f"{self.get_spatial_align_name}"                            
            f"{self.get_mask_out_nonallen_name}"                                        
            f"{self.get_resize_name}"                       
            f"{pca_str}"
        )


        if segment_from_existing_data:
            assert (
                existing_data_segment_length is not None
            ), "Segments are asked to be created from an existing segmented data, but segment length of the existing data is None."
            processing_str += f"_from_segment_length{existing_data_segment_length}"

        if self.config.get("perturb_existing_segments", False):
            p = self.config.perturbation_params
            rot = p.max_rotation_deg
            s_min, s_max = p.scale_range
            sh_x, sh_y = p.max_shift_xy
                                           
            processing_str += f"_perturb_rot{rot}_sc{s_min}-{s_max}_sh{sh_x}-{sh_y}"

        hash_str = hashlib.sha256(bytes(processing_str, "utf-8")).hexdigest()[:5]
        return processing_str, hash_str


    def process_segments_from_existing_data(self):
        """
        Overriding the base class method to support 4D imaging tensors 
        [time, channels, height, width] alongside 2D behavior tensors.
        """
        _, existing_segments_processing_hash_str = self.get_segments_processing_hash(
            segment_length=self.config.existing_data_segment_length,
            segment_from_existing_data=False,
        )

        existing_metadata_path = os.path.join(
            self.config.save_dir,
            f"metadata_{existing_segments_processing_hash_str}.csv",
        )
        existing_metadata = Metadata(load_path=existing_metadata_path)

        self.logger.info(
            f"Processing {self.config.segment_length} second segments from {self.config.existing_data_segment_length} second segments with metadata_{existing_segments_processing_hash_str}.csv..."
        )

        num_splits = int(
            np.round(self.config.existing_data_segment_length * (self.config.delta / 1000)) // self.config.segment_length
        )
        new_metadata_df = []

        for row_id in range(len(existing_metadata)):
            row = existing_metadata.get_row_by_index(row_id)
            existing_segment_data = torch.load(row.path)
            segment_id = self.get_segment_id_from_path(path=row.path)

                                                           
            new_segment_data = {}
            for name in existing_segment_data.keys():
                if name not in ["imaging", "kinem"]:
                    new_segment_data[name] = existing_segment_data[name]                         
                    continue
                existing_num_steps = existing_segment_data[name].shape[0]
                new_num_steps_in_segment = int(
                    self.config.segment_length / (self.config.delta / 1000)
                )
                num_discard_steps = int(existing_num_steps % new_num_steps_in_segment)
                
                                                       
                                                                                         
                new_data = rearrange(
                    existing_segment_data[name][
                        : (existing_num_steps - num_discard_steps), ...
                    ],
                    "(b t) ... -> b t ...", 
                    t=new_num_steps_in_segment,
                )
                new_segment_data[name] = new_data

                                                              
            for split_id in range(num_splits):
                new_segment_id = segment_id * num_splits + split_id
                data_dict = {
                    k: v[split_id].clone() for k, v in new_segment_data.items() if k in ["imaging", "kinem"]                                                             
                }
                                                                                                                                 
                data_dict.update({k: v for k, v in existing_segment_data.items() if k not in ["imaging", "kinem"]})

                new_segment_path, new_segment_filename = self.save_segment_data(
                    data_dict=data_dict,
                    subject=row.subject,
                    session=row.session,
                    segment_id=new_segment_id,
                )

                new_meta_row = row.to_dict()
                new_meta_row["path"] = new_segment_path
                new_meta_row["segment_filename"] = new_segment_filename
                new_meta_row["segments_processing_str"] = self.segments_processing_str
                new_metadata_df.append(new_meta_row)

        new_metadata_df = pd.DataFrame(new_metadata_df)
        self.metadata.concat(new_metadata_df=new_metadata_df)

    def process_segments_with_perturbation_from_existing_data(self):
        """
        Loads existing unperturbed segments, applies a consistent random affine 
        transformation per session (skipping no_shift_sessions), and saves them.
        """
                                                        
        original_perturb_flag = self.config.perturb_existing_segments
        self.config.perturb_existing_segments = False
        _, clean_hash = self.get_segments_processing_hash(
            segment_length=self.config.existing_data_segment_length,
            segment_from_existing_data=False
        )
        self.config.perturb_existing_segments = original_perturb_flag               
        
        existing_metadata_path = os.path.join(
            self.config.save_dir,
            f"metadata_{clean_hash}.csv",
        )
        self.logger.info(f"Loading clean metadata from {existing_metadata_path} to apply perturbations...")
        existing_metadata = Metadata(load_path=existing_metadata_path)

                                                               
        session_params = {}
        p_cfg = self.config.perturbation_params
        no_shift_sessions = p_cfg.get("no_shift_sessions", [])
        max_rot = p_cfg.get("max_rotation_deg", 10)
        s_min, s_max = p_cfg.get("scale_range", [0.9, 1.1])
        max_sh_x, max_sh_y = p_cfg.get("max_shift_xy", [5, 5])

        unique_sessions = existing_metadata._metadata_df["subject_session"].unique()
        for sess in unique_sessions:
            if sess in no_shift_sessions:
                                                               
                session_params[sess] = {"angle": 0.0, "translate": [0, 0], "scale": 1.0, "shear": 0.0}
            else:
                                                              
                session_params[sess] = {
                    "angle": random.uniform(-max_rot, max_rot),
                    "translate": [random.randint(-max_sh_x, max_sh_x), random.randint(-max_sh_y, max_sh_y)],
                    "scale": random.uniform(s_min, s_max),
                    "shear": 0.0
                }

                                                                         
        os.makedirs(self.processed_segments_data_dir, exist_ok=True)
        params_path = os.path.join(self.processed_segments_data_dir, "perturbation_params.json")
        with open(params_path, "w") as f:
            json.dump(session_params, f, indent=4)
        self.logger.info(f"Saved session perturbation parameters to {params_path}")

                                                                   
        new_metadata_df = []
        for row_id in range(len(existing_metadata)):
            row = existing_metadata.get_row_by_index(row_id)
            segment_data = torch.load(row.path)
            
            sess = row.subject_session
            aff_params = session_params[sess]

                                                  
            img_tensor = segment_data["imaging"] 
            
                                                                      
            if sess not in no_shift_sessions:
                perturbed_img = TF.affine(
                    img_tensor,
                    angle=aff_params["angle"],
                    translate=aff_params["translate"],
                    scale=aff_params["scale"],
                    shear=aff_params["shear"],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0.0                          
                )
                segment_data["imaging"] = perturbed_img

                                            
            segment_id = self.get_segment_id_from_path(row.path)
            new_segment_path, new_segment_filename = self.save_segment_data(
                data_dict=segment_data,
                subject=row.subject,
                session=row.session,
                segment_id=segment_id,
            )

                                 
            new_meta_row = row.to_dict()
            new_meta_row["path"] = new_segment_path
            new_meta_row["segment_filename"] = new_segment_filename
            new_meta_row["segments_processing_str"] = self.segments_processing_str
            new_metadata_df.append(new_meta_row)

                                     
        self.metadata.concat(new_metadata_df=pd.DataFrame(new_metadata_df))        



class BaseImagingDatasetOld(BaseImagingDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)


    @property
    def column_names(self):
        return [
            "subject",
            "session",
            "subject_session",
            "experiment_name",
            "d_imaging",
            "d_binary_beh",
            "d_kinem",
            "imaging_modality",
            "path",
            "split",
            "segment_filename",
            "segments_processing_str",
        ]

    @property
    def download_data(self):
        success = []
        for dc in self.download_command:
                                                                                             
                                                              
                                            
            proc = subprocess.run(dc, shell=True)
            success.append(proc.returncode)
        return all([s == 0 for s in success])              

    @property
    def imaging_processing_str(self):
                                                                                     
        imaging_type_name = self.config.imaging_dtype                          
        return f"{imaging_type_name}Imaging"                 


    def get_z_scoring_names(self):
        if self.config.z_score_imaging:
            z_score_imaging_name = "zScImg"
        else:
            z_score_imaging_name = "noZScImg"

        if self.config.z_score_binary_behavior:
            z_score_kinem_name = "zScBinBeh"
        else:
            z_score_kinem_name = "nozScBinBeh"
        return z_score_imaging_name, z_score_kinem_name

    @property
    def get_spatial_align_name(self):
        return f"_spatialAlign" if self.config.align_to_allen_atlas else ""

    @property
    def get_downsampling_name(self):
        return f"_dsrate{str(self.config.spatial_downsampling_factor)}" if self.config.spatial_downsampling else ""

    @property
    def get_mask_out_nonallen_name(self):
        return f"_masknonallen" if self.config.mask_out_nonallen else ""


    @property
    def processed_raw_data_dir(self):
        """
        filename for processed raw data, i.e., image extraction
        """
        z_score_image_name, z_score_beh_name = self.get_z_scoring_names()
        return os.path.join(
            self.config.save_dir,
            f"processed_raw_data_{z_score_image_name}_{z_score_beh_name}"
            f"{self.get_spatial_align_name}"
            f"{self.get_mask_out_nonallen_name}"
            f"{self.get_downsampling_name}",
        )

    @property
    def processed_segments_data_dir(self):
        """
        data dir for constructing the segmented trials from extracted images
        """
        return os.path.join(
            self.config.save_dir,
            f"processed_segments_{self.segments_processing_hash_str}",
        )


    def get_segments_processing_hash(
        self,
        segment_length,
        segment_from_existing_data: Optional[bool] = False,
        existing_data_segment_length: Optional[int] = None,
        apply_kinem_preprocessing: Optional[bool] = False,
    ):
        """
        Returns a tuple where the key is the processing str, value is the hashed key,
        specific for imaging data processing.
        """

                                                
                                                                              
               
                                                                 

        trial_align_str = f"_trial_align{self.config.trial_align}"
        if self.config.trial_align:
            segment_length_str = ""
        else:
            segment_length_str = f"_segment_length{segment_length}"                                  

        if apply_kinem_preprocessing:
            segment_length_str += "_kinemPre"

        processing_str = (
            f"imaging_{trial_align_str}"                          
            f"{segment_length_str}_val_ratio{self.config.val_ratio:.1e}_test_ratio{self.config.test_ratio:.1e}"
            f"_{self.get_z_scoring_names()[0]}"                            
            f"_{self.get_z_scoring_names()[1]}"                            
            f"{self.get_spatial_align_name}"                            
            f"{self.get_mask_out_nonallen_name}"                                        
            f"{self.get_downsampling_name}"                       
        )


        if segment_from_existing_data:
            assert (
                existing_data_segment_length is not None
            ), "Segments are asked to be created from an existing segmented data, but segment length of the existing data is None."
            processing_str += f"_from_segment_length{existing_data_segment_length}"

        hash_str = hashlib.sha256(bytes(processing_str, "utf-8")).hexdigest()[:5]
        return processing_str, hash_str
    