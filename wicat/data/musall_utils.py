                     
import os
import numpy as np
import pandas as pd
from wicat.utility.matlab_to_python import loadmat
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, decimate                 
from scipy.ndimage import rotate, zoom, shift
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import convolve2d
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA

import cv2
import numpy as np

def preprocess_raw_session_imaging(
        blue_data,
        uv_data,
        config,
        allen_mask=None,
        atlas_transform=None,
):
    """
    Performs the full, heavy preprocessing pipeline on an entire session's data.
    Handles dtype conversion, resizing, filtering, and hemo-correction.
    """
                                                    
    blue_data = blue_data.astype(np.float32)
    uv_data = uv_data.astype(np.float32) if config.hemodynamic_correction and uv_data is not None else None
                             
    if config.pixelwise_demean:
        blue_data -= blue_data.mean(axis=0, keepdims=True)
        if uv_data is not None:
            uv_data -= uv_data.mean(axis=0, keepdims=True)

                                 
    if config.global_std_normalize:
        std = blue_data.std()

        blue_data /= std
        if uv_data is not None:
            uv_std = uv_data.std()
            uv_data /= uv_std

                                             
    if config.do_highpass:
                                                                                   
        T, H, W = blue_data.shape
        blue_flat = blue_data.reshape(T, -1)

        filtered_blue_flat = highpass_filter_filtfilt(blue_flat, cutoff=config.highpass_cutoff,
                                                            fs=config.get('fs', 30))

        blue_data = filtered_blue_flat.reshape(T, H, W).astype(np.float32)
        if uv_data is not None:
            uv_flat = uv_data.reshape(T, -1)
            filtered_uv_flat = highpass_filter_filtfilt(uv_flat, cutoff=config.highpass_cutoff,
                                                            fs=config.get('fs', 30))
            uv_data = filtered_uv_flat.reshape(T, H, W).astype(np.float32)

    if config.get('use_pca', False):
        print(f"Applying PCA denoising with {config.num_pcs} components...")
        T, H, W = blue_data.shape
        
                                                     
        data_2d = blue_data.reshape(T, H * W)
        

        pca = PCA(n_components=config.num_pcs, svd_solver='randomized', random_state=42)
        pca.fit(data_2d)
        print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")

                                                    
        print("Transforming and reconstructing video from top components...")
        data_transformed = pca.transform(data_2d)
        data_reconstructed_2d = pca.inverse_transform(data_transformed)
        
                                                           
        blue_data = data_reconstructed_2d.reshape(T, H, W)
        print("PCA denoising complete.")

        data_2d = uv_data.reshape(T, H * W)
        
        pca = PCA(n_components=config.num_pcs, svd_solver='randomized', random_state=42)
        pca.fit(data_2d)
        print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")

                                                    
        print("Transforming and reconstructing video from top components...")
        data_transformed = pca.transform(data_2d)
        data_reconstructed_2d = pca.inverse_transform(data_transformed)
        
                                                           
        uv_data = data_reconstructed_2d.reshape(T, H, W)
        print("PCA denoising complete.")


                               
    if config.hemodynamic_correction and uv_data is not None:
        imaging_data, _ = widefield_hemocorrect(blue_data, uv_data)                        
        std = imaging_data.std()
        imaging_data /= std
    else:
        imaging_data = blue_data


    if config.get('align_to_allen_atlas', False):
        if allen_mask is not None and atlas_transform is not None:
            M = atlas_transform.squeeze().astype(np.float32)
            target_dims = imaging_data.shape[1:3]          
            allen_mask = cv2.resize(allen_mask.astype(np.float32), (target_dims[1], target_dims[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            
            scale_factor_x = 512 / target_dims[1]
            scale_factor_y = 512 / target_dims[0]
            S_288_to_512 = np.array([
                [scale_factor_x, 0, 0],
                [0, scale_factor_y, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            M = M @ S_288_to_512
            M = M[0:2, :]
            M_inv = cv2.invertAffineTransform(M)
                                                    
            for t in range(T):
                imaging_data[t] = cv2.warpAffine(
                    imaging_data[t],
                    M_inv,
                    (target_dims[1], target_dims[0]),
                    flags=cv2.INTER_LINEAR,                                      
                    borderMode=cv2.BORDER_CONSTANT,                              
                    borderValue=0                                            
                )

                            
            if config.get('mask_out_nonallen', False):
                                                                   
                imaging_data = imaging_data * allen_mask[np.newaxis, :, :]
            

                                                
    if config.resize_dims:
        print(f"Applying spatial resizing to {config.resize_dims}...")
        T, H, W = imaging_data.shape
        new_W, new_H = config.resize_dims
        if H != new_H or W != new_W:
            resized_imaging = np.zeros((T, new_H, new_W), dtype=np.float32)
            for t in range(T):
                                                        
                resized_imaging[t] = cv2.resize(imaging_data[t], 
                                                tuple(config.resize_dims), 
                                                interpolation=cv2.INTER_AREA)
            imaging_data = resized_imaging


    return imaging_data.astype(np.float32)                      


def preprocess_imaging_chunk(data, do_zscore=True):
    """
    Simplified function for minimal, final preprocessing on a data chunk.
    The heavy lifting is now done in preprocess_raw_session_imaging.
    """
                                
    if do_zscore:
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-8:
            data = (data - mean) / std
        else:
            data = data - mean                          

    return data.astype(np.float32)


def butter_highpass(cutoff, fs=30, order=2, output='ba'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    if output == 'sos':
        sos = butter(order, normal_cutoff, btype='high', analog=False, output=output)
        return sos
    elif output == 'ba':
        b, a = butter(order, normal_cutoff, btype='high', analog=False, output=output)
    return b, a


def highpass_filter(data, cutoff, fs=30, order=2, output='ba'):
    b, a = butter_highpass(cutoff, fs, order=order, output=output)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def highpass_filter_filtfilt(data, cutoff, fs=30, order=2):
    sos = butter_highpass(cutoff, fs, order=order, output='sos')
    filtered_data = sosfiltfilt(sos, data, axis=0)
    return filtered_data




def widefield_hemocorrect(data, hemodata, kernel_size=1, pxl_size_per_op=200):
    '''
        pixelwise widefield hemodynamic correction:
        assuming both intrinsic and neural channels
        are demeaned and normalized
    '''
    T, H, W = data.shape

                      
                   
    if kernel_size > 1:
        averaging_kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
        for i in range(T):
            hemodata[i] = convolve2d(hemodata[i], averaging_kernel, mode='same', boundary='symm', fillvalue=0).astype(np.float32)

    data = data.reshape((T, -1))
    hemodata = hemodata.reshape((T, -1))

    theta = np.zeros(data.shape[1])

    inds = np.arange(0, 1 + pxl_size_per_op * np.ceil(data.shape[1] / pxl_size_per_op), pxl_size_per_op, dtype=int)
    for i in range(inds.shape[0] - 1):
        a = data[:, inds[i]:inds[i + 1]]
        b = hemodata[:, inds[i]:inds[i + 1]]
        temp_theta = (a * b).sum(axis=0) / (b * b).sum(axis=0)
        temp_theta[np.isnan(temp_theta)] = 0
        theta[inds[i]:inds[i + 1]] = temp_theta

    data = data - hemodata * theta                                         
    theta = theta.reshape((H, W))
    data = data.reshape((T, H, W))

    return data, theta


def widefield_hemocorrect_chunked(
    data,
    hemodata,
    window_size=300,                                        
    chunk_size=2000,                                        
):
    '''
    A memory-efficient, CAUSAL hemodynamic correction.
    To correct frame `t`, it uses a regression window from `t - window_size + 1` to `t`.
    '''
    T, H, W = data.shape
    num_pixels = H * W

    data_flat = data.reshape((T, -1))
    hemodata_flat = hemodata.reshape((T, -1))

    corrected_data_flat = np.zeros_like(data_flat)

    print(f"Processing {num_pixels} pixels in chunks of {chunk_size}...")
                                
    for i in range(0, num_pixels, chunk_size):
        end_idx = min(i + chunk_size, num_pixels)

        data_chunk = data_flat[:, i:end_idx]
        hemo_chunk = hemodata_flat[:, i:end_idx]

                                                                              
                                                                                      
        data_rolled = sliding_window_view(data_chunk, window_shape=window_size, axis=0)
        hemo_rolled = sliding_window_view(hemo_chunk, window_shape=window_size, axis=0)

        ab_sum = np.sum(data_rolled * hemo_rolled, axis=-1)
        bb_sum = np.sum(hemo_rolled * hemo_rolled, axis=-1)

                                                                                  
        theta_chunk = np.divide(ab_sum, bb_sum, out=np.zeros_like(ab_sum), where=bb_sum!=0)

                                   
                                                                                                     
        start_frame = window_size - 1

                                                                                   
        data_to_correct = data_chunk[start_frame:]
        hemo_to_correct = hemo_chunk[start_frame:]

                                                                                
        corrected_chunk = data_to_correct - hemo_to_correct * theta_chunk

                                                           
        corrected_data_flat[start_frame:, i:end_idx] = corrected_chunk

                                                       
                                                                                                      
        first_theta = theta_chunk[0]
        data_edge_start = data_chunk[:start_frame]
        hemo_edge_start = hemo_chunk[:start_frame]
        corrected_data_flat[:start_frame, i:end_idx] = data_edge_start - hemo_edge_start * first_theta
                                                        

    print("Correction complete.")
    return corrected_data_flat.reshape((T, H, W))


def widefield_svd_hemo_correct(U, blueV, hemoV, sRate, smooth_blue=False):
    """
    Performs hemodynamic correction on widefield calcium imaging data in SVD space.

    (Docstring - same as before, no changes needed here)
    """
    hemo_smooth = 10                             

                                             
    A, B, C = blueV.shape
    blueV = blueV.transpose(0,2,1).reshape(A, -1).T                                           
    hemoV = hemoV.transpose(0,2,1).reshape(A, -1).T

                    
    blueV = blueV - np.nanmean(blueV, axis=0)
    hemoV = hemoV - np.nanmean(hemoV, axis=0)


                                                  
    b, a = butter(4, 0.1 / sRate, 'highpass')
    not_nan_mask = ~np.isnan(blueV[:, 0])
    blueV[not_nan_mask, :] = filtfilt(b, a, blueV[not_nan_mask, :].astype(np.float32), axis=0).astype(np.float32)
    hemoV[not_nan_mask, :] = filtfilt(b, a, hemoV[not_nan_mask, :].astype(np.float32), axis=0).astype(np.float32)



                          
    temp = np.nansum(blueV, axis=1) * np.nansum(hemoV, axis=1)
    temp = np.abs((temp - np.nanmedian(temp)) / np.nanstd(temp))
    rej_idx = np.where(temp > 100)[0]
    rej_idx = np.concatenate([rej_idx, rej_idx + 1])
    blueV[rej_idx, :] = np.nan
    hemoV[rej_idx, :] = np.nan


                            
    mask = np.isnan(U[:, :, 0])
                                           
    U = U.transpose(1,0,2).reshape(U.shape[0] * U.shape[1], -1)                                
    U = U[~mask.T.flatten(), :]                                





                   
    if sRate > hemo_smooth:
        b, a = butter(4, hemo_smooth / sRate, 'lowpass')
        blueV = blueV.T.reshape(A, C, B).transpose(0,2,1)                                                       
        hemoV = hemoV.T.reshape(A, C, B).transpose(0,2,1)

        for iTrials in range(C):
            cIdx = ~np.isnan(hemoV[0, :, iTrials])                  
            if smooth_blue:
                temp = blueV[:, cIdx, iTrials].T
                temp = np.pad(temp, ((10, 10), (0, 0)), 'edge')                       
                temp = filtfilt(b, a, temp.astype(np.float32), axis=0).astype(np.float32).T
                blueV[:, cIdx, iTrials] = temp[:, 10:-10]

            temp = hemoV[:, cIdx, iTrials].T
            temp = np.pad(temp, ((10, 10), (0, 0)), 'edge')      
            temp = filtfilt(b, a, temp.astype(np.float32), axis=0).astype(np.float32).T
            hemoV[:, cIdx, iTrials] = temp[:, 10:-10]


        blueV = blueV.transpose(0,2,1).reshape(A, -1).T                            
        hemoV = hemoV.transpose(0,2,1).reshape(A, -1).T



                                                                  
    regC = np.zeros(U.shape[0], dtype=np.float32)
    ind = np.arange(0, U.shape[0] + 1, 5000)                               

    for x in range(len(ind) - 1):                              
        start = ind[x]
        end = ind[x+1] if x < len(ind) - 2 else U.shape[0]                     

        a = U[start:end, :] @ blueV.T
        b = U[start:end, :] @ hemoV.T
        regC[start:end] = np.nansum(a * b, axis=1) / np.nansum(b * b, axis=1)



                                                             
    T = np.linalg.pinv(U) @ (regC[:, np.newaxis] * U)



                         
    Vout = blueV - hemoV @ T.T
    Vout = Vout - np.nanmean(Vout, axis=0)

                                
    f1Pow = np.nansum(blueV.flatten()**2)
    f1Powcor = np.nansum(Vout.flatten()**2)
    hemoVar = 100 * (f1Pow - f1Powcor) / f1Pow
    print(f"{hemoVar:.6f} percent variance explained by hemo signal")

                                                           
    Vout = Vout.T.reshape(A, C, B).transpose(0,2,1)

    return Vout, regC, T, hemoVar


             
                                                                         
                                                                         
                                 
                                 
                         
                          
     
                                                                                                        


def loadmask(mask_path):
    atlas = loadmat(mask_path)
    atlasarea = 1 - atlas['dorsalMaps']['allenMask']
    return atlasarea.astype(np.float32)                  


def spatial_downsample(U, resize_dims, method=cv2.INTER_LINEAR):
    return cv2.resize(U, (resize_dims[1], resize_dims[0]), method)


def floor_np(x):
    """
    Numpy-aware floor function that handles potential numpy float inputs correctly.
    """
    return int(np.floor(x))


def align_allen_transform_image(im, trans_params):
    """
    Take a brain image and rotate, scale, and translate it according to the
    parameters in trans_params (produced by alignBrainToAllen GUI - MATLAB).

    Args:
        im (numpy.ndarray): A brain image or image stack (height, width, channels/depth).
        trans_params (dict): Dictionary containing transformation parameters
                             ('angleD', 'scaleConst', 'tC', etc.).

    Returns:
        numpy.ndarray: Transformed brain image, with pixels outside the original
                       defined area set to NaN.
    """
    offset = 50.0                                
    d_size = im.shape


                                                                                    
    the_min = np.nanmin(im)                                             
    min_idx_flat = np.nanargmin(im)                                  

    im = im - the_min + offset

                                                                           
    angle_degrees = trans_params['angleD']                          
    im_rotated = rotate(im, angle=angle_degrees, order=1, reshape=True, cval=np.nan)                                      

                                                                        
    scale_const = trans_params['scaleConst']                              
    if scale_const != 1:
        im_scaled = zoom(im_rotated, zoom=scale_const, order=1, cval=np.nan)
    else:
        im_scaled = im_rotated                    

                                                                  
    nans = np.isnan(im_scaled)
    if np.any(nans):
        im_scaled[nans] = 0

                                         
        translation_coords = trans_params['tC'][::-1]                                              
    im_translated = shift(im_scaled, shift=translation_coords, order=1, cval=0)                                           

                                                                                         
    im_translated[im_translated < offset] = np.nan

                    
    im_final = im_translated + the_min - offset

                                                                        
                                                              
    min_idx_unflat = np.unravel_index(min_idx_flat, d_size)

    im_final[min_idx_unflat[0], min_idx_unflat[1]] = the_min

    new_size = im_final.shape
    trim_h = floor_np((new_size[0] - d_size[0]) / 2) if im.ndim >= 2 else 0
    trim_w = floor_np((new_size[1] - d_size[1]) / 2) if im.ndim >= 2 else 0

    im_trimmed = im_final[trim_h:trim_h + d_size[0], trim_w:trim_w + d_size[1]]

    if im_trimmed.shape != d_size:                                                         
        im_trimmed = im_trimmed.reshape(d_size)


    return im_trimmed


def select_behavior_trials(bhv, trials, nTrials_input=None):
    """[Your select_behavior_trials function code here]"""
                                                             
    if not bhv:
        bFields = []
    else:
        bFields = list(bhv.keys())

    nTrials = nTrials_input
    if 'nTrials' in bhv:
        if isinstance(bhv['nTrials'], list) or isinstance(bhv['nTrials'], np.ndarray):
            bhv['nTrials'] = np.sum(bhv['nTrials'])                                          
                                           
        nTrials = bhv['nTrials']

    if trials is None:                                                 
        return bhv

                                                                                       
    if not isinstance(trials, bool) and (not isinstance(trials, np.ndarray) or trials.dtype != bool):                                      
        temp = np.zeros(nTrials, dtype=bool)
                                                                                           
        valid_trial_indices = trials[np.logical_and(trials >= 1, trials <= nTrials)]                                                         
        if valid_trial_indices.size < len(trials):
            print('Warning: Trial index contains values outside the valid range (1 to nTrials). These indices will be ignored.')

        temp[valid_trial_indices.astype(int)-1] = True                                 

        if len(temp) != nTrials:
            print('Warning: Trial index is larger as available trials in behavioral dataset')
            trials = temp[:nTrials]
        else:
            trials = temp
    else:
        if len(trials) != nTrials:
            print('Warning: Trial index has different length as available trials in behavioral dataset')
            trials = trials[:nTrials]


    if 'nTrials' in bhv:
        bhv['nTrials'] = np.sum(trials)                                       


                                                                    
    for field_name in bFields:
        field_data = bhv[field_name]
        if not any(dim == len(trials) for dim in np.shape(field_data if isinstance(field_data, np.ndarray) else [field_data] if field_data is not None else []) ):                                                              

            if isinstance(field_data, dict):                                              
                tFields = list(field_data.keys())
                if tFields:                                
                    if len(field_data[tFields[0]]) == len(trials) and (isinstance(field_data[tFields[0]], list) or isinstance(field_data[tFields[0]], np.ndarray)):                                                      
                         bhv[field_name][tFields[0]] = _array_index(field_data[tFields[0]], trials)                                                
                    else:
                        pass                                                                                                                      
                else:
                    pass                                                   

            else:
                pass                                                                                                                      


        else:                                    
            if isinstance(field_data, list) or (isinstance(field_data, np.ndarray) and field_data.ndim == 1):           
                bhv[field_name] = _array_index(field_data, trials)                                                
            elif isinstance(field_data, np.ndarray):                   
                cIdx = -1                                        
                for i, dim_size in enumerate(field_data.shape):                      
                    if dim_size == len(trials):
                        cIdx = i
                        break
                if cIdx != -1:                             
                    bhv[field_name] = _array_index_dim(field_data, trials, cIdx)                                              

    return bhv


def _array_index(array_like, indices):
    """
    Indexes a list or numpy array with a boolean or integer index array.
    Handles both list and numpy array inputs for flexibility.
    """
    if isinstance(array_like, list):
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            return [item for i, item in enumerate(array_like) if indices[i]]
        elif isinstance(indices, list) or isinstance(indices, np.ndarray):               
            return [array_like[i] for i in np.array(indices).astype(int)-1]                      
        else:                      
             return [array_like[int(indices)-1]]              

    elif isinstance(array_like, np.ndarray):
        return array_like[indices]
    else:                                                                       
        return array_like


def _array_index_dim(array, indices, dim_index):
    """
    Indexes a multi-dimensional numpy array along a specified dimension using a boolean or integer index array.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input array must be a numpy ndarray.")
    if not isinstance(dim_index, int):
        raise TypeError("Dimension index must be an integer.")
    if dim_index < 0 or dim_index >= array.ndim:
        raise ValueError("Dimension index out of bounds.")

                                              
    index_tuple = [slice(None)] * array.ndim                                          
    index_tuple[dim_index] = indices                                                   
    return array[tuple(index_tuple)]                                               


def smooth_data(data, window_length, polyorder, frames):
    """
    Smooths data using Savitzky-Golay filter and pads with NaNs to match frames length.

    Args:
        data (np.ndarray): 1D array of data to smooth.
        window_length (int): Window length for Savitzky-Golay filter.
        polyorder (int): Polynomial order for Savitzky-Golay filter.
        frames (int): Desired length of the output data (UNUSED in this version as we keep original length).

    Returns:
        np.ndarray: Smoothed data (no padding in this version).
    """
    if not data.size:
        return np.array([])                                      

    smoothed_data = savgol_filter(data, window_length, polyorder, mode='interp')                                         

    return smoothed_data


def reshape_video_data(video_data, bhvDimCnt):
    """Reshapes video data to trial format, handling potential errors due to inconsistent trial numbers."""
    original_shape = video_data.shape
    reshaped_vData = np.reshape(video_data, (original_shape[0], -1, bhvDimCnt))                                      

    return reshaped_vData


def extract_time_series_data(cPath, Animal, Paradigm, Rec, dType='Widefield', use_prehc_pcs=False):
    """[Your extract_time_series_data function code here]"""
                                                               
    if not cPath.endswith(os.sep):
        cPath += os.sep

    if not dType:
        dType = 'Widefield'           

    if dType.lower() == 'twop':
        sRate = 31                                           
        piezoLine = 5
        stimLine = 4
    elif dType.lower() == 'widefield':
        sRate = 30                                                
        piezoLine = 2
        stimLine = 6
    else:
        raise ValueError("Invalid dType. Choose 'Widefield' or 'twoP'")

    preStimDur = np.ceil(1.8 * sRate) / sRate                                       
    postStimDur = np.ceil(4.5 * sRate) / sRate                                            
    frames = int(round((preStimDur + postStimDur) * sRate))                              
    trialDur = frames * (1 / sRate)                                

    mPreTime = np.ceil(0.5 * sRate) / sRate
    mPostTime = np.ceil(2 * sRate) / sRate
    motorIdx = np.arange(int(-mPreTime * sRate), int(mPostTime * sRate) + 1)                    
    tapDur = 0.1
    leverMoveDur_sec = 0.25            
    leverMoveDur = int(np.ceil(leverMoveDur_sec * sRate))                   
    ridgeFolds = 10
    opMotorLabels = ['lLick', 'rLick', 'lGrab', 'lGrabRel', 'rGrab', 'rGrabRel']
    bhvDimCnt = 200
    gaussShift = 1
    trialSegments = [np.arange(1, 55), np.arange(55, 82), np.arange(89, 133), np.arange(140, 163), np.arange(170, 189)]                           
    motorLabels = ['motor_label_1', 'motor_label_2']                                                      

                     
    cPath_full = os.path.join(cPath, Animal, Rec)                       
    sPath = os.path.join(r'\\grid-hs\churchland_nlsas_data\data\BpodImager\Animals', Animal, Paradigm, Rec)                                      

                        
    bhv_files = [f for f in os.listdir(cPath_full) if f.startswith(f"{Animal}_{Paradigm}_") and f.endswith(".mat")]
    if not bhv_files:
        bhv_files_server = [f for f in os.listdir(sPath) if f.startswith(f"{Animal}_{Paradigm}_") and f.endswith(".mat")]
        if bhv_files_server:
             bhv_file_path = os.path.join(sPath, bhv_files_server[0])                            
             bhv_data = loadmat(bhv_file_path)                                                    
        else:
            raise FileNotFoundError(f"No behavior file found in {cPath_full} or {sPath}")

    else:
        bhv_file_path = os.path.join(cPath_full, bhv_files[0])                            
        bhv_data = loadmat(bhv_file_path)                     

    SessionData = bhv_data['SessionData']
    SessionData['TrialStartTime'] = SessionData['TrialStartTime'] * 86400                      
    SessionData_TrialStartTime = SessionData['TrialStartTime']
    if dType.lower() == 'widefield':
        if use_prehc_pcs:
            vc_file_path = [os.path.join(cPath_full, 'blueV.mat'), os.path.join(cPath_full, 'hemoV.mat')]
        else:
            vc_file_path = [os.path.join(cPath_full, 'Vc.mat')]
        mask_file_path = os.path.join(cPath_full, 'mask.mat')

        if not os.path.exists(vc_file_path[0]) or not os.path.exists(mask_file_path):
            print(f"Vc.mat or mask.mat not found locally, copying from server: {sPath}")
            server_vc_path = os.path.join(sPath, 'Vc.mat')
            server_mask_path = os.path.join(sPath, 'mask.mat')
            bhv_files_server = [f for f in os.listdir(sPath) if f.startswith(f"{Animal}_{Paradigm}") and f.endswith(".mat")]
            server_bhv_file_path = os.path.join(sPath, bhv_files_server[0])

            if os.path.exists(server_vc_path) and os.path.exists(server_mask_path) and bhv_files_server:
                import shutil
                shutil.copyfile(server_vc_path, vc_file_path)
                shutil.copyfile(server_mask_path, mask_file_path)
                shutil.copyfile(server_bhv_file_path, os.path.join(cPath_full, bhv_files_server[0]))                                     
            else:
                 raise FileNotFoundError(f"Vc.mat or mask.mat not found on server: {sPath}")

        mask_data = loadmat(mask_file_path)                     
        mask = mask_data['mask']
        if use_prehc_pcs:
            blue_data = loadmat(vc_file_path[0])                     
            hemo_data = loadmat(vc_file_path[1])                     
            blueV = blue_data['blueV']
            hemoV = hemo_data['hemoV']
            U = blue_data['U']
            Vc = widefield_svd_hemo_correct(U, blueV, hemoV, sRate)[0]
            trials_vc = blue_data['trials']
        else:
            vc_data = loadmat(vc_file_path[0])                     
            Vc = vc_data['Vc']
            U = vc_data['U']
            trials_vc = vc_data['trials']

        dims = Vc.shape[0]
        Vc = Vc[:dims, :, :]

        ind_trials_too_many = trials_vc > SessionData['nTrials']
        trials_vc = trials_vc[~ind_trials_too_many]
        Vc = Vc[:, :, ~ind_trials_too_many.flatten()]                                    
        bTrials_var_exists = 'bTrials' in locals() or 'bTrials' in globals()                                     
        if not bTrials_var_exists:                                          
            bTrials = trials_vc.astype(int)                                   


    elif dType.lower() == 'twop':
        data_file_path = os.path.join(cPath_full, 'data.mat')
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"TwoP data file 'data.mat' not found in {cPath_full}")
        data_loaded = loadmat(data_file_path)                     
        data = data_loaded['data']

        bTrials = data['trialNumbers'].flatten()                              
        trials_2p = bTrials.copy()                            
        bTrials = bTrials[np.isin(data['trialNumbers'], data['bhvTrials'])].flatten()                                      


        valid_trials_mask = ~(SessionData['DidNotChoose'][bTrials-1] | SessionData['DidNotLever'][bTrials-1] | ~SessionData['Assisted'][bTrials-1])
        bTrials = bTrials[valid_trials_mask].astype(int)                                       


        data_dFOF = data['dFOF']
        data_DS = data['DS']
        data_analog = data['analog']

        data_dFOF_filtered = data_dFOF[:,:, np.isin(data['trialNumbers'], bTrials).flatten()]             
        data_DS_filtered = data_DS[:,:, np.isin(data['trialNumbers'], bTrials).flatten()]
        data_analog_filtered = data_analog[:,:, np.isin(data['trialNumbers'], bTrials).flatten()]

        data_dFOF = data_dFOF_filtered
        data_DS = data_DS_filtered
        data_analog = data_analog_filtered

        Vc = data_dFOF                                  
        dims = data_dFOF.shape[0]                                   
        trials = trials_2p                                        


    bhv = select_behavior_trials(SessionData, bTrials)                                           
    trialCnt = len(bTrials)

                              
    bhv_video_dir = os.path.join(cPath_full, 'BehaviorVideo')
    sPath_bhv_video = os.path.join(sPath, 'BehaviorVideo')

    svd_combined_segments_path = os.path.join(bhv_video_dir, 'SVD_CombinedSegments.mat')
    motion_svd_combined_segments_path = os.path.join(bhv_video_dir, 'motionSVD_CombinedSegments.mat')

    if not os.path.exists(svd_combined_segments_path) or not os.path.exists(motion_svd_combined_segments_path):
        print(f"Behavior video SVD files not found locally, copying from server: {sPath_bhv_video}")
        if not os.path.exists(bhv_video_dir):
            os.makedirs(bhv_video_dir, exist_ok=True)

        server_svd_combined_segments_path = os.path.join(sPath_bhv_video, 'SVD_CombinedSegments.mat')
        server_motion_svd_combined_segments_path = os.path.join(sPath_bhv_video, 'motionSVD_CombinedSegments.mat')
        server_filtered_pupil_path = os.path.join(sPath_bhv_video, 'FilteredPupil.mat')
        server_segInd1_path = os.path.join(sPath_bhv_video, 'segInd1.mat')
        server_segInd2_path = os.path.join(sPath_bhv_video, 'segInd2.mat')

        if os.path.exists(server_svd_combined_segments_path):
            import shutil
            shutil.copyfile(server_svd_combined_segments_path, svd_combined_segments_path)
            shutil.copyfile(server_motion_svd_combined_segments_path, motion_svd_combined_segments_path)
            shutil.copyfile(server_filtered_pupil_path, os.path.join(bhv_video_dir, 'FilteredPupil.mat'))
            shutil.copyfile(server_segInd1_path, os.path.join(bhv_video_dir, 'segInd1.mat'))
            shutil.copyfile(server_segInd2_path, os.path.join(bhv_video_dir, 'segInd2.mat'))

        mov_files_server = [f for f in os.listdir(sPath_bhv_video) if f.endswith('Video_*1.mj2')]
        if mov_files_server:
             shutil.copyfile(os.path.join(sPath_bhv_video, mov_files_server[0]), os.path.join(bhv_video_dir, mov_files_server[0]))

        mov_files_server2 = [f for f in os.listdir(sPath_bhv_video) if f.endswith('Video_*2.mj2')]
        if mov_files_server2:
             shutil.copyfile(os.path.join(sPath_bhv_video, mov_files_server2[0]), os.path.join(bhv_video_dir, mov_files_server2[0]))


        svd_files_server = [f for f in os.listdir(sPath_bhv_video) if 'SVD' in f and 'Seg' in f]
        for svd_file in svd_files_server:
            shutil.copyfile(os.path.join(sPath_bhv_video, svd_file), os.path.join(bhv_video_dir, svd_file))


    vidv_data = loadmat(svd_combined_segments_path)                     
    vidV = vidv_data['vidV']
    V1 = vidV[:, :bhvDimCnt]

    motion_vidv_data = loadmat(motion_svd_combined_segments_path)                     
    vidV_motion = motion_vidv_data['vidV']
    V2 = vidV_motion[:, :bhvDimCnt]

    pupil_data = loadmat(os.path.join(bhv_video_dir, 'FilteredPupil.mat'))                     
    
    pTime = pupil_data['pTime'].tolist()                                               
    fPupil = pupil_data['fPupil'].tolist()
    sPupil = pupil_data['sPupil'].tolist()
    whisker = pupil_data['whisker'].tolist()
    faceM = pupil_data['faceM'].tolist()
    bodyM = pupil_data['bodyM'].tolist()
    nose = pupil_data['nose'].tolist()
    bTime = pupil_data['bTime'].tolist()

    def ensure_float64_list(data_in):
        """Helper to ensure data is a list of float64 arrays to prevent overflow."""
                                                                                       
        if isinstance(data_in, np.ndarray):
            if data_in.dtype == 'object':
                data_in = data_in.flatten()
                                                                                    
            elif data_in.ndim == 1 and np.issubdtype(data_in.dtype, np.number):
                 data_in = [data_in]
        
                                                   
                                                   
        return [np.array(x).astype(np.float64) for x in data_in]

    pTime = ensure_float64_list(pupil_data['pTime'])
    bTime = ensure_float64_list(pupil_data['bTime'])


                                            
    timeCheck1 = (SessionData_TrialStartTime[0]) - (pTime[0][0])
    timeCheck2 = (SessionData_TrialStartTime[0]) - (bTime[0][0])

    if (3590 < timeCheck1 < 3610) and (3590 < timeCheck2 < 3610):                   
        print('Behavioral and video timestamps are shifted by 1h. Adjusting video timestamps.')
        for iTrials in range(len(pTime)):
            pTime[iTrials] = pTime[iTrials] + 3600                            
            bTime[iTrials] = bTime[iTrials] + 3600
    elif timeCheck1 > 30 or timeCheck1 < -30 or timeCheck2 > 30 or timeCheck2 < -30:
        raise ValueError('Something wrong with timestamps in behavior and video data. Time difference > 30 seconds.')

    if any(bTrials > len(pTime)):
        warning_trials_removed = sum(bTrials > len(pTime))
        print(f'Warning: Insufficient trials in pupil data. Rejected last {warning_trials_removed} trial(s)')
        bTrials = bTrials[bTrials <= len(pTime)]                                        
        trialCnt = len(bTrials)

    opts_path = os.path.join(cPath, Animal, Rec, 'opts2.mat')
    opts = loadmat(opts_path)['opts']


                                   
    lickL = [[]] * trialCnt
    lickR = [[]] * trialCnt
    leverIn = np.full(trialCnt, np.nan)
    levGrabL = [[]] * trialCnt
    levGrabR = [[]] * trialCnt
    levReleaseL = [[]] * trialCnt
    levReleaseR = [[]] * trialCnt
    water = np.full(trialCnt, np.nan)
    stimGrab = np.full(trialCnt, np.nan)
    stimTime = np.full(trialCnt, np.nan)
    spoutTime = np.full(trialCnt, np.nan)
    spoutOutTime = np.full(trialCnt, np.nan)


    for iTrials in range(trialCnt):
        trial_data = bhv['RawEvents']['Trial'][iTrials]

        leverTimes_WaitForAnimal = []                                  
        for state_name in ['WaitForAnimal1', 'WaitForAnimal2', 'WaitForAnimal3']:
            if state_name in trial_data['States']:
                state_times = trial_data['States'][state_name]
                if state_times.ndim == 1:                                     
                    leverTimes_WaitForAnimal.append(state_times)
                else:                                          
                    for time_pair in state_times:
                        leverTimes_WaitForAnimal.append(time_pair)

        leverTimes_WaitForAnimal = np.concatenate(leverTimes_WaitForAnimal)
        WaitForCam_start_time = trial_data['States']['WaitForCam'][0]

        stimGrab_val = leverTimes_WaitForAnimal[np.where(leverTimes_WaitForAnimal == WaitForCam_start_time)[0]-1]                                                   
        stimGrab[iTrials] = stimGrab_val if stimGrab_val.size > 0 else np.nan                                   

        try:
            stimTime[iTrials] = trial_data['Events']['Wire3High'] - stimGrab[iTrials]                               
        except KeyError:
            stimTime[iTrials] = np.nan

        if 'MoveSpout' in trial_data['States']:
            spoutTime[iTrials] = trial_data['States']['MoveSpout'][0] - stimGrab[iTrials]

            if bhv['Rewarded'][iTrials]:
                spoutOutTime[iTrials] = trial_data['States']['Reward'][0] - stimGrab[iTrials]
            else:
                spoutOutTime[iTrials] = trial_data['States']['HardPunish'][0] - stimGrab[iTrials]
        else:
            spoutTime[iTrials] = np.nan
            spoutOutTime[iTrials] = np.nan

        if 'Port1In' in trial_data['Events']:            
            lickL_times = trial_data['Events']['Port1In']
            if not isinstance(lickL_times, list) and not isinstance(lickL_times, np.ndarray):
                lickL_times = [lickL_times]                                                  
            lickL_times = np.array(lickL_times)
            lickL_times = lickL_times[lickL_times >= trial_data['States']['MoveSpout'][0]]                                       
            lickL[iTrials] = lickL_times - stimGrab[iTrials]

        if 'Port3In' in trial_data['Events']:                  
            lickR_times = trial_data['Events']['Port3In']
                                                                                                    
            if not isinstance(lickR_times, list) and not isinstance(lickR_times, np.ndarray):
                lickR_times = [lickR_times]                                                  
            lickR_times = np.array(lickR_times)                                        
            lickR_times = lickR_times[lickR_times >= trial_data['States']['MoveSpout'][0]]
            lickR[iTrials] = lickR_times - stimGrab[iTrials]

        leverIn[iTrials] = np.min(trial_data['States']['Reset']) - stimGrab[iTrials]                                    

        if 'Wire2High' in trial_data['Events']:                  
            levGrabL[iTrials] = trial_data['Events']['Wire2High'] - stimGrab[iTrials]
        if 'Wire1High' in trial_data['Events']:                   
            levGrabR[iTrials] = trial_data['Events']['Wire1High'] - stimGrab[iTrials]

        if 'Wire2Low' in trial_data['Events']:                     
            levReleaseL[iTrials] = trial_data['Events']['Wire2Low'] - stimGrab[iTrials]
        if 'Wire1Low' in trial_data['Events']:                      
            levReleaseR[iTrials] = trial_data['Events']['Wire1Low'] - stimGrab[iTrials]

        if 'Reward' in trial_data['States'] and not np.isnan(trial_data['States']['Reward'][0]):                  
            water[iTrials] = trial_data['States']['Reward'][0] - stimGrab[iTrials]

    extracted_data = {                                               
        'sRate': sRate,
        'preStimDur': preStimDur,
        'postStimDur': postStimDur,
        'frames': frames,
        'trialDur': trialDur,
        'bhvDimCnt': bhvDimCnt,
        'dims': dims if 'dims' in locals() else None,                                           
        'trialCnt': trialCnt,
        'bTrials': bTrials,
        'bhv': bhv,
        'U':U.astype(np.float32),
        'Vc': Vc if 'Vc' in locals() else None,                                        
        'V1': V1.astype(np.float32),
        'V2': V2.astype(np.float32),
        'opts':opts,
        'data_DS': data_DS if 'data_DS' in locals() else None,                   
        'data_analog': data_analog if 'data_analog' in locals() else None,                      
        'pTime': pTime,
        'fPupil': fPupil,
        'sPupil': sPupil,
        'whisker': whisker,
        'faceM': faceM,
        'bodyM': bodyM,
        'nose': nose,
        'bTime': bTime,
        'stimTime': stimTime,
        'spoutTime': spoutTime,
        'spoutOutTime': spoutOutTime,
        'lickL': lickL,
        'lickR': lickR,
        'leverIn': leverIn,
        'levGrabL': levGrabL,
        'levGrabR': levGrabR,
        'levReleaseL': levReleaseL,
        'levReleaseR': levReleaseR,
        'water': water,
        'stimGrab': stimGrab
    }

    return extracted_data



def process_combined_data(extracted_data, dType='Widefield'):
    """
    Combines processing of imaging (Vc) and continuous behavior data, extracting variable length trials
    based on the maximal shared intervals of valid data intervals for both modalities,
    with CORRECTED trialTime_pupil calculation and separate indexing for timeseries.

    Args:
        extracted_data (dict): Output from extract_time_series_data function.
        dType (str, optional): Data type ('Widefield' or 'twoP'). Defaults to 'Widefield'.

    Returns:
        dict: A dictionary containing NumPy arrays of variable length trials for combined Vc and behavior data,
              using separate indexing for each timeseries based on maximal shared interval.
    """

    sRate = extracted_data['sRate']
    preStimDur = extracted_data['preStimDur']
    postStimDur = extracted_data['postStimDur']
    bhvDimCnt = extracted_data['bhvDimCnt']
    dims = extracted_data['dims'] if 'dims' in extracted_data else 0
    trialCnt = extracted_data['trialCnt']
    bTrials = extracted_data['bTrials'] - 1
    Vc = extracted_data['Vc']
    V1 = extracted_data['V1']
    V2 = extracted_data['V2']
    data_DS = extracted_data['data_DS']
    stimTime_time_series = extracted_data['stimTime']

    pTime = extracted_data['pTime']
    fPupil = extracted_data['fPupil']
    sPupil = extracted_data['sPupil']
    whisker = extracted_data['whisker']
    faceM = extracted_data['faceM']
    bodyM = extracted_data['bodyM']
    nose = extracted_data['nose']
    bTime = extracted_data['bTime']
    stimGrab = extracted_data['stimGrab']
    bhv = extracted_data['bhv']
    stimTime = extracted_data['stimTime']

    V1 = V1.reshape((-1, 205, bhvDimCnt))
    V2 = V2.reshape((-1, 205, bhvDimCnt))

    valid_bTrials_vidR = bTrials[bTrials < V1.shape[0]]
    valid_bTrials_moveR = bTrials[bTrials < V2.shape[0]]

    vidR = V1[valid_bTrials_vidR] if dType.lower() == 'widefield' and valid_bTrials_vidR.size > 0 else None
    moveR = V2[valid_bTrials_moveR] if dType.lower() == 'widefield' and valid_bTrials_moveR.size > 0 else None

    all_trials = []
    combined_Vc_aligned = []
    combined_vidR_aligned = []
    combined_moveR_aligned = []
    combined_DS_aligned = []
    combined_fPupil_time_series = []
    combined_slowPupil_time_series = []
    combined_whisker_time_series = []
    combined_faceM_time_series = []
    combined_bodyM_time_series = []
    combined_nose_time_series = []

    lickR_time_series = []                              
    lickL_time_series = []
    levGrabL_time_series = []
    levGrabR_time_series = []
    levReleaseL_time_series = []
    levReleaseR_time_series = []
    water_time_series = []


    shVal = sRate * 3                                                  

    for iTrials in range(trialCnt):
        stim_time_sec = stimTime_time_series[iTrials] if not np.isnan(stimTime_time_series[iTrials]) else 0
        stim_time_frames_shift = int(np.round(stim_time_sec * sRate))
        pivot_ind = int(shVal - stim_time_frames_shift)


                                                                                                
        vc_start_idx_in_vc_frame = np.array([], dtype=int)
        vc_end_idx_in_vc_frame = np.array([], dtype=int)
        pupil_start_idx_in_pupil_frame = np.array([], dtype=int)
        pupil_end_idx_in_pupil_frame = np.array([], dtype=int)

                                                                                                      
        lickR_trial_time_series = np.zeros(205)                                   
        lickL_trial_time_series = np.zeros(205)
        levGrabL_trial_time_series = np.zeros(205)
        levGrabR_trial_time_series = np.zeros(205)
        levReleaseL_trial_time_series = np.zeros(205)
        levReleaseR_trial_time_series = np.zeros(205)
        water_trial_time_series = np.zeros(205)


                                                                          
        if Vc is not None and Vc.ndim > 2 and Vc.shape[2] > iTrials and iTrials < len(pTime) and pTime[bTrials[iTrials]] is not None and len(pTime[bTrials[iTrials]]) > 0:


            trialOn = bhv['TrialStartTime'][iTrials] + (stimGrab[iTrials])
            trialTime_pupil = np.array(pTime[bTrials[iTrials]]) - trialOn                                   


            if trialTime_pupil.size > 0:
                start_idx_vc = -pivot_ind
                end_idx_vc = 205 - pivot_ind

                start_idx_pupil = int(trialTime_pupil[0] * sRate) if trialTime_pupil.size > 0 else 0                               
                end_idx_pupil = int(trialTime_pupil[-1] * sRate) if trialTime_pupil.size > 0 else 0                               


                start_index_frame = max(start_idx_vc, start_idx_pupil)                             
                end_index_frame = min(end_idx_vc, end_idx_pupil)                           


                vc_frame_times = np.arange(-pivot_ind, 205 - pivot_ind, dtype=int)
                pupil_frame_times = np.arange(start_idx_pupil, end_idx_pupil, dtype=int)
                vc_start_indices_in_vc_frame = np.where(vc_frame_times == start_index_frame)[0]                               
                vc_end_indices_in_vc_frame = np.where(vc_frame_times == end_index_frame)[0]                               

                pupil_start_indices_in_pupil_frame = np.where(pupil_frame_times == start_index_frame)[0]                                  
                pupil_end_indices_in_pupil_frame = np.where(pupil_frame_times == end_index_frame)[0]                                  

                if vc_start_indices_in_vc_frame.size > 0:                                                
                    vc_start_idx_in_vc_frame = vc_start_indices_in_vc_frame[0]
                if vc_end_indices_in_vc_frame.size == 0:
                    vc_end_idx_in_vc_frame = 205                                    
                elif vc_end_indices_in_vc_frame.size > 0:
                    vc_end_idx_in_vc_frame = vc_end_indices_in_vc_frame[0]


                if pupil_start_indices_in_pupil_frame.size > 0:
                    pupil_start_idx_in_pupil_frame = pupil_start_indices_in_pupil_frame[0]
                if pupil_end_indices_in_pupil_frame.size == 0:
                     if pupil_frame_times.size > 0:
                        pupil_end_idx_in_pupil_frame = len(pupil_frame_times) - 1                                             
                     else:
                        pupil_end_idx_in_pupil_frame = np.array([], dtype=int)                            
                elif pupil_end_indices_in_pupil_frame.size > 0:
                    pupil_end_idx_in_pupil_frame = pupil_end_indices_in_pupil_frame[0]



            if dType.lower() == 'widefield' and Vc is not None and Vc.ndim > 2 and Vc.shape[2] > iTrials:
                Vc_trial = Vc[:, vc_start_idx_in_vc_frame : vc_end_idx_in_vc_frame , iTrials]                                               
            else:
                Vc_trial = None
            if vidR is not None and vidR.ndim > 1 and vidR.shape[0] > iTrials:
                vidR_trial = vidR[iTrials, vc_start_idx_in_vc_frame : vc_end_idx_in_vc_frame, :]                                               
            else:
                vidR_trial = None
            if moveR is not None and moveR.ndim > 1 and moveR.shape[0] > iTrials:
                moveR_trial = moveR[iTrials, vc_start_idx_in_vc_frame : vc_end_idx_in_vc_frame, :]                                               
            else:
                moveR_trial = None
            if dType.lower() == 'twop' and data_DS is not None and data_DS.ndim > 2 and data_DS.shape[2] > iTrials:
                DS_trial = data_DS[:, vc_start_idx_in_vc_frame : vc_end_idx_in_vc_frame, iTrials]                                               
            else:
                DS_trial = None

            fPupil_trial = np.array(fPupil[bTrials[iTrials]][pupil_start_idx_in_pupil_frame : pupil_end_idx_in_pupil_frame ]) if pupil_start_idx_in_pupil_frame.size > 0 and pupil_end_idx_in_pupil_frame.size > 0 else np.array([])                                                     
            sPupil_trial = np.array(sPupil[bTrials[iTrials]][pupil_start_idx_in_pupil_frame : pupil_end_idx_in_pupil_frame ]) if pupil_start_idx_in_pupil_frame.size > 0 and pupil_end_idx_in_pupil_frame.size > 0 else np.array([])                                                     
            whisker_trial = np.array(whisker[bTrials[iTrials]][pupil_start_idx_in_pupil_frame : pupil_end_idx_in_pupil_frame ]) if pupil_start_idx_in_pupil_frame.size > 0 and pupil_end_idx_in_pupil_frame.size > 0 else np.array([])                                                     
            nose_trial = np.array(nose[bTrials[iTrials]][pupil_start_idx_in_pupil_frame : pupil_end_idx_in_pupil_frame ]) if pupil_start_idx_in_pupil_frame.size > 0 and pupil_end_idx_in_pupil_frame.size > 0 else np.array([])                                                     
            faceM_trial = np.array(faceM[bTrials[iTrials]][pupil_start_idx_in_pupil_frame : pupil_end_idx_in_pupil_frame ]) if pupil_start_idx_in_pupil_frame.size > 0 and pupil_end_idx_in_pupil_frame.size > 0 else np.array([])                                                     
                                               
                                                                                                                      
            bodyM_trial = np.array(bodyM[bTrials[iTrials]][pupil_start_idx_in_pupil_frame : pupil_end_idx_in_pupil_frame ]) if pupil_start_idx_in_pupil_frame.size > 0 and pupil_end_idx_in_pupil_frame.size > 0 else np.array([])                                                     


                                                                      
            trial_data = bhv['RawEvents']['Trial'][iTrials]


            if 'Port1In' in trial_data['Events']:            
                lickL_times = trial_data['Events']['Port1In']
                if not isinstance(lickL_times, list) and not isinstance(lickL_times, np.ndarray):
                    lickL_times = [lickL_times]
                lickL_times = np.array(lickL_times)
                lickL_times = lickL_times[lickL_times >= trial_data['States']['MoveSpout'][0]]
                for lick_t in lickL_times:
                                                                                                                 
                    frame_index = int(np.round(shVal + (-stimTime[iTrials] + lick_t - stimGrab[iTrials]) * sRate))           
                             
                    if 0 <= frame_index < 205:                                   
                         lickL_trial_time_series[frame_index] = 1


            if 'Port3In' in trial_data['Events']:                  
                lickR_times = trial_data['Events']['Port3In']
                if not isinstance(lickR_times, list) and not isinstance(lickR_times, np.ndarray):
                    lickR_times = [lickR_times]
                lickR_times = np.array(lickR_times)
                lickR_times = lickR_times[lickR_times >= trial_data['States']['MoveSpout'][0]]
                for lick_t in lickR_times:
                                                                                                                 
                    frame_index = int(np.round(shVal + (-stimTime[iTrials] + lick_t - stimGrab[iTrials]) * sRate))           
                    if 0 <= frame_index < 205:                                   
                        lickR_trial_time_series[frame_index] = 1


            if 'Wire2High' in trial_data['Events']:                  
                levGrabL_times = trial_data['Events']['Wire2High']
                if not isinstance(levGrabL_times, list) and not isinstance(levGrabL_times, np.ndarray):
                    levGrabL_times = [levGrabL_times]
                levGrabL_times = np.array(levGrabL_times)
                for grab_t in levGrabL_times:
                                                                                                                 
                    frame_index = int(np.round(shVal + (-stimTime[iTrials] + grab_t - stimGrab[iTrials]) * sRate))           
                    if 0 <= frame_index < 205:                                   
                        levGrabL_trial_time_series[frame_index] = 1

            if 'Wire1High' in trial_data['Events']:                   
                levGrabR_times = trial_data['Events']['Wire1High']
                if not isinstance(levGrabR_times, list) and not isinstance(levGrabR_times, np.ndarray):
                    levGrabR_times = [levGrabR_times]
                levGrabR_times = np.array(levGrabR_times)
                for grab_t in levGrabR_times:
                                                                                                                  
                     frame_index = int(np.round(shVal + (-stimTime[iTrials] + grab_t - stimGrab[iTrials]) * sRate))           
                     if 0 <= frame_index < 205:                                   
                        levGrabR_trial_time_series[frame_index] = 1

            if 'Wire2Low' in trial_data['Events']:                     
                levReleaseL_times = trial_data['Events']['Wire2Low']
                if not isinstance(levReleaseL_times, list) and not isinstance(levReleaseL_times, np.ndarray):
                    levReleaseL_times = [levReleaseL_times]
                levReleaseL_times = np.array(levReleaseL_times)
                for release_t in levReleaseL_times:
                                                                                                                    
                    frame_index = int(np.round(shVal + (-stimTime[iTrials] + release_t - stimGrab[iTrials]) * sRate))           
                    if 0 <= frame_index < 205:                                   
                        levReleaseL_trial_time_series[frame_index] = 1

            if 'Wire1Low' in trial_data['Events']:                      
                levReleaseR_times = trial_data['Events']['Wire1Low']
                if not isinstance(levReleaseR_times, list) and not isinstance(levReleaseR_times, np.ndarray):
                    levReleaseR_times = [levReleaseR_times]
                levReleaseR_times = np.array(levReleaseR_times)
                for release_t in levReleaseR_times:
                                                                                                                    
                    frame_index = int(np.round(shVal + (-stimTime[iTrials] + release_t - stimGrab[iTrials]) * sRate))           
                    if 0 <= frame_index < 205:                                   
                        levReleaseR_trial_time_series[frame_index] = 1

            if 'Reward' in trial_data['States'] and not np.isnan(trial_data['States']['Reward'][0]):                  
                water_t = trial_data['States']['Reward'][0] - stimGrab[iTrials]
                                                                                                                           
                water_frame_idx = int(np.round(shVal + (-stimTime[iTrials] + trial_data['States']['Reward'][0] - stimGrab[iTrials]) * sRate))           

                if 0 <= water_frame_idx < 205:                                   
                    water_trial_time_series[water_frame_idx] = 1




        else:
            continue


        if fPupil_trial.shape[0] < 205 or bodyM_trial.shape[0] < 205:
            print(f"Trial {iTrials}: REMOVED due to short pupil data")                   
                                                  
                                                                                        
                                                                                                                                                                                                                        
                                                                                                                             
                                                                                                                       
                                                                                                                                               
                                                                                                                                         
                                      
                                                                                                        
                                          
                                                                                                                
            continue                                                                  

                                                     
        combined_Vc_aligned.append(Vc_trial)
        all_trials.append(iTrials)

        combined_vidR_aligned.append(vidR_trial)
        combined_moveR_aligned.append(moveR_trial)
        combined_DS_aligned.append(DS_trial)
        combined_fPupil_time_series.append(fPupil_trial)
        combined_slowPupil_time_series.append(sPupil_trial)
        combined_whisker_time_series.append(whisker_trial)
        combined_faceM_time_series.append(faceM_trial)
        combined_bodyM_time_series.append(bodyM_trial)
        combined_nose_time_series.append(nose_trial)
        lickR_time_series.append(lickR_trial_time_series)                               
        lickL_time_series.append(lickL_trial_time_series)
        levGrabL_time_series.append(levGrabL_trial_time_series)
        levGrabR_time_series.append(levGrabR_trial_time_series)
        levReleaseL_time_series.append(levReleaseL_trial_time_series)
        levReleaseR_time_series.append(levReleaseR_trial_time_series)
        water_time_series.append(water_trial_time_series)




                                   
                                                                                                                                          
                                                                                                                                                            
    combined_data_np = {
        'fPupil_time_series': np.array(combined_fPupil_time_series, dtype=np.float32),
        'slowPupil_time_series': np.array(combined_slowPupil_time_series, dtype=np.float32),
        'whisker_time_series': np.array(combined_whisker_time_series, dtype=np.float32),
        'faceM_time_series': np.array(combined_faceM_time_series, dtype=np.float32),
        'bodyM_time_series': np.array(combined_bodyM_time_series, dtype=np.float32),
        'nose_time_series': np.array(combined_nose_time_series, dtype=np.float32),
        'lickR_time_series': np.array(lickR_time_series, dtype=np.float32),                                 
        'lickL_time_series': np.array(lickL_time_series, dtype=np.float32),
        'levGrabL_time_series': np.array(levGrabL_time_series, dtype=np.float32),
        'levGrabR_time_series': np.array(levGrabR_time_series, dtype=np.float32),
        'levReleaseL_time_series': np.array(levReleaseL_time_series, dtype=np.float32),
        'levReleaseR_time_series': np.array(levReleaseR_time_series, dtype=np.float32),
        'water_time_series': np.array(water_time_series, dtype=np.float32)
    }

    binary_keys = ['lickR_time_series', 'lickL_time_series', 'levGrabL_time_series', 'levGrabR_time_series', 'levReleaseL_time_series', 'levReleaseR_time_series', 'water_time_series']
    continuous_keys = ['fPupil_time_series', 'slowPupil_time_series', 'whisker_time_series', 'faceM_time_series', 'bodyM_time_series', 'nose_time_series']
    behavior_binary = [combined_data_np[item] for item in binary_keys if item in combined_data_np]
    behavior_cont = [combined_data_np[item] for item in continuous_keys if item in combined_data_np]

    return {
        'all_trials': np.array(all_trials),
        'Vc': np.array(combined_Vc_aligned, dtype=np.float32).transpose(0, 2, 1),
                                                         
        'vidR_aligned': np.array(combined_vidR_aligned, dtype=np.float32),
        'moveR_aligned': np.array(combined_moveR_aligned, dtype=np.float32),
        'behavior_binary': np.array(behavior_binary, dtype=np.float32).transpose(1, 2, 0),
        'behavior_cont': np.array(behavior_cont, dtype=np.float32).transpose(1, 2, 0),
        'binary_keys': binary_keys,
        'continuous_keys': continuous_keys
    }




def process_imaging_video_data(extracted_data, dType='Widefield'):
    """
    Processes imaging (Vc) and behavior video (V1, V2) data: aligns and reshapes them.

    Args:
        extracted_data (dict): Output from extract_time_series_data function.
        dType (str, optional): Data type ('Widefield' or 'twoP'). Defaults to 'Widefield'.

    Returns:
        dict: A dictionary containing aligned and reshaped Vc, vidR, moveR, and DS (if applicable).
    """

    sRate = extracted_data['sRate']
    preStimDur = extracted_data['preStimDur']
    postStimDur = extracted_data['postStimDur']
    bhvDimCnt = extracted_data['bhvDimCnt']
    dims = extracted_data['dims'] if 'dims' in extracted_data else 0                                       
    trialCnt = extracted_data['trialCnt']
    bTrials = extracted_data['bTrials'] - 1
    Vc = extracted_data['Vc']
    V1 = extracted_data['V1']
    V2 = extracted_data['V2']
    data_DS = extracted_data['data_DS']
    stimTime_time_series = extracted_data['stimTime']


    if dType.lower() == 'widefield':
        pass                                                                                

    V1 = V1.reshape((-1, 205, bhvDimCnt))                     
    V2 = V2.reshape((-1, 205, bhvDimCnt))                     

    valid_bTrials_vidR = bTrials[bTrials < V1.shape[0]]                                                            
    valid_bTrials_moveR = bTrials[bTrials < V2.shape[0]]                                                             


    vidR = V1[valid_bTrials_vidR] if dType.lower() == 'widefield' and valid_bTrials_vidR.size > 0 else None                                                
    moveR = V2[valid_bTrials_moveR] if dType.lower() == 'widefield' and valid_bTrials_moveR.size > 0 else None                                                


                         
    temp1 = np.full((dims,extracted_data['frames'],trialCnt), np.nan) if dims else None                                         
    temp2 = np.full((trialCnt,extracted_data['frames'],bhvDimCnt), np.nan)
    temp3 = np.full((trialCnt,extracted_data['frames'],bhvDimCnt), np.nan)
    temp4 = np.full((2,extracted_data['frames'],trialCnt), np.nan) if data_DS is not None else None                                    
    shVal = sRate * 3                                                                                                           


    for x in range(trialCnt):                                           
        try:
            stim_time_sec = stimTime_time_series[x] if not np.isnan(stimTime_time_series[x]) else 0                                                                                   
            stim_time_frames_shift = int(np.round(stim_time_sec * sRate))                                   
            start_frame_idx = int(shVal - stim_time_frames_shift) - int(preStimDur * sRate)                                                    
            end_frame_idx = int(shVal - stim_time_frames_shift) + int(postStimDur * sRate)
            print(start_frame_idx, end_frame_idx)
            if dType.lower() == 'widefield' and temp1 is not None:                                                 
                if Vc.shape[2] > x:                                
                    temp1[:,:,x] = Vc[:, start_frame_idx:end_frame_idx, x]                                      
            if vidR is not None and vidR.shape[1] > x:                                  
                temp2[x,:,:] = vidR[x,start_frame_idx:end_frame_idx,:]                                        
            if moveR is not None and moveR.shape[1] > x:                                   
                temp3[x,:,:] = moveR[x,start_frame_idx:end_frame_idx,:]                                         
            if dType.lower() == 'twop' and temp4 is not None:                                                 
                 if data_DS.shape[2] > x:                                     
                    temp4[:,:,x] = data_DS[:, start_frame_idx:end_frame_idx, x]

        except Exception as e:                                                        
            print(f'Could not align trial {bTrials[x]}. Error: {e}')


    Vc_aligned = np.reshape(np.transpose(temp1, (0, 2, 1)),(dims,-1)) if dType.lower() == 'widefield' and temp1 is not None else None                             
    vidR_aligned = np.reshape(np.transpose(temp2, (0, 1, 2)),(-1,bhvDimCnt))
    moveR_aligned = np.reshape(np.transpose(temp3, (0, 1, 2)),(-1,bhvDimCnt))
    DS_aligned = np.reshape(temp4,(2,-1)) if dType.lower() == 'twop' and temp4 is not None else None                                         

    return {
        'Vc_aligned': Vc_aligned,
        'vidR_aligned': vidR_aligned,
        'moveR_aligned': moveR_aligned,
        'DS_aligned': DS_aligned
    }





def process_continuous_bhv_data(extracted_data, dType='Widefield'):
    """
    Processes continuous behavior data (pupil, whisker, etc.) to create time series.

    Args:
        extracted_data (dict): Output from extract_time_series_data function.
        frames (int): Number of frames per trial.
        sRate (int): Sampling rate of imaging.

    Returns:
        dict: A dictionary containing time series data for continuous behaviors.
    """
    frames = extracted_data['frames']
    sRate = extracted_data['sRate']
    trialCnt = extracted_data['trialCnt']
    bTrials = extracted_data['bTrials'] - 1                                                

    pTime = extracted_data['pTime']
    fPupil = extracted_data['fPupil']
    sPupil = extracted_data['sPupil']
    whisker = extracted_data['whisker']
    faceM = extracted_data['faceM']
    bodyM = extracted_data['bodyM']
    nose = extracted_data['nose']
    bTime = extracted_data['bTime']
    stimGrab = extracted_data['stimGrab']
    preStimDur = extracted_data['preStimDur']
    bhv = extracted_data['bhv']


    fPupil_time_series = [np.full(frames, np.nan) for _ in range(trialCnt)]                                 
    slowPupil_time_series = [np.full(frames, np.nan) for _ in range(trialCnt)]                                   
    sPupil_time_series = sPupil                            
    whisker_time_series = [np.full(frames, np.nan) for _ in range(trialCnt)]
    faceM_time_series = [np.full(frames, np.nan) for _ in range(trialCnt)]
    bodyM_time_series = [np.full(frames, np.nan) for _ in range(trialCnt)]
    nose_time_series = [np.full(frames, np.nan) for _ in range(trialCnt)]
    bTime_time_series = bTime                      


    for iTrials in range(trialCnt):
                                                                   
        if iTrials < len(pTime) and pTime[bTrials[iTrials]] is not None and len(pTime[bTrials[iTrials]]) > 0:                                      
            trialOn = bhv['TrialStartTime'][iTrials] + (stimGrab[iTrials] - preStimDur)
            trialTime_pupil = np.array(pTime[bTrials[iTrials]]) - trialOn                                   
                                                                      
            idx_pupil_initial = (trialTime_pupil > 0)                                

            first_valid_indices = np.where(trialTime_pupil > 0)[0]
            if first_valid_indices.size > 0:
                start_index = first_valid_indices[0]
                idx_pupil = np.arange(start_index, min(start_index + frames, len(trialTime_pupil)))                                                     
            else:
                idx_pupil = np.array([], dtype=int)                                                   
                                                                                                 
                                                                                                                                                                                                                               
                                                                                                                                                         
            print(iTrials, idx_pupil[0], idx_pupil[-1])
            if trialTime_pupil.size > 0 and trialTime_pupil[0] > 0:                                                            
                print(f'Warning: Trial {bTrials[iTrials] + 1}: Missing behavioral video frames at trial onset for pupil data.')                                        
                fPupil_time_series[iTrials] = np.nan * np.ones(frames)
                slowPupil_time_series[iTrials] = np.nan * np.ones(frames)
                whisker_time_series[iTrials] = np.nan * np.ones(frames)
                nose_time_series[iTrials] = np.nan * np.ones(frames)
                faceM_time_series[iTrials] = np.nan * np.ones(frames)
            else:
                print(fPupil[bTrials[iTrials]].shape)                    
                fPupil_trial = np.array(fPupil[bTrials[iTrials]][idx_pupil]) if np.any(idx_pupil) else np.array([])
                fPupil_padded = np.nan * np.ones(frames)                                
                fPupil_padded[:len(fPupil_trial)] = fPupil_trial[:frames]                                                                                                                                                        

                fPupil_time_series[iTrials][:] = fPupil_padded


                sPupil_trial = np.array(sPupil[bTrials[iTrials]][idx_pupil]) if np.any(idx_pupil) else np.array([])
                sPupil_padded = np.nan * np.ones(frames)
                sPupil_padded[:len(sPupil_trial)] = sPupil_trial[:frames]
                slowPupil_time_series[iTrials][:] = sPupil_padded                                                             

                whisker_trial = np.array(whisker[bTrials[iTrials]][idx_pupil]) if np.any(idx_pupil) else np.array([])
                whisker_padded = np.nan * np.ones(frames)
                whisker_padded[:len(whisker_trial)] = whisker_trial[:frames]
                whisker_time_series[iTrials][:] = whisker_padded

                nose_trial = np.array(nose[bTrials[iTrials]][idx_pupil]) if np.any(idx_pupil) else np.array([])
                nose_padded = np.nan * np.ones(frames)
                nose_padded[:len(nose_trial)] = nose_trial[:frames]
                nose_time_series[iTrials][:] = nose_padded

                faceM_trial = np.array(faceM[bTrials[iTrials]][idx_pupil]) if np.any(idx_pupil) else np.array([])
                faceM_padded = np.nan * np.ones(frames)
                faceM_padded[:len(faceM_trial)] = faceM_trial[:frames]
                faceM_time_series[iTrials][:] = faceM_padded


        else:
             fPupil_time_series[iTrials] = np.nan * np.ones(frames)
             slowPupil_time_series[iTrials] = np.nan * np.ones(frames)
             whisker_time_series[iTrials] = np.nan * np.ones(frames)
             nose_time_series[iTrials] = np.nan * np.ones(frames)
             faceM_time_series[iTrials] = np.nan * np.ones(frames)


        if iTrials < len(bTime) and bTime[bTrials[iTrials]] is not None and len(bTime[bTrials[iTrials]]) > 0:                                      
            bhvFrameRate_body = round(1/np.mean(np.diff(bTime[bTrials[iTrials]]))) if len(bTime[bTrials[iTrials]]) > 1 else sRate                                                              
            trialOn = bhv['TrialStartTime'][iTrials] + (stimGrab[iTrials] - preStimDur)
            trialTime_body = np.array(bTime[bTrials[iTrials]]) - trialOn                                   
            idx_body = (trialTime_body < extracted_data['trialDur'] + 1/sRate)  &  (trialTime_body > 0)                                                                                            

            first_valid_indices = np.where(trialTime_body > 0)[0]
            if first_valid_indices.size > 0:
                start_index = first_valid_indices[0]
                idx_body = np.arange(start_index, min(start_index + frames, len(trialTime_pupil)))                                                     
            else:
                idx_body = np.array([], dtype=int)                                            

            trialTime_body_valid = trialTime_body[idx_body]
                                                                                                                                       
            idx_valid_frames_body = np.arange(frames)[(np.arange(frames)/sRate) < trialTime_body_valid.max()]                                              


            bodyM_trial = np.array(bodyM[bTrials[iTrials]][idx_body]) if np.any(idx_body) else np.array([])
            bodyM_padded = np.nan * np.ones(frames)
            bodyM_padded[:len(bodyM_trial)] = bodyM_trial[:frames]                                                                     
            bodyM_time_series[iTrials][:] = bodyM_padded

        else:
             bodyM_time_series[iTrials] = np.nan * np.ones(frames)


    return {
        'fPupil_time_series': fPupil_time_series,
        'slowPupil_time_series': slowPupil_time_series,
        'sPupil_time_series': sPupil_time_series,
        'whisker_time_series': whisker_time_series,
        'faceM_time_series': faceM_time_series,
        'bodyM_time_series': bodyM_time_series,
        'nose_time_series': nose_time_series,
        'bTime_time_series': bTime_time_series
    }


def process_event_timeseries_data(extracted_data, dType='Widefield'):
    """
    Processes event-based behavior data to create time series (currently just returns existing timeseries).
    Can be extended for further processing like smoothing or alignment if needed.

    Args:
        extracted_data (dict): Output from extract_time_series_data function.

    Returns:
        dict: A dictionary containing time series data for event-based behaviors.
    """
    frames = extracted_data['frames']
    trialCnt = extracted_data['trialCnt']
    stimTime = extracted_data['stimTime']
    spoutTime = extracted_data['spoutTime']
    spoutOutTime = extracted_data['spoutOutTime']

    stimTime_time_series = extracted_data['stimTime']                                                  
    spoutTime_time_series = extracted_data['spoutTime']
    spoutOutTime_time_series = extracted_data['spoutOutTime']
    lickR_time_series = [np.zeros(frames) for _ in range(trialCnt)]                            
    lickL_time_series = [np.zeros(frames) for _ in range(trialCnt)]
    levGrabL_time_series = [np.zeros(frames) for _ in range(trialCnt)]
    levGrabR_time_series = [np.zeros(frames) for _ in range(trialCnt)]
    levReleaseL_time_series = [np.zeros(frames) for _ in range(trialCnt)]
    levReleaseR_time_series = [np.zeros(frames) for _ in range(trialCnt)]
    water_time_series = [np.zeros(frames) for _ in range(trialCnt)]
    stimGrab = extracted_data['stimGrab']
    sRate = extracted_data['sRate']
    preStimDur = extracted_data['preStimDur']
    bhv = extracted_data['bhv']

    for iTrials in range(trialCnt):

        trial_data = bhv['RawEvents']['Trial'][iTrials]

        if 'Port1In' in trial_data['Events']:            
            lickL_times = trial_data['Events']['Port1In']
            if not isinstance(lickL_times, list) and not isinstance(lickL_times, np.ndarray):
                lickL_times = [lickL_times]                                                  
            print(lickL_times, trial_data['States']['MoveSpout'][0])
            lickL_times = np.array(lickL_times)
            lickL_times = lickL_times[lickL_times >= trial_data['States']['MoveSpout'][0]]                                       
            for lick_t in lickL_times:
                lick_frame_idx = int(np.round((preStimDur + (lick_t - stimGrab[iTrials])) * sRate))
                if 0 <= lick_frame_idx < frames:
                     lickL_time_series[iTrials][lick_frame_idx] = 1


        if 'Port3In' in trial_data['Events']:                  
            lickR_times = trial_data['Events']['Port3In']
                                                                                                    
            if not isinstance(lickR_times, list) and not isinstance(lickR_times, np.ndarray):
                lickR_times = [lickR_times]                                                  
            lickR_times = np.array(lickR_times)                                        
            lickR_times = lickR_times[lickR_times >= trial_data['States']['MoveSpout'][0]]
            for lick_t in lickR_times:
                lick_frame_idx = int(np.round((preStimDur + (lick_t - stimGrab[iTrials])) * sRate))
                if 0 <= lick_frame_idx < frames:
                    lickR_time_series[iTrials][lick_frame_idx] = 1


        if 'Wire2High' in trial_data['Events']:                  
            levGrabL_times = trial_data['Events']['Wire2High']
                                               
            if not isinstance(levGrabL_times, list) and not isinstance(levGrabL_times, np.ndarray):
                levGrabL_times = [levGrabL_times]
            levGrabL_times = np.array(levGrabL_times)

            for grab_t in levGrabL_times:
                grab_frame_idx = int(np.round((preStimDur + (grab_t - stimGrab[iTrials])) * sRate))
                if 0 <= grab_frame_idx < frames:
                    levGrabL_time_series[iTrials][grab_frame_idx] = 1

        if 'Wire1High' in trial_data['Events']:                   
            levGrabR_times = trial_data['Events']['Wire1High']
            if not isinstance(levGrabR_times, list) and not isinstance(levGrabR_times, np.ndarray):
                levGrabR_times = [levGrabR_times]
            levGrabR_times = np.array(levGrabR_times)
            for grab_t in levGrabR_times:
                 grab_frame_idx = int(np.round((preStimDur + (grab_t - stimGrab[iTrials])) * sRate))
                 if 0 <= grab_frame_idx < frames:
                    levGrabR_time_series[iTrials][grab_frame_idx] = 1

        if 'Wire2Low' in trial_data['Events']:                     
            levReleaseL_times = trial_data['Events']['Wire2Low']
            if not isinstance(levReleaseL_times, list) and not isinstance(levReleaseL_times, np.ndarray):
                levReleaseL_times = [levReleaseL_times]
            levReleaseL_times = np.array(levReleaseL_times)
            for release_t in levReleaseL_times:
                release_frame_idx = int(np.round((preStimDur + (release_t - stimGrab[iTrials])) * sRate))
                if 0 <= release_frame_idx < frames:
                    levReleaseL_time_series[iTrials][release_frame_idx] = 1

        if 'Wire1Low' in trial_data['Events']:                      
            levReleaseR_times = trial_data['Events']['Wire1Low']
            if not isinstance(levReleaseR_times, list) and not isinstance(levReleaseR_times, np.ndarray):
                levReleaseR_times = [levReleaseR_times]
            levReleaseR_times = np.array(levReleaseR_times)
            for release_t in levReleaseR_times:
                release_frame_idx = int(np.round((preStimDur + (release_t - stimGrab[iTrials])) * sRate))
                if 0 <= release_frame_idx < frames:
                    levReleaseR_time_series[iTrials][release_frame_idx] = 1

        if 'Reward' in trial_data['States'] and not np.isnan(trial_data['States']['Reward'][0]):                  
            water_t = trial_data['States']['Reward'][0] - stimGrab[iTrials]
            water_frame_idx = int(np.round((preStimDur + water_t) * sRate))
            if 0 <= water_frame_idx < frames:
                water_time_series[iTrials][water_frame_idx] = 1

    return {
        'lickR_time_series': lickR_time_series,
        'lickL_time_series': lickL_time_series,
        'levGrabL_time_series': levGrabL_time_series,
        'levGrabR_time_series': levGrabR_time_series,
        'levReleaseL_time_series': levReleaseL_time_series,
        'levReleaseR_time_series': levReleaseR_time_series,
        'water_time_series': water_time_series,
        'stimTime_time_series': stimTime_time_series,
        'spoutTime_time_series': spoutTime_time_series,
        'spoutOutTime_time_series': spoutOutTime_time_series
    }













