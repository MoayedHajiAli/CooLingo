import os
import subprocess
from os.path import join
from tqdm import tqdm
import numpy as np
import torch
from collections import OrderedDict
import librosa
from skimage.io import imread
import cv2
import scipy.io as sio
import argparse
import yaml
import albumentations as A
import albumentations.pytorch
from pathlib import Path

from src.modules.LiveSpeechPortraits.options.test_audio2feature_options import TestOptions as FeatureOptions
from src.modules.LiveSpeechPortraits.options.test_audio2headpose_options import TestOptions as HeadposeOptions
from src.modules.LiveSpeechPortraits.options.test_feature2face_options import TestOptions as RenderOptions

from src.modules.LiveSpeechPortraits.datasets import create_dataset
from src.modules.LiveSpeechPortraits.models import create_model
from src.modules.LiveSpeechPortraits.models.networks import APC_encoder
import src.modules.LiveSpeechPortraits.util.util as util
from src.modules.LiveSpeechPortraits.util.visualizer import Visualizer
from src.modules.LiveSpeechPortraits.funcs import utils
from src.modules.LiveSpeechPortraits.funcs import audio_funcs
import gdown
import zipfile

import warnings
warnings.filterwarnings("ignore")


class LiveSpeechPortraits:
    ckpt_g_id = {'May':'1E_chAFvsX6Q9YKA8J8lFy4XU4YRSLvlu',
                'APC_epoch_160.model':'1uUU6iZ8CdgsCk3JAG6V7BhJXnfWpaQ7a'}
    
    def __init__(self, id='May', apc_model_name='APC_epoch_160.model'):
        self.device = torch.device("cuda")
        
        with open(join('src/modules/LiveSpeechPortraits/config/', id + '.yaml')) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        
        data_root = 'pretrained_models/LiveSpeechPortraits/data/'
        
        # download chpt if not exitsts
        if not os.path.exists(join(data_root, id)):
            zip_file_path = join(data_root, f'{id}.zip')
            gdown.download(id=self.ckpt_g_id[id], output=zip_file_path)
            
            # unzip downloaded file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(data_root)
        
        if not os.path.exists(os.path.join(data_root, apc_model_name)):
            gdown.download(id=self.ckpt_g_id[apc_model_name], output=data_root)
            
            
            
        data_root = join(data_root, id)
       

        ############################ Hyper Parameters #############################
        self.mouth_indices = np.concatenate([np.arange(4, 11), np.arange(46, 64)])
        eye_brow_indices = [27, 65, 28, 68, 29, 67, 30, 66, 31, 72, 32, 69, 33, 70, 34, 71]
        self.eye_brow_indices = np.array(eye_brow_indices, np.int32)    
    

    
        ############################ Pre-defined Data #############################
        self.mean_pts3d = np.load(join(data_root, 'mean_pts3d.npy'))  
        fit_data = np.load(config['dataset_params']['fit_data_path'])
        pts3d = np.load(config['dataset_params']['pts3d_path']) - self.mean_pts3d
        trans = fit_data['trans'][:,:,0].astype(np.float32)
        self.mean_translation = trans.mean(axis=0)
        self.candidate_eye_brow = pts3d[10:, eye_brow_indices]
        self.std_mean_pts3d = np.load(config['dataset_params']['pts3d_path']).mean(axis=0)
        
        # candidates images    
        img_candidates = []
        for j in range(4):
            output = imread(join(data_root, 'candidates', f'normalized_full_{j}.jpg'))
            output = A.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 
                                                              'std':(0.5,0.5,0.5)})(image=output)['image']
            img_candidates.append(output)
        img_candidates = torch.cat(img_candidates).unsqueeze(0).to(self.device) 

        # shoulders
        self.shoulders = np.load(join(data_root, 'normalized_shoulder_points.npy'))
        self.shoulder3D = np.load(join(data_root, 'shoulder_points3D.npy'))[1]
        self.ref_trans = trans[1]    

        # camera matrix, we always use training set intrinsic parameters.
        self.camera = utils.camera() 
        self.camera_intrinsic = np.load(join(data_root, 'camera_intrinsic.npy')).astype(np.float32)
        self.APC_feat_database = np.load(join(data_root, 'APC_feature_base.npy'))

        # load reconstruction data
        self.scale = sio.loadmat(join(data_root, 'id_scale.mat'))['scale'][0,0]
        # Audio2Mel_torch = audio_funcs.Audio2Mel(n_fft=512, hop_length=int(16000/120), win_length=int(16000/60), sampling_rate=16000, 
        #                                         n_mel_channels=80, mel_fmin=90, mel_fmax=7600.0).to(device)



        ########################### Experiment Settings ###########################
        #### user config
        self.use_LLE = config['model_params']['APC']['use_LLE']
        self.Knear = config['model_params']['APC']['Knear']
        self.LLE_percent = config['model_params']['APC']['LLE_percent']
        self.headpose_sigma = config['model_params']['Headpose']['sigma']
        self.Feat_smooth_sigma = config['model_params']['Audio2Mouth']['smooth']
        self.Head_smooth_sigma = config['model_params']['Headpose']['smooth']
        self.Feat_center_smooth_sigma, Head_center_smooth_sigma = 0, 0
        self.AMP_method = config['model_params']['Audio2Mouth']['AMP'][0]
        self.Feat_AMPs = config['model_params']['Audio2Mouth']['AMP'][1:]
        self.rot_AMP, self.trans_AMP = config['model_params']['Headpose']['AMP']
        self.shoulder_AMP = config['model_params']['Headpose']['shoulder_AMP']
        self.save_feature_maps = config['model_params']['Image2Image']['save_input']

        #### common settings
        self.Featopt = FeatureOptions().parse() 
        self.Headopt = HeadposeOptions().parse()
        Renderopt = RenderOptions().parse()
        self.Featopt.load_epoch = config['model_params']['Audio2Mouth']['ckp_path']
        self.Headopt.load_epoch = config['model_params']['Headpose']['ckp_path']
        Renderopt.dataroot = config['dataset_params']['root']
        Renderopt.load_epoch = config['model_params']['Image2Image']['ckp_path']
        Renderopt.size = config['model_params']['Image2Image']['size']
        ## GPU or CPU
        if self.device == 'cpu':
            self.Featopt.gpu_ids = self.Headopt.gpu_ids = Renderopt.gpu_ids = []



        ############################# Load Models #################################
        print('---------- Loading Model: APC-------------')
        self.APC_model = APC_encoder(config['model_params']['APC']['mel_dim'],
                                config['model_params']['APC']['hidden_size'],
                                config['model_params']['APC']['num_layers'],
                                config['model_params']['APC']['residual'])
        self.APC_model.load_state_dict(torch.load(config['model_params']['APC']['ckp_path']), strict=False)
        self.APC_model.to(self.device) 
        self.APC_model.eval()
        
        print('---------- Loading Model: {} -------------'.format(self.Featopt.task))
        self.Audio2Feature = create_model(self.Featopt)   
        self.Audio2Feature.setup(self.Featopt)  
        self.Audio2Feature.eval()    
        
        
        print('---------- Loading Model: {} -------------'.format(self.Headopt.task))
        self.Audio2Headpose = create_model(self.Headopt)    
        self.Audio2Headpose.setup(self.Headopt)
        self.Audio2Headpose.eval()              
        if self.Headopt.feature_decoder == 'WaveNet':
            if self.device == 'cuda':
                self.Headopt.A2H_receptive_field = self.Audio2Headpose.Audio2Headpose.module.WaveNet.receptive_field
            else:
                self.Headopt.A2H_receptive_field = self.Audio2Headpose.Audio2Headpose.WaveNet.receptive_field
                
        print('---------- Loading Model: {} -------------'.format(Renderopt.task))
        self.facedataset = create_dataset(Renderopt) 
        self.Feature2Face = create_model(Renderopt)
        self.Feature2Face.setup(Renderopt)   
        self.Feature2Face.eval()
        self.visualizer = Visualizer(Renderopt)


        
        
    def generate_protrait(self, driving_audio, sr=16000, vid_res=(512, 512), fps=60):
        
        h, w, FPS = vid_res[0], vid_res[1], fps
        ############################## Inference ##################################
        # read audio
        audio, _ = librosa.load(driving_audio, sr=sr)
        total_frames = np.int32(audio.shape[0] / sr * FPS) 


        #### 1. compute APC features   
        print('1. Computing APC features...')                    
        mel80 = utils.compute_mel_one_sequence(audio, device=self.device)
        mel_nframe = mel80.shape[0]
        with torch.no_grad():
            length = torch.Tensor([mel_nframe])
            mel80_torch = torch.from_numpy(mel80.astype(np.float32)).to(device).unsqueeze(0)
            hidden_reps = self.APC_model.forward(mel80_torch, length)[0]   # [mel_nframe, 512]
            hidden_reps = hidden_reps.cpu().numpy()
        audio_feats = hidden_reps


        #### 2. manifold projection
        if self.use_LLE:
            print('2. Manifold projection...')
            ind = utils.KNN_with_torch(audio_feats, self.APC_feat_database, K=self.Knear)
            weights, feat_fuse = utils.compute_LLE_projection_all_frame(audio_feats, self.APC_feat_database, ind, audio_feats.shape[0])
            audio_feats = audio_feats * (1-self.LLE_percent) + feat_fuse * self.LLE_percent


        #### 3. Audio2Mouth
        print('3. Audio2Mouth inference...')
        pred_Feat = self.Audio2Feature.generate_sequences(audio_feats, sr, FPS, fill_zero=True, opt=self.Featopt)


        #### 4. Audio2Headpose
        print('4. Headpose inference...')
        # set history headposes as zero
        pre_headpose = np.zeros(self.Headopt.A2H_wavenet_input_channels, np.float32)
        pred_Head = self.Audio2Headpose.generate_sequences(audio_feats, pre_headpose, fill_zero=True, sigma_scale=0.3, opt=self.Headopt)


        #### 5. Post-Processing 
        print('5. Post-processing...')
        nframe = min(pred_Feat.shape[0], pred_Head.shape[0])
        pred_pts3d = np.zeros([nframe, 73, 3])
        pred_pts3d[:, self.mouth_indices] = pred_Feat.reshape(-1, 25, 3)[:nframe]

        ## mouth
        pred_pts3d = utils.landmark_smooth_3d(pred_pts3d, self.Feat_smooth_sigma, area='only_mouth')
        pred_pts3d = utils.mouth_pts_AMP(pred_pts3d, True, self.AMP_method, self.Feat_AMPs)
        pred_pts3d = pred_pts3d + self.mean_pts3d
        pred_pts3d = utils.solve_intersect_mouth(pred_pts3d)  # solve intersect lips if exist

        ## headpose
        pred_Head[:, 0:3] *= self.rot_AMP
        pred_Head[:, 3:6] *= self.trans_AMP
        pred_headpose = utils.headpose_smooth(pred_Head[:,:6], self.Head_smooth_sigma).astype(np.float32)
        pred_headpose[:, 3:] += self.mean_translation
        pred_headpose[:, 0] += 180

        ## compute projected landmarks
        pred_landmarks = np.zeros([nframe, 73, 2], dtype=np.float32)
        final_pts3d = np.zeros([nframe, 73, 3], dtype=np.float32)
        final_pts3d[:] = self.std_mean_pts3d.copy()
        final_pts3d[:, 46:64] = pred_pts3d[:nframe, 46:64]
        for k in tqdm(range(nframe)):
            ind = k % self.candidate_eye_brow.shape[0]
            final_pts3d[k, self.eye_brow_indices] = self.candidate_eye_brow[ind] + self.mean_pts3d[self.eye_brow_indices]
            pred_landmarks[k], _, _ = utils.project_landmarks(self.camera_intrinsic, self.camera.relative_rotation, 
                                                              self.camera.relative_translation, self.scale, 
                                                              pred_headpose[k], final_pts3d[k]) 

        ## Upper Body Motion
        pred_shoulders = np.zeros([nframe, 18, 2], dtype=np.float32)
        pred_shoulders3D = np.zeros([nframe, 18, 3], dtype=np.float32)
        for k in range(nframe):
            diff_trans = pred_headpose[k][3:] - self.ref_trans
            pred_shoulders3D[k] = self.shoulder3D + diff_trans * self.shoulder_AMP
            # project
            project = self.camera_intrinsic.dot(pred_shoulders3D[k].T)
            project[:2, :] /= project[2, :]  # divide z
            pred_shoulders[k] = project[:2, :].T


        #### 6. Image2Image translation & Save resuls
        print('6. Image2Image translation & Saving results...')
        for ind in tqdm(range(0, nframe), desc='Image2Image translation inference'):
            # feature_map: [input_nc, h, w]
            current_pred_feature_map = self.facedataset.dataset.get_data_test_mode(pred_landmarks[ind], 
                                                                              pred_shoulders[ind], 
                                                                              self.facedataset.dataset.image_pad)
            input_feature_maps = current_pred_feature_map.unsqueeze(0).to(device)
            pred_fake = self.Feature2Face.inference(input_feature_maps, img_candidates) 
            # save results
            visual_list = [('pred', util.tensor2im(pred_fake[0]))]
            if self.save_feature_maps:
                visual_list += [('input', np.uint8(current_pred_feature_map[0].cpu().numpy() * 255))]
            visuals = OrderedDict(visual_list)
            self.visualizer.save_images(save_root, visuals, str(ind+1))


        ## make videos
        # generate corresponding audio, reused for all results
        tmp_audio_path = join(save_root, 'tmp.wav')
        tmp_audio_clip = audio[ : np.int32(nframe * sr / FPS)]
        librosa.output.write_wav(tmp_audio_path, tmp_audio_clip, sr)


        final_path = join(save_root, audio_name + '.avi')
        write_video_with_audio(tmp_audio_path, final_path, 'pred_')
        feature_maps_path = join(save_root, audio_name + '_feature_maps.avi')
        write_video_with_audio(tmp_audio_path, feature_maps_path, 'input_')

        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        
        if not opt.save_intermediates:
            _img_paths = list(map(lambda x:str(x), list(Path(save_root).glob('*.jpg'))))
            for i in tqdm(range(len(_img_paths)), desc='deleting intermediate images'):
                os.remove(_img_paths[i])

        print('Finish!')