import torch
from torch.utils.data import Dataset, DataLoader

#Pixel level classification
from spectral import imshow, view_cube, save_rgb
import spectral.io.envi as envi
import numpy as np
import json
import os
from PIL import Image
import numpy as np
from scipy.signal import savgol_filter

def preprocess(input_data):
    data_preprocess = np.zeros_like(input_data)
    data_preprocess = savgol_filter(input_data, 15, 2)
    data_preprocess = np.gradient(data_preprocess, axis = 2)
    data_preprocess = (data_preprocess - np.mean(data_preprocess))/np.std(data_preprocess)
    
    return data_preprocess
def mask2rgb(mask):
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    
    for i in np.unique(mask):
            rgb[mask==i] = LABEL_TO_COLOR[i]
            
    return rgb

def rgb2mask(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))
    for k,v in LABEL_TO_COLOR.items():
        mask[np.all(rgb==v, axis=2)] = k
        
    return mask

def createPixelData(data_filepath, mask_filepath):
    LABEL_TO_COLOR =  {0:[0,0,0], 1:[255,0,0], 2:[0,255,0], 3:[0,0,255], 4:[255,255,0], 5:[255,0,255], 6:[0,255,255], 7: [255,255,128], 8:[255,128,255], 9:[128,255,255]}

    filenames = os.listdir(data_filepath)
    hsi_pixel_data = []
    hsi_pixel_label = []

    for i in range(len(filenames)):
        filename = filenames[i]
        maskname = filename+"_mask.png"
        print(f"computing for {filename}")
        print(f"computing for mask {maskname}")
        dark_ref = envi.open(data_filepath + '/' + filename + '/capture/DARKREF_' + filename + '.hdr', data_filepath + '/' + filename + '/capture/DARKREF_' + filename + '.raw')
        white_ref = envi.open(data_filepath + '/' + filename + '/capture/WHITEREF_' + filename + '.hdr', data_filepath + '/' + filename + '/capture/WHITEREF_' + filename + '.raw')
        data_ref = envi.open(data_filepath + '/' + filename + '/capture/' + filename + '.hdr', data_filepath + '/' + filename + '/capture/' + filename + '.raw')
        
        white_nparr = np.array(white_ref.load())
        dark_nparr = np.array(dark_ref.load())
        data_nparr = np.array(data_ref.load())
        corrected_nparr = np.divide(
            np.subtract(data_nparr, np.mean(dark_nparr, axis = 0)),
            np.subtract(np.mean(white_nparr, axis = 0), np.mean(dark_nparr, axis = 0)))
        
        if corrected_nparr.shape[0] != 640:
            #print(corrected_nparr.shape[0])
            corrected_nparr = np.concatenate((corrected_nparr,corrected_nparr[-1].reshape(1,640,224)), axis=0)
        
        corrected_nparr = preprocess(corrected_nparr[:,:,8:208])
        print(f"corrected_nparr shape {corrected_nparr.shape}")
        
        img = Image.open(mask_filepath + "/" + maskname)
        mask = np.array(img)
        print(f"mask shape {mask.shape}")
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j][0] == 255 and mask[i][j][1] == 0 and mask[i][j][2] == 0:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[0])
                elif mask[i][j][0] == 0 and mask[i][j][1] == 255 and mask[i][j][2] == 0:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[1])
                elif mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 255:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[2])
                elif mask[i][j][0] == 255 and mask[i][j][1] == 255 and mask[i][j][2] == 0:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[3])
                elif mask[i][j][0] == 255 and mask[i][j][1] == 0 and mask[i][j][2] == 255:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[4])
                elif mask[i][j][0] == 0 and mask[i][j][1] == 255 and mask[i][j][2] == 255:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[5])
                elif mask[i][j][0] == 128 and mask[i][j][1] == 255 and mask[i][j][2] == 128:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[6])
                elif mask[i][j][0] == 128 and mask[i][j][1] == 128 and mask[i][j][2] == 255:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[7])
                elif mask[i][j][0] == 255 and mask[i][j][1] == 128 and mask[i][j][2] == 128:
                    hsi_pixel_data.append(corrected_nparr[i][j])
                    hsi_pixel_label.append(np.eye(9)[8])

    return np.array(hsi_pixel_data), np.array(hsi_pixel_label)

class makeDataset(Dataset):
    def __init__(self, X, Y): 
        self.x = X
        self.y = Y
        #self.i = index
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        feature = self.x[index, :]
        label = self.y[index, :]
        #i = self.i[index]
        
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return {
            'features': feature,
            'labels' : label,
            #'index' : i
        }