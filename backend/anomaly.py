import sys, os
import argparse
import torch
import glob
import numpy as np
import subprocess
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from models.model import generate_model
from learner import Learner
from PIL import Image, ImageFilter, ImageOps, ImageChops
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import shutil

parser = argparse.ArgumentParser(description='Video Anomaly Detection')
parser.add_argument('--n', default='', type=str, help='file name')
parser.add_argument('--output_video', default='', type=str, help='output video path')
parser.add_argument('--output_graph', default='', type=str, help='output graph path')
args = parser.parse_args()

file_name = args.n
if not os.path.isdir(args.n[:-4]):
    os.mkdir(args.n[:-4])

save_name = './' + args.n[:-4] + '/%05d.jpg'

os.system('ffmpeg -i %s -r 25 -q:v 2 -vf scale=320:240 %s' % (file_name, save_name))

import subprocess

def get_video_fps(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        fps_str = result.stdout.strip()
        return eval(fps_str)
    else:
        return None

fps = get_video_fps(file_name)
try:
    import accimage
except ImportError:
    accimage = None

class ToTensor(object):

    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass
        

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass

#############################################################
#                        MAIN CODE                          #
#############################################################

model = generate_model()  # feature extractor
classifier = Learner().cuda()  # classifier

checkpoint = torch.load('./weight/RGB_Kinetics_16f.pth')
model.load_state_dict(checkpoint['state_dict'])
checkpoint = torch.load('./weight/ckpt.pth')
classifier.load_state_dict(checkpoint['net'])

model.eval()
classifier.eval()

path = args.n[:-4] + '/*'
save_path = args.n[:-4] + '_result'
img = glob.glob(path)
img.sort()

segment = len(img) // 16
x_value = [i for i in range(segment)]

inputs = torch.Tensor(1, 3, 16, 240, 320)
x_time = [jj for jj in range(len(img))]
y_pred = [0] * 16

for num, i in enumerate(img):
    if num < 16:
        inputs[:,:,num,:,:] = ToTensor(1)(Image.open(i))
        cv_img = cv2.imread(i)
        print(cv_img.shape)
        h,w,_ =cv_img.shape
        cv_img = cv2.putText(cv_img, 'Pred: 0.0', (5,30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
    else:
        inputs[:,:,:15,:,:] = inputs[:,:,1:,:,:]
        inputs[:,:,15,:,:] = ToTensor(1)(Image.open(i))
        inputs = inputs.cuda()
        start = time.time()
        output, feature = model(inputs)
        feature = F.normalize(feature, p=2, dim=1)
        out = classifier(feature)
        y_pred.append(out.item())
        end = time.time()
        out_str = str(out.item())[:5]
        print(len(x_value)/len(y_pred))
                
        cv_img = cv2.imread(i)
        cv_img = cv2.putText(cv_img, 'Pred: '+out_str, (5,30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
        if out.item() > 0.4:
            cv_img = cv2.rectangle(cv_img,(0,0),(w,h), (0,0,255), 3)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    path = './'+save_path+'/'+os.path.basename(i)
    cv2.imwrite(path, cv_img)

result_dir = os.path.splitext(args.n)[0] + '_anomaly_result'
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
os.system('ffmpeg -i "%s" "%s"' % (save_path + '/%05d.jpg', args.output_video))

# Find the indices of the 5 most drastic spikes
spike_indices = []
for i in range(1, len(y_pred)-1):
    if y_pred[i] > max(y_pred[i-1], y_pred[i+1]):
        spike_indices.append(i)
spike_indices.sort(key=lambda x: y_pred[x], reverse=True)
spike_indices = spike_indices[:5]

# Save anomaly frames
anomaly_frames_dir = os.path.join(result_dir, '../anomaly_frames')
os.makedirs(anomaly_frames_dir, exist_ok=True)

for idx in spike_indices:
    src_path = os.path.join(save_path, f'{idx+16:05d}.jpg')
    dst_path = os.path.join(anomaly_frames_dir, f'{idx+16:05d}.jpg')
    shutil.copy(src_path, dst_path)

# Create anomaly.json file
import json

anomaly_data = {
    'anomaly_frames': [round(i / fps, 2) for i in spike_indices],
    'anomaly_certain': any(y > 0.4 for y in y_pred)
}

with open(os.path.join(result_dir, '../anomaly.json'), 'w') as f:
    json.dump(anomaly_data, f)

x_time = [frame_num / fps for frame_num in range(len(img))]
plt.plot(x_time, y_pred)
plt.savefig(args.output_graph, dpi=300)
plt.cla()
shutil.rmtree(args.n[:-4])
shutil.rmtree(save_path)
shutil.rmtree(result_dir)


        
    
