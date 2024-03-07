import torch
import torchaudio
import torchvision
import torch.nn as nn

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCH = 64

"""
Write Code for Downloading Image and Audio Dataset Here
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import torch
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import TimeMasking, FrequencyMasking
from torch.utils.data import Dataset, random_split
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Image Downloader
transform = transforms.Compose([
            transforms.ToTensor(),
])
image_dataset_downloader = torchvision.datasets.CIFAR10(root = './data', train=True, download = True, transform=transform)


# Audio Downloader
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(root = './data', url='speech_commands_v0.02', download = True)



class ImageDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split

        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
        ])

        # Download CIFAR10 dataset
        self.cifar10_dataset = image_dataset_downloader

        if split=='test':
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        else:
            train_size = int(0.8 * len(self.cifar10_dataset))
            val_size = len(self.cifar10_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(self.cifar10_dataset, [train_size, val_size])
            if split == 'train':
                self.dataset = train_dataset
            elif split == 'val':
                self.dataset = val_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    

class AudioDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise ValueError("Data split must be in ['train', 'test', 'val']")

        self.datasplit = split
        self.dataset = audio_dataset_downloader

        # Define transformations
        self.transform = torchvision.transforms.Compose([
            MelSpectrogram(sample_rate=16000, n_mels=128),
            TimeMasking(time_mask_param=30),
            FrequencyMasking(freq_mask_param=30)
        ])

        # Split dataset into train, val, test
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.2 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        if split == "train":
            self.dataset, _ = random_split(self.dataset, [train_size, len(self.dataset) - train_size])
        elif split == "val":
            self.dataset, _ = random_split(self.dataset, [val_size, len(self.dataset) - val_size])
        else:
            _, self.dataset = random_split(self.dataset, [train_size + val_size, test_size])

        self.label_dict = {
            'backward': 0, 'bed': 1, 'bird': 2, 'cat': 3, 'dog': 4,
            'down': 5, 'eight': 6, 'five': 7, 'follow': 8, 'forward': 9,
            'four': 10, 'go': 11, 'happy': 12, 'house': 13, 'learn': 14,
            'left': 15, 'marvin': 16, 'nine': 17, 'no': 18, 'off': 19,
            'on': 20, 'one': 21, 'right': 22, 'seven': 23, 'sheila': 24,
            'six': 25, 'stop': 26, 'three': 27, 'tree': 28, 'two': 29,
            'up': 30, 'visual': 31, 'wow': 32, 'yes': 33, 'zero': 34
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
        waveform = self.transform(waveform)
        # print("Wave shape:", waveform.shape)

        # desired_length = 16000  # Desired length of waveform
        # if waveform.size(1) < desired_length:
        #     # Pad if the waveform is shorter than the desired length
        #     pad_amount = desired_length - waveform.size(2)
        #     waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        # elif waveform.size(1) > desired_length:
        #     # Truncate if the waveform is longer than the desired length
        #     waveform = waveform[:, :desired_length]
        pad_length = 81 - waveform.shape[2]
        if pad_length > 0:
            padding = torch.zeros(waveform.shape[0], waveform.shape[1], pad_length)
            waveform = torch.cat([waveform, padding], dim=2)

        numerical_label = self.label_dict[label]  # Convert label to numerical representation

        return waveform, numerical_label
    
class ResNet_block(nn.Module):
    def __init__(self, in_channels, out_channels, is_audio=False):
        super(ResNet_block, self).__init__()

        self.is_audio = is_audio

        self.conv1_im = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_im = nn.BatchNorm2d(32)
        self.conv2_im = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn2_im = nn.BatchNorm2d(16)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

        self.conv1_aud = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1_aud = nn.BatchNorm1d(out_channels)
        self.relu_aud = nn.ReLU()
        self.conv2_aud = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2_aud = nn.BatchNorm1d(out_channels)
        self.skip_connection_aud = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # self.upsample_aud = nn.Upsample(scale_factor=(4), mode='nearest')

    def forward(self, x):

        if not self.is_audio:
            out = self.conv1_im(x)
            out = self.bn1_im(out)
            out = nn.ReLU()(out)
            out = self.bn2_im(self.conv2_im(out))
            # out = self.upsample(out)
            # out += self.conv_skip(x)
            out += x
            out = nn.ReLU()(out)
            return out

        else:
            residual = x
            x = self.conv1_aud(x)
            x = self.bn1_aud(x)
            x = self.relu_aud(x)
            x = self.conv2_aud(x)
            x = self.bn2_aud(x)
            res = self.skip_connection_aud(residual)
            x += res
            x = self.relu_aud(x)
            return x

class Resnet_Q1(nn.Module):
    def __init__(self,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_audio = False

        self.conv1_im = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_im = nn.BatchNorm2d(16)
        self.layer1_im = self._make_layer(ResNet_block, 18)
        self.fc_im = nn.Linear(64, 10)

        self.conv_block1 = ResNet_block(128, 64, is_audio=True)
        self.conv_blocks = nn.ModuleList([ResNet_block(64, 64, is_audio=True) for _ in range(18 - 1)])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 81, 36)  # T is the number of time steps

    def _make_layer(self, block, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(128, 64))
        return nn.Sequential(*layers)

    def forward(self, x):

        if len(x.shape) == 3:
            self.is_audio = True

        if not self.is_audio:
            out = self.conv1_im(x)
            out = self.bn1_im(out)
            out = nn.ReLU()(out)
            out = self.layer1_im(out)
            out = nn.AvgPool2d(8)(out)
            out = out.view(out.size(0), -1)
            out = self.fc_im(out)
            return out

        else:
            x = self.conv_block1(x)
            for conv_block in self.conv_blocks:
                x = conv_block(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

class VGG_Q2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.is_audio = False

        self.conv_initial_im = nn.Conv2d(3, 288, kernel_size=1, padding=1)
        self.conv_initial_aud = nn.Conv1d(128, 288, kernel_size=1, padding=1)

        self.conv1_im = nn.Sequential(
            nn.Conv2d(288, 240, kernel_size=1, padding=1),
            nn.BatchNorm2d(240),
            nn.Conv2d(240, 180, kernel_size=1, padding=1),
            nn.BatchNorm2d(180),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.conv1_aud = nn.Sequential(
            nn.Conv1d(288, 240, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(240),
            nn.Conv1d(240, 180, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(180),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.conv2_im = nn.Sequential(
            nn.Conv2d(180, 144, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(144),
            nn.Conv2d(144, 130, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(130),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.conv2_aud = nn.Sequential(
            nn.Conv1d(180, 144, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(144),
            nn.Conv1d(144, 130, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(130),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.conv3_im = nn.Sequential(
            nn.Conv2d(130, 130, kernel_size=5, padding=1),
            nn.BatchNorm2d(130),
            nn.Conv2d(130, 110, kernel_size=5, padding=1),
            nn.BatchNorm2d(110),
            nn.Conv2d(110, 72, kernel_size=5, padding=1),
            nn.BatchNorm2d(72),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.conv3_aud = nn.Sequential(
            nn.Conv1d(130, 130, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(130),
            nn.Conv1d(130, 110, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(110),
            nn.Conv1d(110, 72, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(72),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.conv4_im = nn.Sequential(
            nn.Conv2d(72, 60, kernel_size=7, padding=1),
#             nn.ReLU(inplace=True)
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 48, kernel_size=7, padding=1),
#             nn.ReLU(inplace=True)
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, kernel_size=7, padding=1),
#             nn.ReLU(inplace=True)
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.conv4_aud = nn.Sequential(
            nn.Conv1d(72, 60, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(60),
            nn.Conv1d(60, 48, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(48),
            nn.Conv1d(48, 48, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(48),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.conv5_im = nn.Sequential(
            nn.Conv2d(48, 40, kernel_size=9, padding=2),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 32, kernel_size=9, padding=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=9, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.conv5_aud = nn.Sequential(
            nn.Conv1d(48, 40, kernel_size=9, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(40),
            nn.Conv1d(40, 32, kernel_size=9, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=9, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.fc_im = nn.Sequential(
            nn.Linear(288, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

        self.fc_aud = nn.Sequential(
            nn.Linear(1664, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 36)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            self.is_audio = True

        if not self.is_audio:
            x = self.conv_initial_im(x)
            x = self.conv1_im(x)
            x = self.conv2_im(x)
            x = self.conv3_im(x)
            x = self.conv4_im(x)
            x = self.conv5_im(x)
            x = x.view(x.size(0), -1)
            x = self.fc_im(x)
        else:
            x = self.conv_initial_aud(x)
            x = self.conv1_aud(x)
            x = self.conv2_aud(x)
            x = self.conv3_aud(x)
            x = self.conv4_aud(x)
            x = self.conv5_aud(x)
            x = x.view(x.size(0), -1)
            x = self.fc_aud(x)

        return x

        
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_audio=False):
        super(InceptionBlock, self).__init__()

        self.is_audio = is_audio

        # Path 1: 1x1 Convolution
        self.path1_im = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.path1_aud = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[0], kernel_size=1),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Path 2: 3x3 Convolution -> 5x5 Convolution
        self.path2_1_im = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.path2_2_im = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.path2_1_aud = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.path2_2_aud = nn.Sequential(
            nn.Conv1d(out_channels[1], out_channels[2], kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels[2]),
            nn.ReLU(inplace=True)
        )

        # Path 3: 3x3 Convolution -> 5x5 Convolution
        self.path3_1_im = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )
        self.path3_2_im = nn.Sequential(
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels[4]),
            nn.ReLU(inplace=True)
        )

        self.path3_1_aud = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels[3]),
            nn.ReLU(inplace=True)
        )
        self.path3_2_aud = nn.Sequential(
            nn.Conv1d(out_channels[3], out_channels[4], kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels[4]),
            nn.ReLU(inplace=True)
        )

        # Path 4: 3x3 Max Pooling
        self.path4_conv_im = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels[5], kernel_size=1),
            nn.BatchNorm2d(out_channels[5]),
            nn.ReLU(inplace=True)
        )

        self.path4_conv_aud = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels[5], kernel_size=1),
            nn.BatchNorm1d(out_channels[5]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if not self.is_audio:
            # Path 1: 1x1 Convolution
            out1 = self.path1_im(x)

            # Path 2: 3x3 Convolution -> 5x5 Convolution
            out2 = self.path2_1_im(x)
            out2 = self.path2_2_im(out2)

            # Path 3: 3x3 Convolution -> 5x5 Convolution
            out3 = self.path3_1_im(x)
            out3 = self.path3_2_im(out3)

            # Path 4: 3x3 Max Pooling
            out4 = self.path4_conv_im(x)
        else:
            # Path 1: 1x1 Convolution
            out1 = self.path1_aud(x)

            # Path 2: 3x3 Convolution -> 5x5 Convolution
            out2 = self.path2_1_aud(x)
            out2 = self.path2_2_aud(out2)

            # Path 3: 3x3 Convolution -> 5x5 Convolution
            out3 = self.path3_1_aud(x)
            out3 = self.path3_2_aud(out3)

            # Path 4: 3x3 Max Pooling
            out4 = self.path4_conv_aud(x)

        # Concatenate along the channel dimension
        out = torch.cat((out1, out2, out3, out4), dim=1)

        return out
    
class Inception_Q3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.is_audio = False
        
        self.inception1_aud = InceptionBlock(in_channels=128, out_channels=[8, 16, 16, 16, 16, 8], is_audio=True)
        self.inception2_aud = InceptionBlock(in_channels=48, out_channels=[32, 32, 32, 16, 16, 16], is_audio=True)
        self.inception3_aud = InceptionBlock(in_channels=96, out_channels=[48, 72, 72, 48, 48, 48], is_audio=True) 
        self.inception4_aud = InceptionBlock(in_channels=216, out_channels=[32, 64, 64, 32, 32, 32], is_audio=True) 
        
        self.inception1 = InceptionBlock(in_channels=3, out_channels=[8, 16, 16, 16, 16, 8])
        self.inception2 = InceptionBlock(in_channels=48, out_channels=[32, 32, 32, 16, 16, 16])
        self.inception3 = InceptionBlock(in_channels=96, out_channels=[48, 72, 72, 48, 48, 48]) 
        self.inception4 = InceptionBlock(in_channels=216, out_channels=[32, 64, 64, 32, 32, 32]) 

        self.fc1_im = nn.Linear(10240, 1024)
        self.fc2_im = nn.Linear(1024, 128)
        self.fc3_im = nn.Linear(128, 10)

        self.fc1_aud = nn.Linear(12960, 2048)  # T is the number of time steps
        self.fc2_aud = nn.Linear(2048, 128)
        self.fc3_aud = nn.Linear(128, 36)

    def forward(self, x):
        if len(x.shape) == 3:
            self.is_audio = True

        if not self.is_audio:
            x = self.inception1(x)
            x = self.inception2(x)
            x = self.inception3(x)
            x = self.inception4(x)
            x = F.avg_pool2d(x, kernel_size=4)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1_im(x))
            x = F.relu(self.fc2_im(x))
            x = self.fc3_im(x)
        else:
            x = self.inception1_aud(x)
            x = self.inception2_aud(x)
            x = self.inception3_aud(x)
            x = self.inception4_aud(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1_aud(x))
            x = F.relu(self.fc2_aud(x))
            x = self.fc3_aud(x)

        return x
        
class CustomNetwork_Q4(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.is_audio = False

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks
        self.residual_block1_im = ResNet_block(in_channels=16, out_channels=32)
        self.residual_block2_im = ResNet_block(in_channels=16, out_channels=32)

        # Inception Blocks
        self.inception_block1_im = InceptionBlock(in_channels=16, out_channels=[8, 16, 16, 16, 16, 8])
        self.inception_block2_im = InceptionBlock(in_channels=48, out_channels=[32, 32, 32, 16, 16, 16])
        self.change_conv_im = nn.Conv2d(96, 16, kernel_size=1)

        #3
        self.change_conv3_im = nn.Conv2d(96, 16, kernel_size=1)
        self.residual_block3_im = ResNet_block(in_channels=64, out_channels=128)
        self.inception_block3_im = InceptionBlock(in_channels=16, out_channels=[48, 72, 72, 48, 48, 48])
        
        
        #4
        self.change_conv4_im = nn.Conv2d(216, 16, kernel_size=1)
        self.residual_block4_im = ResNet_block(in_channels=64, out_channels=128)
        self.inception_block4_im = InceptionBlock(in_channels=16, out_channels=[48, 72, 72, 48, 48, 48])
        
        
        # 5
        self.change_conv5_im = nn.Conv2d(216, 16, kernel_size=1)
        self.residual_block5_im = ResNet_block(in_channels=64, out_channels=128)
        self.inception_block5_im = InceptionBlock(in_channels=16, out_channels=[48, 72, 72, 48, 48, 48])

        # Final Fully Connected Layers
        self.fc1_im = nn.Linear(3456, 1024)
        self.fc2_im = nn.Linear(1024, 128)
        self.fc3_im = nn.Linear(128, 10)
        
        
    
        # AUDIOOOOOOOOOOOOOOOOO
        # Residual Blocks
        self.conv_block1 = ResNet_block(128, 64, is_audio=True)
        self.residual_block1 = ResNet_block(in_channels=64, out_channels=64, is_audio=True)
        self.residual_block2 = ResNet_block(in_channels=64, out_channels=64, is_audio=True)

        # Inception Blocks
        self.inception_block1 = InceptionBlock(in_channels=64, out_channels=[8, 16, 16, 16, 16, 8], is_audio=True)
        self.inception_block2 = InceptionBlock(in_channels=48, out_channels=[32, 32, 32, 16, 16, 16], is_audio=True)

        #3
        self.change_conv3 = nn.Conv1d(96, 64, kernel_size=1)
        self.residual_block3 = ResNet_block(in_channels=64, out_channels=64, is_audio=True)
        self.inception_block3 = InceptionBlock(in_channels=64, out_channels=[48, 72, 72, 48, 48, 48], is_audio=True)
        
        #4
        self.change_conv4 = nn.Conv1d(216, 64, kernel_size=1)
        self.residual_block4 = ResNet_block(in_channels=64, out_channels=128, is_audio=True)
        self.inception_block4 = InceptionBlock(in_channels=128, out_channels=[48, 72, 72, 48, 48, 48], is_audio=True)
        
        # 5
        self.change_conv5 = nn.Conv1d(216, 64, kernel_size=1)
        self.residual_block5 = ResNet_block(in_channels=64, out_channels=128, is_audio=True)
        self.inception_block5 = InceptionBlock(in_channels=128, out_channels=[48, 72, 72, 48, 48, 48], is_audio=True)
        

        # Final Fully Connected Layers
        self.fc1 = nn.Linear(270, 128)
        self.fc2 = nn.Linear(128, 35)

    def forward_im(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual Blocks
        x = self.residual_block1_im(x)
        x = self.residual_block2_im(x)

        # Inception Blocks
        x = self.inception_block1_im(x)
        x = self.inception_block2_im(x)

        # 3
        x = self.change_conv3_im(x)
        x = self.residual_block3_im(x)
        x = self.inception_block3_im(x)
        
        #4
        x = self.change_conv4_im(x)
        x = self.residual_block4_im(x)
        x = self.inception_block4_im(x)
        
        #5
        x = self.change_conv5_im(x)
        x = self.residual_block5_im(x)
        x = self.inception_block5_im(x)

        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_im(x))
        x = F.relu(self.fc2_im(x))
        x = self.fc3_im(x)

        return x

    
    def forward_aud(self, x):
        # Residual Blocks
        x = self.conv_block1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)

        # Inception Blocks
        x = self.inception_block1(x)
        x = self.inception_block2(x)

        # 3
        x = self.change_conv3(x)
        x = self.residual_block3(x)
        x = self.inception_block3(x)

        #4
        x = self.change_conv4(x)
        x = self.residual_block4(x)
        x = self.inception_block4(x)
        
        #5
        x = self.change_conv5(x)
        x = self.residual_block5(x)
        x = self.inception_block5(x)

        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def forward(self, x):
        if len(x.shape) == 3:
            self.is_audio = True

        if self.is_audio:
            return self.forward_aud(x)
        else:
            return self.forward_im(x)
        
def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    network = network.to(device)
    
    max_acc = 0
    for epoch in range(EPOCH):

        if epoch> 3 and max_acc > 60:
            break
        
        network.train()
        for idx, (images, labels) in enumerate(dataloader):
            
            if(images.shape[3] == 81):
                images = images.squeeze(1)
            images, labels = images.to(device), labels.to(device)
    
            # Forward pass ➡
            outputs = network(images)
            loss = criterion(outputs, labels)

            # Backward pass ⬅
            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()
            
        network.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dataloader:
                if(images.shape[3] == 81):
                    images = images.squeeze(1)
                images, labels = images.to(device), labels.to(device)
                outputs = network(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

        if accuracy > max_acc:
            max_acc = accuracy
            torch.save(network.state_dict(), 'model.pth')

        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            loss, 
            accuracy
        ))

def validator(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    # network = network.to(device)
    network.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    network = network.to(device)


    max_acc = 0
    for epoch in range(EPOCH):

        if epoch > 3 and max_acc > 60:
            break

        network.train()
        for idx, (images, labels) in enumerate(dataloader):
            
            if(images.shape[3] == 81):
                images = images.squeeze(1)
            images, labels = images.to(device), labels.to(device)
    
            # Forward pass ➡
            outputs = network(images)
            loss = criterion(outputs, labels)

            # Backward pass ⬅
            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()

        network.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dataloader:
                if(images.shape[3] == 81):
                    images = images.squeeze(1)
                images, labels = images.to(device), labels.to(device)
                outputs = network(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

        if accuracy > max_acc:
            max_acc = accuracy
            torch.save(network.state_dict(), 'model.pth')
        
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            loss, 
            accuracy
        ))



def evaluator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    # network = network.to(device)
    network.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    network = network.to(device)

    criterion = nn.CrossEntropyLoss()

    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            if(images.shape[3] == 81):
                images = images.squeeze(1)
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    
    