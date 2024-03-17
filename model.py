
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, features, img_size):
        super().__init__()
        self.img_size = img_size
        bottleneck_size = self.img_size[-1]//(2**5)

        self.encoder = nn.Sequential(

            nn.Conv2d(self.img_size[0], features*8, (3,3), (1,1), 1),
            nn.BatchNorm2d(features*8),
            nn.ReLU(),

            self._enc_block(features*8 , features*16),
            self._enc_block(features*16, features*16),
            self._enc_block(features*16, features*32),
            self._enc_block(features*32, features*16),
            self._enc_block(features*16, features*16),
             
              #256x4x4

            nn.Flatten(),
            nn.Linear(features*16*bottleneck_size*bottleneck_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            #nn.Softmax(dim=1), #Normalise

        )


    def _enc_block(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(out_channels), #normalises data
            nn.ReLU6(),
            nn.Dropout(0.1),
        )

    def forward(self, images):
        return self.encoder(images)
    
    def isSame(self, image1, image2, threshold=0.01):
        self.eval()
        image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)
        mod1 = self(image1)
        mod2 = self(image2)
        mse = F.mse_loss(mod1, mod2, reduction="None")
        mse = mse.mean().item()
        return mse < threshold