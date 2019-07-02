from torchvision.models import resnet152
import torch.nn as nn
import torch.nn.functional as NF


__all__ = ['pdn']

class PartDetectNet(nn.Module):
	def __init__(self):
		super(PartDetectNet, self).__init__()

		# b5: download pretrained resnet152, remove last two layers, change strides of conv layers to 1
		base_net = list(resnet152(pretrained=True).children())[:-2]
		base_net[-1][0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		base_net[-1][0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.b5 = nn.Sequential(*base_net)
		#print(base_net[-1]); exit(1)

		# b6
		self.b6 = nn.Conv2d(2048, 16, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)

		# b7
		self.b7 = nn.ConvTranspose2d(16, 16, kernel_size=(4, 4), stride=(4, 4), bias=False)


	def forward(self, x):
		x = self.b5(x)
		x = self.b6(x)
		x = self.b7(x)
		#x = NF.interpolate(x, 256, mode='bilinear', align_corners=True)

		return x

def pdn(**kwargs):
    model = PartDetectNet()
    return model
