import torch
import glob
from PIL import Image
from torchvision import transforms

class VGGDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size=256, split='train'):
        super().__init__()
        self.files = glob.glob(path + '/**/*.jpg', recursive=True)
        self.files.sort()
        if split == 'train':
            self.files = self.files[1000:]
        else:
            self.files = self.files[:1000]
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                             transforms.ToTensor(),  # scale to 0~1
                                             ])

    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(file).convert('RGB')
        x = self.transform(image)
        return x

    def __len__(self):
        return len(self.files)
