import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import pytorch_lightning as pl

####################
# DATA RELATED HOOKS
####################
class HymenopteraDataModule(pl.LightningDataModule):
    def __init__(self,data_dir='small_hymenoptera_data', batch_size=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                       transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                            std=[1., 1., 1.]),
                                       ])

    def _get_grid_images(self):
        imgs, labels = next(iter(self.val_dataloader()))
        grid = self.invTrans(utils.make_grid(imgs, nrow=4, padding=2))
        return grid

    def setup(self, stage=None): # called on every GPU
        # Assign train/val datasets for use in dataloaders
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in ['train', 'val']}
        self.train_dataset_size = len(self.image_datasets['train'])
        self.class_names = self.image_datasets['train'].classes

    def train_dataloader(self):
        return DataLoader(self.image_datasets['train'], batch_size=self.batch_size,
                                             shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.image_datasets['val'], batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.image_datasets['val'], batch_size=self.batch_size, num_workers=4)

