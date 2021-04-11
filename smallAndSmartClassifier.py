#
# Code de https://learnopencv.com/getting-started-with-pytorch-lightning/
# et de https://learnopencv.com/tensorboard-with-pytorch-lightning/
#
# Montre l'utilisation de Lightning sur un exemple simple (MNIST)
# ainsi que divers types de log (scalar, image, ...) à l'aide de TensorBoard
#

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt


class smallAndSmartModel(pl.LightningModule):
    def __init__(self):
        super(smallAndSmartModel, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,28,kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(28,10,kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout1=torch.nn.Dropout(0.25)
        self.fc1=torch.nn.Linear(250,18)
        self.dropout2=torch.nn.Dropout(0.08)
        self.fc2=torch.nn.Linear(18,10)

    def forward(self,x):
          x=self.layer1(x)
          x=self.layer2(x)
          x=self.dropout1(x)
          x=torch.relu(self.fc1(x.view(x.size(0), -1)))
          x=F.leaky_relu(self.dropout2(x))

          return F.softmax(self.fc2(x))

    def configure_optimizers(self):
        # Essential fuction
        #we are using Adam optimizer for our model
        return torch.optim.Adam(self.parameters())

    def training_step(self,batch,batch_idx):
        # REQUIRED- run at every batch of training data
        if(batch_idx==0):
            self.reference_image=(batch[0][0]).unsqueeze(0)

        # extracting input and output from the batch
        x, labels = batch

        # forward pass on a batch
        pred = self.forward(x)

        # identifying number of correct predections in a given batch
        correct = pred.argmax(dim=1).eq(labels).sum().item()

        # identifying total number of labels in a given batch
        total = len(labels)

        # calculating the loss
        train_loss = F.cross_entropy(pred, labels)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Train_loss (batch_idx)",
                                          train_loss,
                                          batch_idx + self.nb_of_batch * self.current_epoch)

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,

            # info to be used at epoch end
            "correct": correct,
            "total": total
        }

        return batch_dictionary

    def training_epoch_end(self,outputs):
        #  the function is called after every epoch is completed
        self.showActivations(self.reference_image)

        if self.current_epoch==1:

            sampleImg = torch.rand((1,1,28,28))
            self.logger.experiment.add_graph(smallAndSmartModel(),sampleImg)

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correct and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          correct / total,
                                          self.current_epoch)

    def makegrid(self, output, numrows):
        outer = (torch.Tensor.cpu(output).detach())
        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while (i < outer.shape[1]):
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if (j == numrows):
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c

    def showActivations(self, x):
        # logging reference image
        self.logger.experiment.add_image("input", torch.Tensor.cpu(x[0][0]), self.current_epoch, dataformats="HW")

        # logging layer 1 activations
        out = self.layer1(x)
        c = self.makegrid(out, 4)
        self.logger.experiment.add_image("layer 1", c, self.current_epoch, dataformats="HW")

        # logging layer 2 activations
        out = self.layer2(out)
        c = self.makegrid(out, 8)
        self.logger.experiment.add_image("layer 2", c, self.current_epoch, dataformats="HW")

    # This contains the manupulation on data that needs to be done only once such as downloading it
    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    def train_dataloader(self):
        # This is an essential function. Needs to be included in the code
        # See here i have set download to false as it is already downloaded in prepare_data
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())

        # Dividing into validation and training set
        self.train_set, self.val_set = random_split(mnist_train, [55000, 5000])
        self.nb_of_batch = len(self.train_set) / 128

        return DataLoader(self.train_set, batch_size=128)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_set, batch_size=128)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor()), batch_size=128)

#
#  Programme
#
logger = TensorBoardLogger('lightning_logs', name='model')
myTrainer=pl.Trainer(gpus=1, max_epochs=3, logger=logger)
model = smallAndSmartModel()
myTrainer.fit(model)
