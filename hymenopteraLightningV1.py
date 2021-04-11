
# Utilisation de TensorBoard : https://learnopencv.com/tensorboard-with-pytorch-lightning/
# C'est du transfert learning depuis un Resnet18
# Les informations sont loggées via TensorBoardLogger dans le répertoire tb_logs
#

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.optim import Adam,SGD
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import  models
from torch.optim import lr_scheduler
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger
from hymenoptereDataModule import HymenopteraDataModule
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


plt.switch_backend('agg')

class LitHymenoptera(pl.LightningModule):

    def __init__(self, hidden_size=64, learning_rate=2e-4, batch_size=4):
        super().__init__()
        torch.manual_seed(42)
        self.batch_size = batch_size
        self.dataModule = HymenopteraDataModule('pieces_pilar')
        self.dataModule.setup()
        self.criterion = nn.CrossEntropyLoss()
        self.logger = TensorBoardLogger('tb_logs', name=f'Model')
        # Define the model
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

    # Training

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        # Compute loss
        loss = self.criterion(logits, y)
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        return {'loss': loss,
                'acc': acc,
                'num_correct': num_correct}

    def training_epoch_end(self, outputs):
        self._calculate_epoch_metrics(outputs, 'Train')
        self.exp_lr_scheduler.step()

    # Validation

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        return {'loss': loss,
                'acc': acc,
                'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        self._calculate_epoch_metrics(outputs, 'Validation')

    # Test

    def test_step(self, batch, batch_idx):
        if batch_idx < 5:
            inputs, classes = batch
            self.logger.experiment.add_figure(f'predictions vs. actuals {batch_idx}',
                              self.plot_classes_preds(inputs, classes))
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self._calculate_epoch_metrics(outputs, 'Test')


    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return optimizer

    ######################### Methods used for logs

    def images_to_probs(self, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = self(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(self, images, labels):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self.images_to_probs(images)
        # print(preds)
        # print(probs)
        # plot the images in the batch, along with predicted and true labels
        my_dpi = 96 # For my monitor (see https://www.infobyip.com/detectmonitordpi.php)
        fig = plt.figure(figsize=(4 * 224/my_dpi, 224/my_dpi), dpi=my_dpi)
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
            img = self.dataModule.invTrans(images[idx])
            npimg = img.cpu().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.dataModule.class_names[preds[idx]],
                probs[idx] * 100.0,
                self.dataModule.class_names[labels[idx]]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        return fig

    def _calculate_epoch_metrics(self, outputs, name):
         loss_mean = torch.stack([output['loss'] for output in outputs]).mean()
         acc_mean = torch.stack([output['num_correct'] for output in outputs]).sum().float()
         acc_mean /= (len(outputs) * self.batch_size)
         self.logger.experiment.add_scalar(f'Loss/{name}', loss_mean, self.current_epoch) # self.logger.experiment est un objet de classe SummaryWriter
         self.logger.experiment.add_scalar(f'Accuracy/{name}', acc_mean, self.current_epoch)


############################################################################
#  Programme principal
############################################################################
import torchvision

def imshow(images, title=None):
    """Imshow for Tensor."""
    img_grid = torchvision.utils.make_grid(images).cpu()
    inp = img_grid.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)

seed = 123

model = LitHymenoptera()
#checkpoint_callback = ModelCheckpoint(dirpath='tb_logs/')
trainer = pl.Trainer(gpus=1, max_epochs=20, progress_bar_refresh_rate=20, logger=model.logger) #, callbacks=[checkpoint_callback])
trainer.fit(model,model.dataModule)
trainer.test(model)

# Test du modèle
#model = LitHymenoptera.load_from_checkpoint(checkpoint_path="tb_logs/Model/version_10/checkpoints/epoch=4-step=304.ckpt")

import matplotlib
matplotlib.use('TkAgg')  # Pour que les images s'affichent à l'écran

# Save the model predictions and true labels
y_pred = []
y_valid = []
model.freeze()
for inputs, classes in model.dataModule.val_dataloader():
    #imshow(inputs)
    preds, probs = model.images_to_probs(inputs.cuda())
    y_pred.extend(preds)
    y_valid.extend(classes)

print('ok')
# Calculate needed metrics
print(y_valid)
print(y_pred)
print(f'Accuracy score on validation data:\t{accuracy_score(y_valid, y_pred)}')
