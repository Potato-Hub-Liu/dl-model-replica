import torch
from torch import optim
from torch.nn.functional import batch_norm
from torch.optim import lr_scheduler


torch.random.manual_seed(42)

from pipeline.training.image_classifier import ImageClassifierTrainer
from models.resnet import (resnet18, resnet34, resnet50, resnet101, resnet152)
from models.vgg import (vgg11, vgg13, vgg16, vgg19)

model_dict = {
    'ResNet18': {'model': resnet18, 'args': {'pre_activation': False}},
    'ResNet18_Preactivation': {'model': resnet18, 'args': {'pre_activation': True}},
    'ResNet34': {'model': resnet34, 'args': {'pre_activation': False}},
    'ResNet34_Preactivation': {'model': resnet34, 'args': {'pre_activation': True}},
    'ResNet50': {'model': resnet50, 'args': {'pre_activation': False}},
    'ResNet50_Preactivation': {'model': resnet50, 'args': {'pre_activation': True}},
    'ResNet101': {'model': resnet101, 'args': {'pre_activation': False}},
    'ResNet101_Preactivation': {'model': resnet101, 'args': {'pre_activation': True}},
    'ResNet152': {'model': resnet152, 'args': {'pre_activation': False}},
    'ResNet152_Preactivation': {'model': resnet152, 'args': {'pre_activation': True}},
    'VGG11': {'model': vgg11, 'args': { 'batch_norm':True }},
    'VGG13': {'model': vgg13, 'args': { 'batch_norm':True }},
    'VGG16': {'model': vgg16, 'args': { 'batch_norm':True }},
    'VGG19': {'model': vgg19, 'args': { 'batch_norm':True }},
}

def train_image_classification_models(models: list)-> None:
    from dataset.image.classification import get_CIFAR10
    train_set, test_set = get_CIFAR10()

    for name in models:
        model, args = model_dict[name].values()
        model = model(*args)
        optimizer = optim.SGD(model.parameters(),
                              lr=0.01,
                              momentum=0.9,
                              weight_decay=1e-4)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 180], gamma=0.1)

        trainer = ImageClassifierTrainer(
            model=model,
            train_dataset=train_set,
            test_dataset=test_set,
            num_classes=10,
            batch_size=128,
            optimizer=optimizer,
            scheduler=scheduler,
            model_name=name,
        )

        trainer.run(num_epochs=200, report_path='Image Classifier Reports')


import sys
if __name__ == '__main__':
    args = sys.argv
    train_image_classification_models(models=args[1:])
