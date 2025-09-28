import os.path
from typing import Optional

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from ptflops import get_model_complexity_info

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from models.base_components.linear import Linear

class ImageClassifierTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 num_classes: int,
                 model_name: Optional[str] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
                 batch_size: int=32, learning_rate: float=0.1, device: Optional[str]=None):
        """
        model: Image Encoding Model, input(B, C, H, W), output(B, feature_size).
        train_dataset: data.Dataset instance for training set.
        test_dataset: data.Dataset instance for test set.
        criterion: Loss function instance.
        optimizer: Training optimizer instance.
        num_classes: num of classes.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._add_classification_head(model, num_classes).to(self.device)
        self.model_name = model_name if model_name is not None else model.__class__.__name__
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.num_classes = num_classes
        self.criterion =  criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = scheduler if scheduler is not None else lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)

    def _add_classification_head(self, model: nn.Module, num_classes: int):
        """
        Add classification head to torch model.
        """
        # Test out feature size.
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = model(dummy_input)
        feature_size = features.shape[1]
        if feature_size == num_classes:
            return model

        # Create a new model with classification head.
        classifier = nn.Sequential(
            model,
            Linear(feature_size, num_classes)
        )
        return classifier

    def train(self, num_epochs):
        self.model.train()
        pbar = tqdm(total=len(self.train_loader) * self.train_loader.batch_size, desc='Training')
        loss_list = []
        for epoch in range(num_epochs):
            pbar.reset()
            running_loss = 0.0
            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                running_loss += loss.item()
                pbar.update(inputs.shape[0])
                if (i + 1) % 100 == 0:
                    pbar.clear()
                    print(
                        f'\rEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0

            epoch_loss = epoch_loss / len(self.train_loader)
            pbar.clear()
            print(f'\rEpoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            loss_list.append(epoch_loss)
        return pd.DataFrame({
            'Loss': loss_list,
            'Epoch': range(num_epochs),
        })

    def test(self):
        """
        Test the model, return predictions and accuracy.
        """
        self.model.eval()
        predictions = []
        labels = []
        correct = 0
        total = 0
        with (torch.no_grad()):
            for test_input, label in self.test_loader:
                test_input = test_input.to(self.device)
                label = label.to(self.device)
                outputs = self.model(test_input)
                prediction = torch.argmax(outputs.data, dim=-1)
                total += label.size(0)
                correct += (prediction == label).sum().item()
                predictions.extend(prediction.cpu().numpy())
                labels.extend(label.cpu().numpy())

        accuracy = 100 * correct / total

        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'labels': labels
        }

    def generate_report(self, test_results, train_loss, output_path):
        """
        Use sklearn to calculate metrics and save evaluation report.
        Use pyflops to calculate MACs and num of parameters.
        """
        labels = test_results['labels']
        predictions = test_results['predictions']

        # Generate report & calculate flops
        report = classification_report(labels, predictions, target_names=[str(i) for i in range(self.num_classes)])
        _, _, precision, recall, f1, support = report.split('\n')[-3].split()
        macs, params = get_model_complexity_info(
            self.model,
            (3, 224, 224),
            as_strings=True,
            print_per_layer_stat=False
        )

        # Plot confusion matrix heatmap.
        cm = confusion_matrix(labels, predictions)
        cm_fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(self.num_classes),
                    yticklabels=range(self.num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Plot loss
        loss_fig = plt.figure(figsize=(10, 8))
        sns.lineplot(data=train_loss, x='Epoch', y='Loss')
        plt.xlabel('Loss')
        plt.ylabel('Epoch')
        plt.title('Loss Plot')

        # Save reports.
        report_file = os.path.join(output_path, os.path.join(self.model_name, 'report.txt'))
        confusion_matrix_file = os.path.join(output_path, os.path.join(self.model_name, 'confusion_matrix.png'))
        model_file = os.path.join(output_path,os.path.join(self.model_name, 'best_model.pth'))
        competitive_report_file = os.path.join(output_path, f'comparison_report.csv')
        loss_plot_file = os.path.join(output_path,os.path.join(self.model_name, 'loss_plot.png'))
        loss_file = os.path.join(output_path, os.path.join(self.model_name, 'loss.csv'))

        if not os.path.exists(report_file):
            os.makedirs(os.path.dirname(report_file), exist_ok=True)

        report = 'Classification Report:\n' + report + f'\nMACs:{macs}\tParams:{params}\n'
        print(report)
        with open(report_file, 'w') as f:
            f.write(report)

        cm_fig.savefig(confusion_matrix_file)
        loss_fig.savefig(loss_plot_file)

        torch.save(self.model.state_dict(), model_file)

        data = {
            'Model': [self.model_name],
            'MACs': [macs],
            'Params': [params],
            'Precision': [float(precision)],
            'Recall': [float(recall)],
            'F1': [float(f1)],
            'Support': [float(support)],
        }
        data = pd.DataFrame(data)
        if os.path.exists(competitive_report_file):
            comparison_report = pd.read_csv(competitive_report_file)
            print(comparison_report)
            if self.model_name in comparison_report['Model'].values:
                print(comparison_report)
                print(data)
                comparison_report[comparison_report['Model'] == self.model_name] = data.values
            else:
                comparison_report = pd.concat([comparison_report, data])
        else:
            comparison_report = data
        comparison_report.to_csv(competitive_report_file, index=False)

        train_loss.to_csv(loss_file, index=False)

        print(f"Report saved to {report_file}")
        print(f"Loss saved to {loss_plot_file} & {loss_file}")
        print(f"Confusion matrix saved to {confusion_matrix_file}")
        print(f"Model saved to {model_file}")

    def run(self, num_epochs, report_path):
        """
        Run training pipeline.
        """
        print(f"Starting training {self.model_name}.", flush=True)
        train_loss = self.train(num_epochs)

        print("Testing.")
        test_results = self.test()

        print("Generating reports.")
        self.generate_report(test_results, train_loss, report_path)


if __name__ == '__main__':
    from models.resnet import resnet18
    model = resnet18()

    from dataset.image.classification import get_CIFAR10
    train_set, test_set = get_CIFAR10()

    trainer = ImageClassifierTrainer(
        model=model,
        train_dataset=train_set,
        test_dataset=test_set,
        num_classes=10,
        batch_size=64,
        learning_rate=0.001
    )

    trainer.run(num_epochs=1, report_path='Image Classifier Reports')