from tqdm import tqdm
import os
import torch
import json
import csv


class ActivationLogger():

    def __init__(self, model, loader, device, args):
        
        self.train_loader = loader['train']
        self.test_loader = loader['test']
        self.val_loader = loader['val']
        self.device = device

        self.model = model
        self.model.to(self.device)
        
        self.log_dir = args.log_dir

    def mean_center_and_unit_norm(self, x, eps=1e-8):
        x_centered = x - x.mean(dim=1, keepdim=True)
        
        norm = x_centered.norm(p=2, dim=1, keepdim=True) + eps
        x_normalized = x_centered / norm
        
        return x_normalized

    def save_activations(self, loader, activation_type):

        self.model.eval()

        save_dir = os.path.join(self.log_dir, f"{activation_type}_activations")
        os.makedirs(save_dir, exist_ok=True)

        pbar = tqdm(loader, desc=f"Saving Activations: {activation_type}", unit="batch")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                _, pre_relu, post_relu = self.model(images)

                # Store current batch dictionary
                batch_dict = {
                    "fc1": pre_relu.cpu() if pre_relu is not None else None,
                    "fc1_activations": post_relu.cpu() if post_relu is not None else None,
                    "labels": labels.cpu()
                }

                batch_path = os.path.join(save_dir, f"batch_{batch_idx:05d}.pt")
                torch.save(batch_dict, batch_path)

        print(f"Activations saved incrementally to {save_dir}")

    def run(self):

        self.model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _, _ = self.model(images)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100.0 * test_correct / test_total

        print(f"Final Test Accuracy: {test_acc:.2f}%")

        self.save_activations(self.train_loader, "train")
        self.save_activations(self.test_loader, "test")


class TrainLoop():
    def __init__(self, model, loader, optim, criterion, device, args, scheduler=None):

        self.model = model

        self.train_loader = loader['train']
        self.test_loader = loader['test']
        self.val_loader = loader['val']

        self.optimizer = optim
        self.loss_fn = criterion

        self.device = device

        self.epochs = args.epochs

        self.model = self.model.to(self.device)
        
        self.log_dir = args.log_dir
        self.scheduler = scheduler
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': None,
            'test_acc': None,
            'epochs': []
        }

    def save_model(self, epoch=None):
        save_path = os.path.join(self.log_dir, f"oracle.pt")
        torch.save(self.model.state_dict(), save_path)
        print(f"[INFO] Model weights saved to {save_path}")

    def save_metrics(self):
        """Save training metrics to JSON and CSV files"""
        
        # Save as JSON
        json_path = os.path.join(self.log_dir, "training_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save as CSV for easy plotting
        csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            
            # Write data
            for i in range(len(self.metrics['epochs'])):
                writer.writerow([
                    self.metrics['epochs'][i],
                    self.metrics['train_loss'][i],
                    self.metrics['train_acc'][i],
                    self.metrics['val_loss'][i],
                    self.metrics['val_acc'][i]
                ])
        
        print(f"[INFO] Training metrics saved to {json_path} and {csv_path}")

    def evaluate_test_set(self):
        """Evaluate model on test set and store metrics"""
        self.model.eval()
        
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _, _ = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                test_loss += loss.item()
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        avg_test_loss = test_loss / len(self.test_loader)
        test_acc = 100.0 * test_correct / test_total
        
        # Store test metrics
        self.metrics['test_loss'] = avg_test_loss
        self.metrics['test_acc'] = test_acc
        
        print(f"[INFO] Final Test - Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        
        return avg_test_loss, test_acc

    def run(self):
        best_val_acc = 0.0

        p_bar = tqdm(range(self.epochs), desc="Training", unit="epoch")

        for epoch in p_bar:
            self.model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            # Training phase
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs, _, _ = self.model(images)

                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = running_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total

            if self.scheduler:
                self.scheduler.step()
                
            # Validation phase
            self.model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs, _, _ = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(self.val_loader)
            val_acc = 100.0 * val_correct / val_total

            # Store metrics
            self.metrics['epochs'].append(epoch + 1)
            self.metrics['train_loss'].append(avg_train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(avg_val_loss)
            self.metrics['val_acc'].append(val_acc)

            p_bar.set_postfix(
                train_loss=avg_train_loss,
                train_acc=f"{train_acc:.2f}%",
                val_loss=avg_val_loss,
                val_acc=f"{val_acc:.2f}%"
            )

            # Save model if validation accuracy improves
            if (epoch + 1) % 5 == 0 and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
            
            # Save metrics every 10 epochs (optional, for monitoring during long training)
            if (epoch + 1) % 10 == 0:
                self.save_metrics()
        
        # Final evaluation and saving
        self.evaluate_test_set()
        self.save_metrics()
        
        print(f"[INFO] Training completed. Best validation accuracy: {best_val_acc:.2f}%")