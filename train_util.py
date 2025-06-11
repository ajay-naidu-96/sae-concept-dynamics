from tqdm import tqdm
import os
import torch


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

        fc1 = []
        fc1_activations = []
        all_labels = []

        save_path = os.path.join(self.log_dir, f"{activation_type}_activation.pt")

        pbar = tqdm(loader, desc=f"Saving Activations: {activation_type}", unit="batch")

        for images, labels in pbar:

            images = images.to(self.device)
            labels = labels.to(self.device)

            _, pre_relu, post_relu = self.model(images)

            fc1.append(pre_relu.cpu())
            fc1_activations.append(post_relu.cpu())
            all_labels.append(labels.cpu())

        torch.save({
            "fc1": torch.cat(fc1),
            "fc1_activations": torch.cat(fc1_activations),
            "fc1_activations_norm": self.mean_center_and_unit_norm(torch.cat(fc1_activations)),
            "labels": torch.cat(all_labels)
        }, save_path)

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
        self.save_activations(self.train_loader, "test")


class TrainLoop():
    def __init__(self, model, loader, optim, criterion, device, args):

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


    def save_model(self, epoch=None):

        save_path = os.path.join(self.log_dir, f"best_model.pt")
        torch.save(self.model.state_dict(), save_path)

        print(f"[INFO] Model weights saved to {save_path}")


    def run(self):
        best_val_acc = 0.0

        p_bar = tqdm(range(self.epochs), desc="Training", unit="epoch")

        for epoch in p_bar:
            self.model.train()

            running_loss = 0.0
            correct = 0
            total = 0

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

            p_bar.set_postfix(
                train_loss=avg_train_loss,
                train_acc=f"{train_acc:.2f}%",
                val_loss=avg_val_loss,
                val_acc=f"{val_acc:.2f}%"
            )

            if (epoch + 1) % 5 == 0 and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
            


        