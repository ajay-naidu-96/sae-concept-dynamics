from tqdm import tqdm
import os
import torch

class TrainLoop():
    def __init__(self, model, loader, optim, criterion, device, args):

        self.model = model
        self.loader = loader
        self.optimizer = optim
        self.loss_fn = criterion
        self.device = device
        self.epochs = args.epochs

        self.model = self.model.to(self.device)
        
        self.log_dir = args.log_dir


    def save_model(self):

        save_path = os.path.join(self.log_dir, "model.pt")
        torch.save(self.model.state_dict(), save_path)
        print(f"[INFO] Model weights saved to {save_path}")


    def save_activations(self):

        self.model.eval()

        fc1 = []
        fc1_activations = []
        all_labels = []

        save_path = os.path.join(self.log_dir, "activation.pt")

        pbar = tqdm(self.loader, desc="Saving activations", unit="batch")

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
            "labels": torch.cat(all_labels)
        }, save_path)


    def run(self):

        self.model.train()

        p_bar = tqdm(range(self.epochs), desc="Training", unit="epoch")
        
        for epoch in p_bar:

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.loader:

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

            avg_loss = running_loss / len(self.loader)
            accuracy = 100.0 * correct / total

            p_bar.set_postfix(loss=avg_loss, acc=f"{accuracy:.2f}%")

        self.save_model()
        self.save_activations()

        


        