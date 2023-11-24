import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
from nanodpo.causal_transformer import CausalTransformer
from nanodpo.simple_sequence_model import SimpleSequenceModel
from nanodpo.preference_data import simulate_dpo_dataset_noise
from nanodpo.sequence_data import prepare_sequence_datasets
import warnings
warnings.filterwarnings("ignore")

class DPOOneModelTrainer:
    def __init__(self, model, model_dir, device=torch.device("cpu"), learning_rate=0.005, batch_size=32, margin=1.0):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        # Model, optimizer setup remains the same
        self.criterion = nn.MarginRankingLoss(margin=margin).to(self.device)
        self.loss_history = []
        self.accuracy_history = []
        self.eval_interval = -1
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f"model_epoch_{epoch}.pt"))

    def load_model(self, model_version=None):
        if model_version is not None:
            model_path = os.path.join(self.model_dir, f"model_epoch_{model_version}.pt")
        else:
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_epoch_") and f.endswith(".pt")]
            if not model_files:
                return  # No model to load
            latest_model = max(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
            model_path = os.path.join(self.model_dir, latest_model)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def train(self, train_dataset, test_dataset, epochs=10, eval_interval=2, step_size=10, gamma=0.1):
        self.eval_interval = eval_interval
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # DataLoader setup remains the same
        self.model.train()

        # Initialize the scheduler
        scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(epochs):
            total_loss = 0
            for inputs, yw, yl in train_loader:
                inputs = inputs.to(self.device)
                yw = yw.to(self.device)
                yl = yl.to(self.device)

                # Data prep steps
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                # Get scores for preferred and rejected actions
                preferred_scores = outputs[torch.arange(outputs.size(0)), yw]
                rejected_scores = outputs[torch.arange(outputs.size(0)), yl]

                # Targets for MarginRankingLoss (all +1s, as preferred should be ranked higher)
                targets = torch.ones(preferred_scores.size(0), device=preferred_scores.device)

                # Compute loss
                loss = self.criterion(preferred_scores, rejected_scores, targets)
                total_loss += loss.item()

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
            
            current_lr = scheduler.get_last_lr()[0]

            epoch_loss = total_loss / len(train_loader)
            self.loss_history.append(epoch_loss)

            # Evaluate at specified intervals
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                print(f"Current Learning Rate: {current_lr}")
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}')
                # Evaluate the model
                self.model.eval()
                accuracy = self.evaluate(test_dataset)
                self.model.train()  # Switch back to training mode
                self.accuracy_history.append(accuracy)
                # Save model after evaluation
                self.save_model(epoch + 1)

                # Log metrics to wandb
                wandb.log({"epoch": epoch, "loss": epoch_loss, "accuracy": accuracy})

    # Evaluation function can remain the same or be adapted for ranking-based metrics
    def evaluate(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()

        total_correct = 0
        total_samples = 0
        action_count = {'preferred': {}, 'rejected': {}, 'predicted': {}}

        with torch.no_grad():
            for inputs, preferred_actions, rejected_actions in test_loader:
                inputs = inputs.to(self.device)
                preferred_actions = preferred_actions.to(self.device)
                rejected_actions = rejected_actions.to(self.device)
                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                for action in torch.unique(preferred_actions).tolist():
                    action_count['preferred'][action] = action_count['preferred'].get(action, 0) + (preferred_actions == action).sum().item()
    
                for action in torch.unique(rejected_actions).tolist():
                    action_count['rejected'][action] = action_count['rejected'].get(action, 0) + (rejected_actions == action).sum().item()
    
                for action in torch.unique(predicted).tolist():
                    action_count['predicted'][action] = action_count['predicted'].get(action, 0) + (predicted == action).sum().item()
    
                total_samples += preferred_actions.size(0)
                total_correct += (predicted == preferred_actions).sum().item()

        accuracy = (total_correct / total_samples) * 100
        print(f'Accuracy: {accuracy}%')

        # Diagnostic information
        print("Action Counts:")
        for action_type, counts in action_count.items():
            print(f"{action_type.capitalize()}: {counts}")
    
        return accuracy
    
    def plot_metrics(self):
        if self.eval_interval < 1:
            return
        plt.figure(figsize=(12, 5))

        # Plotting Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # Adjusting x-axis for Accuracy
        accuracy_x_axis = list(range(self.eval_interval, epochs + 1, self.eval_interval))
        if epochs % eval_interval != 0:
            accuracy_x_axis.append(epochs)  # Add the last epoch if it doesn't align with the interval
    
        # Plotting Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(accuracy_x_axis, self.accuracy_history, label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Evaluation Accuracy')
        plt.legend()
        
        plt.show()
        plt.savefig('loss_accuracy.png')

if __name__ == '__main__':
    config = {'learning_rate':0.001, 'batch_size': 32, 'sequence_len': 10, 'step_size': 5}
    size = 1000
    feature_dim = 3
    dataset, actions = simulate_dpo_dataset_noise(size=size, feature_dim=feature_dim)
    print(f"Dataset size: {len(dataset)}")
    num_actions = len(actions)
    feature = dataset[0][0]
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    sequence_len = config['sequence_len'] 
    step_size = config['step_size']
    epochs = 100
    eval_interval = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = 'transformer'
    if model_type == 'transformer':
        n_head = 2
        d_model = 4
        n_layer = 2
        assert(d_model % n_head == 0)
        model = CausalTransformer(d_feature= feature_dim, d_model=d_model, n_head=n_head, n_layer=n_layer, 
                                       num_actions=num_actions, max_len=sequence_len,
                                       device=device).to(device)
    elif model_type == 'lstm':
        hidden_dim = 64
        model = SimpleSequenceModel(feature_dim=feature_dim, hidden_dim=hidden_dim, 
                                    num_actions=num_actions).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Assuming 'dataset' is your data in the required format
    action_to_idx = {action:i for i,action in enumerate(actions)}
    train_dataset, test_dataset = prepare_sequence_datasets(dataset, action_to_idx=action_to_idx,
                                                            sequence_len=sequence_len, step_size=step_size)

    model_version = None  # Set to a specific version if needed
    trainer = DPOOneModelTrainer(model=model, model_dir=f"dpo_{model_type}_model/", device=device,
                                 learning_rate=learning_rate, batch_size=batch_size)
    trainer.load_model(model_version)

    # Initialize Weights & Biases
    wandb.init(project='nanoDPO', name=f"dpo_{model_type}")

    # Now you can use train_dataset and test_dataset for training and evaluation
    trainer.train(train_dataset, test_dataset, epochs=epochs, eval_interval=eval_interval)
    trainer.plot_metrics(eval_interval)
    trainer.evaluate(test_dataset)

    wandb.finish()