import torch
from sklearn.model_selection import train_test_split

# Function to prepare sequence datasets(TimeSeries data) 
def prepare_sequence_datasets(dataset, action_to_idx, sequence_len, step_size, test_size=0.2):
    # Adjust the loopback period according to step_size
    loopback_period = sequence_len * step_size

    # Convert dataset into features, winning (preferred) actions, and losing (rejected) actions
    features = [torch.tensor([data[0] for data in dataset[i:i+loopback_period:step_size]], dtype=torch.float32) 
                for i in range(len(dataset) - loopback_period + 1)]

    winning_actions = [torch.tensor(action_to_idx[dataset[i+loopback_period-step_size][1]], dtype=torch.long) 
                       for i in range(len(dataset) - loopback_period + 1)]

    losing_actions = [torch.tensor(action_to_idx[dataset[i+loopback_period-step_size][2]], dtype=torch.long) 
                      for i in range(len(dataset) - loopback_period +1)]

    # Combine winning and losing actions for splitting
    actions = list(zip(winning_actions, losing_actions))

    # Splitting the dataset
    X_train, X_test, actions_train, actions_test = train_test_split(features, actions, test_size=test_size)

    # Separate winning and losing actions after splitting
    yw_train, yl_train = zip(*actions_train)
    yw_test, yl_test = zip(*actions_test)

    # Create PyTorch datasets
    train_dataset = list(zip(X_train, yw_train, yl_train))
    test_dataset = list(zip(X_test, yw_test, yl_test))

    return train_dataset, test_dataset
