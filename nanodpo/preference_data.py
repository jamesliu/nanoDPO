# filename: preference_data.py

import numpy as np
import random

# Preference data generation Yw > Yl
def simulate_dpo_dataset_simple(size=1000, feature_dim=3):
    dataset = []
    actions = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

    for _ in range(size):
        # Randomly choose an action
        Yw = random.choice(actions)

        # Generate features based on the action
        if Yw == 'POSITIVE':
            # Example pattern for POSITIVE
            features = np.random.normal(0.5, 0.05, feature_dim)
        elif Yw == 'NEGATIVE':
            # Example pattern for NEGATIVE
            features = np.random.normal(-0.5, 0.05, feature_dim)
        elif Yw == 'NEUTRAL':
            # Example pattern for NEUTRAL 
            features = np.random.normal(0.0, 0.05, feature_dim)
        else:
            raise ValueError(f"Unknown action: {Yw}")

        for Yl in actions:
            if Yl != Yw:
                # Add to dataset
                dataset.append((features, Yw, Yl))

    return dataset, actions

def simulate_dpo_dataset_noise(size=1000, feature_dim=3):
    dataset = []
    actions = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

    for _ in range(size):
        Yw = random.choice(actions)

        if Yw == 'POSITIVE':
            base = np.random.normal(-0.9, 1, feature_dim)
            noise = np.random.normal(0, 0.05, feature_dim)
            transformation = np.sin(np.linspace(-np.pi, np.pi, feature_dim))
            features = base + noise + transformation

        elif Yw == 'NEGATIVE':
            base = np.random.normal(-0.5, 1, feature_dim)
            noise = np.random.normal(0, 0.05, feature_dim)
            transformation = np.cos(np.linspace(-np.pi - 0.5 , np.pi + 0.5, feature_dim))
            features = base + noise + transformation

        elif Yw == 'NEUTRAL':
            base = np.random.uniform(-0.1, 1, feature_dim)
            noise = np.random.normal(0, 0.05, feature_dim)
            transformation = np.tanh(np.linspace(-1, 1, feature_dim))
            features = base + noise + transformation

        else:
            raise ValueError(f"Unknown action: {Yw}")

        for Yl in actions:
            if Yl != Yw:
                dataset.append((features, Yw, Yl))

    return dataset, actions


if __name__ == '__main__':

    dataset, actions = simulate_dpo_dataset()
    print('dataset', len(dataset))
    for i, data in enumerate(dataset):
        print(data)
        if i > 5:
            break
    