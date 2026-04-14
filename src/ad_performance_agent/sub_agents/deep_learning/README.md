# Deep Learning Agent Tools

This directory contains tools for the deep learning agent to train classification models.

## Files

- **agent.py**: The main agent configuration with registered tools
- **tools.py**: Training functions that can be called programmatically
- **prompt.py**: Agent prompt/instructions
- **example_usage.py**: Example scripts showing how to use the tools

## Available Tools

### 1. `train_classification_model()`

Direct function call with individual parameters.

**Parameters:**
- `size` (str): Dataset size - "1k" or "10k" (default: "10k")
- `epochs` (int): Number of training epochs (default: 30)
- `batch_size` (int): Batch size for training (default: 32)
- `lr` (float): Learning rate (default: 1e-4)
- `hidden_dim` (int): Hidden layer dimension (default: 128)
- `loss_type` (str): Loss function - "ce" or "focal" (default: "focal")

**Returns:**
```python
{
    "accuracy": float,  # Test accuracy score
    "report": str       # Classification report
}
```

**Example:**
```python
from src.ad_performance_agent.sub_agents.deep_learning.tools import train_classification_model

results = train_classification_model(
    size="1k",
    epochs=20,
    batch_size=32,
    lr=1e-4,
    hidden_dim=128,
    loss_type="focal"
)

print(f"Accuracy: {results['accuracy']:.4f}")
```

### 2. `train_with_config()`

Train using a configuration dictionary. Great for hyperparameter tuning or when you have config from JSON/YAML.

**Parameters:**
- `config` (dict): Configuration dictionary with any of the parameters from `train_classification_model()`

**Returns:**
Same as `train_classification_model()`

**Example:**
```python
from src.ad_performance_agent.sub_agents.deep_learning.tools import train_with_config

# Full config
config = {
    "size": "1k",
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-3,
    "hidden_dim": 256,
    "loss_type": "ce"
}

results = train_with_config(config)

# Minimal config (uses defaults for unspecified params)
config = {"size": "1k", "epochs": 10}
results = train_with_config(config)
```

## Usage in Other Scripts

### Import and use directly:

```python
from src.ad_performance_agent.sub_agents.deep_learning.tools import (
    train_classification_model,
    train_with_config
)

# Method 1: Direct call
results = train_classification_model(size="1k", epochs=5, lr=1e-3)

# Method 2: Using dict
config = {"size": "1k", "epochs": 5}
results = train_with_config(config)
```

### Hyperparameter tuning example:

```python
from src.ad_performance_agent.sub_agents.deep_learning.tools import train_classification_model

learning_rates = [1e-3, 1e-4, 1e-5]
results = {}

for lr in learning_rates:
    result = train_classification_model(
        size="1k",
        epochs=10,
        lr=lr
    )
    results[lr] = result['accuracy']

best_lr = max(results, key=results.get)
print(f"Best learning rate: {best_lr} with accuracy: {results[best_lr]:.4f}")
```

## Integration with the Agent

The tools are automatically registered with the deep_learning_agent in `agent.py`:

```python
deep_learning_agent = Agent(
    model=AGENT_LLM_MODEL,
    name="deep_learning_agent",
    instruction=DEEP_LEARNING_AGENT_PROMT,
    tools=[train_classification_model, train_with_config],
)
```

This allows the agent to call these functions during conversations.

## Default Values

| Parameter | Default | Options |
|-----------|---------|---------|
| size | "10k" | "1k", "10k" |
| epochs | 30 | any positive int |
| batch_size | 32 | any positive int |
| lr | 1e-4 | any positive float |
| hidden_dim | 128 | any positive int |
| loss_type | "focal" | "ce", "focal" |

## Notes

- Both functions validate inputs and raise `ValueError` if invalid values are provided
- The training uses the dataset from `/data/influencer_dataset_subset_{size}/`
- Results include both accuracy score and detailed classification report
- Progress is printed to stdout during training
