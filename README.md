# GPU Performance Testing Suite

Welcome to the **GPU Performance Testing Suite** repository! This collection of scripts is designed to help you evaluate and benchmark the performance of your multi-GPU server setups. Whether you're testing inference speeds, memory usage, or overall efficiency, this suite provides the tools you need to get accurate and insightful results.

## Features

- **Multi-GPU Inference Testing**: Measure how effectively your system utilizes multiple GPUs for a single inference task.
- **Flexible Model Support**: Easily test different models from Hugging Face's model hub.
- **Customizable Parameters**: Adjust batch sizes, input texts, and iteration counts to suit your specific testing needs.
- **Comprehensive Output**: Generate detailed JSON reports with test configurations and results.
- **Warning Suppression**: Run tests with clean output by suppressing unnecessary warnings.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- PyTorch
- Transformers
- Fire

You can install the required Python packages using pip:

```bash
pip install torch transformers fire
```

### Arguments
--model_name: Name of the Hugging Face model to use (default: "bert-base-uncased").
--input_text: Input text for inference (default: "Hello, world!").
--batch_size: Batch size for inference (default: 32).
--num_iterations: Number of iterations to run (default: 100).
--output_dir: Directory to save output file (default: current directory).

### Example
```bash
python script_name.py --model_name bert-base-uncased --input_text "Hello, world!" --batch_size 32 --num_iterations 100 --output_dir ./results
```

### Output
The script generates a JSON file containing:
Test configuration (model name, input text, batch size, number of iterations)
Number of GPUs used
Total inference time
Average time per iteration
