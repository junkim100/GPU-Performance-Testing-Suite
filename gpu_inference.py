"""
Multi-GPU Inference Performance Tester

This script performs inference on a specified Hugging Face model using all available GPUs
and measures the performance. It uses data parallelism to distribute a single large batch
across all GPUs, providing insights into how efficiently the system can utilize multiple
GPUs for a single inference task.

Author: [Your Name]
Date: [Current Date]
Version: 1.0

Usage:
    python script_name.py [--model_name MODEL_NAME] [--input_text INPUT_TEXT]
                          [--batch_size BATCH_SIZE] [--num_iterations NUM_ITERATIONS]
                          [--output_dir OUTPUT_DIR]

Arguments:
    --model_name (str): Name of the Hugging Face model to use (default: "bert-base-uncased")
    --input_text (str): Input text for inference (default: "Hello, world!")
    --batch_size (int): Batch size for inference (default: 32)
    --num_iterations (int): Number of iterations to run (default: 100)
    --output_dir (str): Directory to save output file (default: current directory)

Requirements:
    - Python 3.6+
    - PyTorch
    - Transformers
    - Fire

Output:
    The script generates a JSON file containing test information and results, including:
    - Test configuration (model name, input text, batch size, number of iterations)
    - Number of GPUs used
    - Total inference time
    - Average time per iteration

Note:
    This script suppresses all warnings to provide cleaner output. Make sure to address
    any relevant warnings if unexpected behavior occurs.

License: Apache License 2.0
"""

import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import fire
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def load_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model


def run_inference(model_name, input_text, batch_size=32, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    model = load_model(model_name)
    model = nn.DataParallel(model)  # Wrap the model for data parallelism
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a larger batch by repeating the input
    inputs = tokenizer(
        [input_text] * batch_size, return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(**inputs)

    torch.cuda.synchronize()

    # Actual inference
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_iteration = total_time / num_iterations

    return {
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "total_time": total_time,
        "avg_time_per_iteration": avg_time_per_iteration,
    }


def main(
    model_name="bert-base-uncased",
    input_text="Hello, world!",
    batch_size=32,
    num_iterations=100,
    output_dir=None,
):
    """
    Run inference on a model using all available GPUs.

    Args:
        model_name (str): Name of the model to use from Hugging Face.
        input_text (str): Input text for inference.
        batch_size (int): Batch size for inference.
        num_iterations (int): Number of iterations to run.
        output_dir (str): Directory to save output file. If None, saves in current directory.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(
            output_dir, f"gpu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    else:
        output_file_path = f"gpu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        results = run_inference(model_name, input_text, batch_size, num_iterations)

        # Prepare the output dictionary
        output = {
            "test_info": {
                "date": datetime.now().isoformat(),
                "model_name": model_name,
                "input_text": input_text,
                "batch_size": batch_size,
                "num_iterations": num_iterations,
            },
            "results": results,
        }

        # Write output to the file
        with open(output_file_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Testing completed. Results saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    fire.Fire(main)
