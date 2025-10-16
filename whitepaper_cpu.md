# White Paper: nanochat-cpu

**Title:** nanochat-cpu: A CPU-Optimized Adaptation of the nanochat LLM Framework

**Version:** 1.0

**Date:** October 16, 2025

---

## Abstract

This white paper details the architecture, modifications, and operational pipeline of `nanochat-cpu`, a CPU-only adaptation of the `nanochat` repository. The original `nanochat` is a full-stack, dependency-lite implementation of a ChatGPT-like LLM, designed for high-performance training on multi-GPU nodes. The `nanochat-cpu` version aims to make this powerful educational tool accessible to users without specialized GPU hardware. By strategically scaling down the model, data, and training parameters, and by abstracting away GPU-specific code paths, `nanochat-cpu` provides a fully functional, end-to-end pipeline that can run in CPU-constrained environments such as Hugging Face Spaces. This adaptation retains the original's ethos of being minimal, clean, and hackable, making it an invaluable resource for learning, prototyping, and functional testing of LLM architectures.

---

## 1. Introduction

### 1.1. The Original nanochat Vision

The `nanochat` project was conceived as "the best ChatGPT that $100 can buy," offering a complete, end-to-end pipeline for building an LLM from scratch. Its design principles emphasize simplicity, readability, and minimal dependencies, making it a powerful educational tool for understanding the entire lifecycle of an LLM: data preparation, tokenization, pre-training, supervised fine-tuning (SFT), reinforcement learning (RL), and inference. However, its reliance on high-end multi-GPU nodes (e.g., 8xH100) presented a significant barrier to entry for developers, students, and researchers with limited hardware resources.

### 1.2. Motivation for a CPU-Only Version

The primary motivation for `nanochat-cpu` is to democratize access to the `nanochat` framework. By enabling the entire pipeline to run on a standard CPU, we open the door for:
-   **Accessibility:** Users can run, modify, and experiment with the codebase on personal laptops or basic cloud instances.
-   **Educational Value:** Students and developers can trace the entire training and inference process without needing to provision expensive hardware.
-   **CI/CD and Functional Testing:** The CPU version serves as a lightweight platform for verifying the logical correctness of new features before deploying them to GPU-based training runs.
-   **Hugging Face Spaces Deployment:** The project is specifically tailored for deployment on Hugging Face's free CPU Spaces, providing a live, interactive demo platform for custom-trained nano-models.

### 1.3. Adaptation Strategy

The core strategy for adapting `nanochat` to a CPU-only environment revolves around two key principles:
1.  **Graceful Fallbacks:** Modifying the code to detect the absence of a CUDA-enabled GPU and automatically switch to CPU execution, disabling GPU-specific optimizations.
2.  **Strategic Downscaling:** Providing a reference pipeline (`train_cpu.sh`) that dramatically reduces the computational load by using a smaller model architecture, a smaller tokenizer vocabulary, a smaller dataset, and a significantly shorter training schedule.

---

## 2. Core Architecture and Components

The `nanochat-cpu` repository retains the well-organized structure of the original. The key components are:

-   **`nanochat/`**: The core Python library containing the essential building blocks.
    -   `gpt.py`: The GPT model implementation, featuring rotary embeddings, MQA, and custom optimizer setup.
    -   `common.py`: Utilities for compute initialization, which now includes the critical CPU fallback logic.
    -   `dataloader.py`: The data loading and tokenization pipeline.
    -   `engine.py`: The efficient KV-cache-based inference engine.
    -   `adamw.py`, `muon.py`: Custom optimizers, including non-distributed versions used for CPU training.
    -   `tokenizer.py`: A wrapper for the tokenizer implementation.

-   **`rustbpe/`**: A high-performance BPE tokenizer trainer written in Rust. Its efficiency is crucial for the CPU version, as tokenizer training can be a bottleneck.

-   **`scripts/`**: Executable scripts that drive the end-to-end pipeline.
    -   `tok_train.py`: Trains the tokenizer.
    -   `base_train.py`: Handles model pre-training.
    -   `chat_cli.py`, `chat_web.py`: Provides interfaces for interacting with the trained model.
    -   `*_eval.py`: Scripts for evaluating the model on various benchmarks.

-   **`train_cpu.sh`**: The master script for running a complete, albeit minimal, training pipeline on a CPU. This script is the cornerstone of the `nanochat-cpu` adaptation.

---

## 3. Key Adaptations for CPU Execution

The transition from a GPU-centric to a CPU-only framework required several targeted modifications.

### 3.1. Compute Environment Initialization

The most critical change resides in `nanochat/common.py`. The `compute_init()` function was enhanced to be hardware-aware. The key logic is as follows:

    // In nanochat/common.py -> compute_init()

    // We will prefer CUDA if available, but fall back to CPU
    is_cuda_available = torch.cuda.is_available()
    default_device_str = "cuda" if is_cuda_available else "cpu"

    // ... (seed and precision setup) ...

    // Distributed setup: Distributed Data Parallel (DDP), optional
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        if not is_cuda_available:
            // For this guide, we assume DDP is not used on CPU.
            // A 'gloo' backend would be needed for multi-CPU DDP.
            raise RuntimeError("Distributed training on CPU is not configured in this setup.")
        device = torch.device("cuda", ddp_local_rank)
        // ... (original GPU DDP setup) ...
    else:
        device = torch.device(default_device_str)


This ensures that `torch.device` is set to `"cpu"` if no GPU is found. It also explicitly prevents the use of Distributed Data Parallel (DDP) in a CPU context, simplifying the execution model to a single process.

### 3.2. Mixed-Precision and Optimizations

GPU-specific performance optimizations are conditionally disabled:

-   **Automatic Mixed Precision (AMP):** The `torch.amp.autocast` context manager, used for `bfloat16` training on GPUs, is replaced with a `contextlib.nullcontext` when running on a CPU. This means all operations default to full `float32` precision. This change is implemented in scripts like `scripts/base_train.py`:

        // In scripts/base_train.py
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )

-   **Data Loader Flags:** The `tokenizing_distributed_data_loader` in `nanochat/dataloader.py` uses flags like `pin_memory=True` and `non_blocking=is_cuda`. When `is_cuda` is `False`, these GPU-specific memory optimizations are automatically disabled by PyTorch, ensuring compatibility without code changes.

### 3.3. Optimizer Fallback

The model's `setup_optimizers` method in `nanochat/gpt.py` already contains logic to switch between distributed and non-distributed optimizers based on the DDP state. Since DDP is disabled for the CPU version, the framework seamlessly falls back to using the single-process `Muon` and `torch.optim.AdamW` optimizers.

### 3.4. Scaled-Down Configuration

The `train_cpu.sh` script orchestrates a pipeline that is computationally feasible for a CPU. It achieves this by overriding the default parameters found in the training scripts:

-   **Model Size:** Trains a `depth=4` model instead of the `depth=20` baseline. This drastically reduces the parameter count and computational complexity.
-   **Batch Size:** Uses a small `device_batch_size=4` to manage memory usage.
-   **Training Duration:** Runs for only `num_iterations=20`, sufficient to verify that the training loop works but not intended for achieving model convergence.
-   **Tokenizer:** Trains a smaller tokenizer with a `vocab_size` of 8,192 on only 100MB of data, making the tokenization step fast.

---

## 4. The CPU-Directed Pipeline (`train_cpu.sh`)

The `train_cpu.sh` script provides a complete, runnable example of the `nanochat-cpu` workflow.

1.  **Environment Setup:** It configures the base directory to `/tmp` for compatibility with ephemeral filesystems like those in Hugging Face Spaces. It then downloads a minimal dataset (16 shards) required for the downscaled run.
2.  **Tokenizer Training:** It invokes `scripts/tok_train.py` with parameters for a smaller vocabulary (`8192`) and a much smaller training dataset (`100M` characters), ensuring this step completes in a reasonable time.
3.  **Resource Management:** Immediately after tokenizer training, the script removes the downloaded raw dataset files to conserve disk space. This is a crucial step for resource-constrained environments.
4.  **Base Model Pre-training:** It runs `scripts/base_train.py` with heavily reduced parameters (`depth=4`, `device_batch_size=4`, `num_iterations=20`). This step validates the core training loop, including forward/backward passes and optimizer steps.
5.  **Evaluation:** It runs `scripts/base_eval.py` to evaluate the CORE metric. While the resulting metric for a 20-step model is not meaningful, this step confirms the evaluation pipeline is functional.
6.  **Reporting:** Finally, `nanochat.report generate` is called to compile a summary of the run, demonstrating the end-to-end integrity of the pipeline.

This script intentionally omits the more advanced stages like mid-training, SFT, and RL, as they would be prohibitively slow on a CPU. The focus is on providing a minimal, verifiable, and complete pre-training loop.

---

## 5. Performance Considerations and Limitations

It is critical to understand that `nanochat-cpu` is not a replacement for GPU-based training for any practical application.
-   **Training Speed:** Training is orders of magnitude slower on a CPU. The `train_cpu.sh` script is a proof-of-concept run that takes minutes; a run comparable to the GPU `speedrun.sh` would take weeks or months.
-   **Inference Latency:** Inference (e.g., in `chat_cli.py`) is also significantly slower. Generating tokens one by one on a CPU can result in high latency, making real-time interaction feel sluggish.
-   **Scope:** The framework is intended for educational and testing purposes. Meaningful model training remains the domain of GPUs.

---

## 6. Future Work and Deployment

This white paper serves as the foundational plan for developing a fully working `nanochat-cpu` version.

### 6.1. Deployment on Hugging Face Spaces

The primary goal is to package this repository for deployment on a Hugging Face CPU Space. The `train_cpu.sh` script is designed with this in mind, using a temporary directory and managing disk space carefully. A working deployment would involve:
1.  Creating a `Dockerfile` or `app.py` for the Space.
2.  Including a pre-trained tiny model checkpoint from a `train_cpu.sh` run.
3.  Running the `scripts/chat_web.py` server to provide an interactive UI.

### 6.2. Further Optimizations

While high-performance training is out of scope, inference speed can be improved. A key area for future work is **model quantization**. Applying techniques like `torch.quantization.quantize_dynamic` could significantly reduce the model's memory footprint and accelerate inference on CPUs, making the web UI more responsive.

### 6.3. Extending the Pipeline

The `train_cpu.sh` script could be extended to include minimal, runnable versions of the SFT and mid-training stages. This would provide a complete, albeit slow, demonstration of the entire `nanochat` pipeline on a CPU.

---

## 7. Conclusion

`nanochat-cpu` successfully adapts the powerful, GPU-native `nanochat` framework for CPU-only environments. Through intelligent code modifications for hardware abstraction and a strategically downscaled execution pipeline, it preserves the original's core value as a minimal, hackable, and educational tool. It lowers the barrier to entry for developers and students, enabling them to explore, modify, and understand a complete LLM training pipeline on widely available hardware. This adaptation is not just a port; it is a thoughtful reimagining of the project's scope to prioritize accessibility and learning, paving the way for broader community engagement and experimentation.
