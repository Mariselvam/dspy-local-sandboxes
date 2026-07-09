# Local DSPy Sandboxes: Sentiment Analysis & Prompt Optimization

This project contains two introductory sandboxes demonstrating how to build declarative AI systems using **DSPy** and local large language models (LLMs) hosted via **LM Studio**. 

Instead of treating prompts like manual English string modifications, these projects treat prompts as **structured software data contracts (Signatures)** that can be executed zero-shot or programmatically optimized.

---

##  Core Architecture & Concept

### 1. What they are doing
* **Program 1 (`sentiment.py`):** Takes an unstructured corporate customer review and extracts a definitive classification (`Positive`, `Negative`, or `Neutral`) along with a step-by-step chain of thought reasoning trail.
* **Program 2 (`optimizer.py`):** Acts as a self-improving AI. It takes a raw, unoptimized router layout and runs it against a small training dataset. It then automatically gathers correct logic paths and compiles a mathematically optimal few-shot prompt to route medical emergency symptoms to specific hospital departments.

### 2. How they are doing it
Traditional frameworks rely on manual string interpolation (f-strings). DSPy replaces this by treating prompt structures like neural network layers:
1.  **Signatures (`dspy.Signature`):** We define strict data contracts declaring explicitly what fields go into the model (`InputField`) and what keys must return (`OutputField`).
2.  **Modules (`dspy.ChainOfThought`):** We pass the contract to a module that tells the LLM to write out its internal reasoning steps before emitting the structured fields.
3.  **Optimizers (`BootstrapFewShot`):** Instead of humans writing example prompts, the script feeds a dataset into an algorithmic optimizer. The framework iterates over your local model, tests outputs against a validation metric, and injects the highest-performing examples into the underlying instructions automatically.

---

## Prerequisites & Local Model Setup

This workflow executes entirely on your local machine using an OpenAI-compatible mock server wrapper.

1. Download and open **LM Studio**.
2. Load your chosen local model (e.g., `Qwen 2.5`, `Llama 3`, or `Mistral`).
3. Click the **Local Server** icon on the left sidebar.
4. Keep the port set to `1234` and click **Start Server**.
5. Note the exact model string identifier displayed in your server logs (e.g., `qwen3.5-9b`).

---

## Install Dependencies

Install the primary DSPy framework within your active environment:

`pip install dspy-ai`

---