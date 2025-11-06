# Uni-Layout
## Uni-Layout: Integrating Human Feedback in Unified Layout Generation and Evaluation
[ACM MM 2025] Official PyTorch Code for "Uni-Layout: Integrating Human Feedback in Unified Layout Generation and Evaluation"

<img width="928" alt="image" src="images/overview.png"> 

## 📢 News

`[2025-09-02]:` 🚀 CoT data has been released! You can now find it in the ["Dataset for Reward Model" link](https://drive.google.com/drive/folders/1VASp90_mqSwJxJH65v5-iP9Sk3tgr23M?usp=drive_link).

`[2025-08-04]:` 🎯 Our paper is now available on arXiv! Check it out here: [https://arxiv.org/abs/2508.02374](https://arxiv.org/abs/2508.02374).

`[2025-07-04]:` 🎉 Exciting news! Our paper has been accepted to ACM MM 2025! Stay tuned for more updates!

## 🚀 Code & Weights Notice

- Layout Evaluator Checkpoints: [Download Link](https://drive.google.com/drive/folders/1evrHmorHW7CBLRhxrV3-3qvFki1ovoJ3?usp=drive_link)

### 🧪 Evaluation

- **Script**: `evaluation.py`

#### Requirements
- Python >= 3.8 (recommend Anaconda/Miniconda)
- PyTorch >= 2.3.1 + CUDA 11.8 (install from official wheel index)
- Extra deps in `requirements.txt`

#### Setup
```bash
conda create -n caig python==3.8.20 -y && conda activate caig
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Run
```bash
python evaluation.py \
  --model_path /path/to/model \
  --input_data_path /path/to/input.json \
  --output_data_path /path/to/output.json
```

#### Notes
- Optional: `--model_base`, `--conv_mode`, generation args (`--temperature`, `--top_p`, `--num_beams`, `--max_new_tokens`, `--generate_nums`), and process args (`--save_interval`, `--batch_size`).
- Input JSON follows the dataset format below; `image` field is optional.

## 📊 Datasets
### 1. Dataset for Layout Generator
[Download Link](https://drive.google.com/drive/folders/1OLWRUZSiecpGuG2sUdQHOnmp46P9ojuD?usp=sharing).

#### Key Fields
- **`sku_id`**: Anonymized sample identifier.
- **`image`**: Path to the image (optional; may be absent for text-only tasks).
- **`conversations`**: List of two messages:
  - **human**: Task description, may include the `<image>` tag, canvas size, element types, and layout constraints.
  - **gpt**: Layout result; `value` is a string in the form `Layout:{...}`, where bounding boxes are `[x_min, y_min, x_max, y_max]`.

### 2. Dataset for Layout Evaluator
[Download Link](https://drive.google.com/drive/folders/1VASp90_mqSwJxJH65v5-iP9Sk3tgr23M?usp=drive_link).

#### Key Fields
- **`image`**: Path to the image.
- **`conversations`**: Single-turn QA pair:
  - **human**: Evaluation instruction with candidate layout and constraints; expects a binary decision (0/1).
  - **gpt**: The answer; `value` is the Ground Truth label (0 or 1).


# Copyright & Licensing
© JD.COM. All rights reserved. The datasets and software provided in this repository are licensed exclusively for academic research purposes. Commercial use, reproduction, or distribution requires express written permission from JD.COM. Unauthorized commercial use constitutes a violation of these terms and is strictly prohibited.
