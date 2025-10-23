#!/usr/bin/env python3
"""
LLaVA Model Evaluation Script

This script performs inference using LLaVA models on multimodal data.
It supports both image-text and text-only tasks with distributed processing.

Author: [Your Name]
Date: [Current Date]
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate import InitProcessGroupKwargs
import datetime

# LLaVA imports
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def load_image(image_file: str) -> Image.Image:
    """Load image from file path or URL."""
    if image_file.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def process_prompt(
    caption: str, 
    args, 
    mm_use_im_start_end: bool, 
    model_name: str, 
    has_image: bool = True
) -> str:
    """Process prompt for model input with image token handling."""
    qs = caption
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if has_image:
        if IMAGE_PLACEHOLDER in qs:
            if mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    else:
        if IMAGE_PLACEHOLDER in qs:
            qs = qs.replace(IMAGE_PLACEHOLDER, "")

    # Determine conversation mode based on model name
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f"[WARNING] Auto inferred conversation mode is {conv_mode}, "
              f"while `--conv-mode` is {args.conv_mode}, using {args.conv_mode}")
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def load_data(data_path: str) -> List[Dict]:
    """Load data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def process_single_item(
    item: Dict, 
    args, 
    tokenizer, 
    model, 
    image_processor, 
    mm_use_im_start_end: bool, 
    model_name: str
) -> Dict:
    """Process a single data item for inference."""
    new_sample = {}
    
    if 'image' in item:
        # Process image-text task
        inp = '\n'.join(item["conversations"][0]['value'].split('\n')[1:])
        prompt = process_prompt(inp, args, mm_use_im_start_end, model_name, has_image=True)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        
        new_sample['question'] = prompt
        new_sample['label_answer'] = item["conversations"][1]['value']
        new_sample['image'] = item['image']
        
        # Load and process image
        try:
            images = [Image.open(item['image']).convert("RGB")]
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images, image_processor, model.config
            ).to(model.device, dtype=torch.float16)
        except Exception as e:
            print(f"Error loading image {item['image']}: {e}")
            return None
        
        # Generate responses
        outputs = []
        for i in range(args.generate_nums):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if len(output) >= 800:  # Prevent overly long outputs
                break
            outputs.append(output)
            
    else:
        # Process text-only task
        inp = item["conversations"][0]['value']
        prompt = process_prompt(inp, args, mm_use_im_start_end, model_name, has_image=False)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        
        new_sample['question'] = prompt
        new_sample['label_answer'] = item["conversations"][1]['value']
        
        # Generate responses
        outputs = []
        for i in range(args.generate_nums):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=None,
                    image_sizes=None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if len(output) >= 800:  # Prevent overly long outputs
                break
            outputs.append(output)
    
    new_sample['gpt_answer'] = outputs if args.generate_nums > 1 else outputs[0]
    return new_sample


def evaluate_model(args):
    """Main evaluation function."""
    disable_torch_init()
    process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=540000))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    
    if args.model_path == "None":
        args.model_path = args.model_base
        args.model_base = None
        print("Changed args.model_path to args.model_base")
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=accelerator.process_index
    )
    mm_use_im_start_end = model.config.mm_use_im_start_end
    
    # Load data
    data = load_data(args.input_data_path)
    print(f"Total number of samples: {len(data)}")
    
    accelerator.wait_for_everyone()
    start = time.time()
    
    with accelerator.split_between_processes(data) as prompts:
        results = []
        
        for num, item in tqdm(enumerate(prompts), total=len(prompts), desc="Processing"):
            try:
                result = process_single_item(
                    item, args, tokenizer, model, image_processor, 
                    mm_use_im_start_end, model_name
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing item {num}: {e}")
                continue
            
            # Save intermediate results periodically
            if num % args.save_interval == 0 and num != 0 and accelerator.is_main_process:
                results_gathered = gather_object(results)
                formatted_data = json.dumps(results_gathered, indent=0, ensure_ascii=False)
                
                timediff = time.time() - start
                output_file_path = f"{args.output_data_path}_checkpoint_{num}.json"
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    file.write(formatted_data)
                print(f"Checkpoint saved at {num}, time elapsed: {timediff:.2f}s")
                start = time.time()
    
    # Save final results
    results_gathered = gather_object(results)
    formatted_data = json.dumps(results_gathered, indent=0, ensure_ascii=False)
    
    if accelerator.is_main_process:
        timediff = time.time() - start
        with open(args.output_data_path, 'w', encoding='utf-8') as file:
            file.write(formatted_data)
        print(f"Final results saved, total time elapsed: {timediff:.2f}s")
        print(f"Processed {len(results_gathered)} samples successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLaVA Model Evaluation")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the LLaVA model")
    parser.add_argument("--model_base", type=str, default=None,
                       help="Base model path (for delta weights)")
    parser.add_argument("--conv_mode", type=str, default=None,
                       help="Conversation mode (auto-detected if not specified)")
    
    # Data arguments
    parser.add_argument("--input_data_path", type=str, required=True,
                       help="Path to input JSON data file")
    parser.add_argument("--output_data_path", type=str, required=True,
                       help="Path to output JSON results file")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--generate_nums", type=int, default=1,
                       help="Number of generations per sample")
    
    # Processing arguments
    parser.add_argument("--save_interval", type=int, default=5000,
                       help="Save intermediate results every N samples")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for processing")
    
    # Optional arguments
    parser.add_argument("--image_file", type=str, help="Single image file for testing")
    parser.add_argument("--query", type=str, help="Single query for testing")
    parser.add_argument("--sep", type=str, default=",", help="Separator for multiple images")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode")
    
    args = parser.parse_args()

    if args.debug:
        args.temperature = 0
        args.generate_nums = 1
        print("Debug mode enabled")
    
    evaluate_model(args)


if __name__ == "__main__":
    main()
