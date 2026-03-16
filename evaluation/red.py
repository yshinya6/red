import argparse
import json
import os
import pdb
import sys
import time

import shortuuid
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import eval_results
from src.decoding.multipath import _sample
from src.decoding.utils import validate_model_kwargs
from src.utils.instructions import INSTRUCTION_TRIPLE
from src.utils.make_input import get_make_input_func

login(token=os.environ["HUGGINGFACE_TOKEN"])
transformers.generation.utils.GenerationMixin._validate_model_kwargs = validate_model_kwargs


def get_automodel_class(model_path: str):
    model_name = model_path.split("/")[-1].lower()
    if model_name.startswith("phi"):
        return AutoModelForCausalLM
    else:
        return AutoModelForImageTextToText


def eval_model(args):

    # -----Model Construction------
    model_path = os.path.expanduser(args.model_path)
    model_name = model_path
    automodel_class = get_automodel_class(model_name)
    make_input = get_make_input_func(model_name)
    model = automodel_class.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    processor.tokenizer.padding_side = "left"
    if args.generation_config:
        generation_config = GenerationConfig.from_pretrained(model_name)
    else:
        generation_config = None
    original_sample_func = transformers.generation.utils.GenerationMixin._sample

    # -----Dataset Construction-----
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    responses = []
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # -----Counter Initialization------
    counter = 0
    total_elapsed_time = 0.0
    total_reasoning_time = 0.0
    total_prefill_time = 0.0
    total_token_per_sec = 0.0
    total_generated_tokens = 0.0

    # -----Main Evaluation Loop-----
    for line in tqdm(questions, total=len(questions)):
        counter += 1
        idx = line["question_id"]
        category = line["category"]
        question = line["text"].replace("<image>", "")
        answer = line["answer"]
        image_path = os.path.join(args.image_folder, line["image"])

        # Get instructions for specified rationale type
        rationale_inst, answer_inst, header = INSTRUCTION_TRIPLE[args.rationale_type]

        # [Step 1] Rationale Generation
        query_rationale = "Question: " + question.split("\n")[0] + "\n\n" + rationale_inst
        model_kwargs = make_input(processor, image_path, query_rationale).to(model.device, dtype=torch.bfloat16)
        input_len = model_kwargs["input_ids"].shape[-1]
        transformers.generation.utils.GenerationMixin._sample = original_sample_func
        with torch.inference_mode():
            start = time.perf_counter()
            output = model.generate(
                **model_kwargs,
                max_new_tokens=args.max_rationale_tokens,
                use_cache=True,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                cache_implementation=args.cache_implementation,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
            end = time.perf_counter()
        rationale = processor.decode(output.sequences[0][input_len:], skip_special_tokens=True).strip()
        reasoning_time = end - start

        # [Step 2] Final Answer Generation
        query_answer_img = question
        query_answer_rat = header + rationale + "\n\n" + answer_inst + "\nQuestion: " + question
        model_kwargs = make_input(processor, [image_path, None], [query_answer_img, query_answer_rat]).to(
            model.device, dtype=torch.bfloat16
        )
        context_score = torch.tensor([1.0, args.lambda_rat]).to(model.device)
        input_len = model_kwargs["input_ids"].shape[-1]
        transformers.generation.utils.GenerationMixin._sample = _sample
        with torch.inference_mode():
            start = time.perf_counter()
            output = model.generate(
                **model_kwargs,
                max_new_tokens=args.max_answer_tokens,
                use_cache=True,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                cache_implementation=args.cache_implementation,
                generation_config=generation_config,
                return_dict_in_generate=True,
                compile_strategy=args.compile_strategy,
                context_scores=context_score,
            )
            end = time.perf_counter()
        outputs = processor.decode(output.sequences[0][input_len:], skip_special_tokens=True).strip()
        elapsed_time = end - start

        # Log metrics
        token_per_sec = len(output.sequences[0][input_len:]) / elapsed_time
        total_elapsed_time += elapsed_time
        total_token_per_sec += token_per_sec
        total_prefill_time += output.prefill_time
        total_reasoning_time += reasoning_time
        total_generated_tokens += len(output.sequences[0]) - input_len

        if args.verbose and (counter % args.print_freq == 0):
            print("-----")
            print(f"Question: {line['text']}")
            print(f"Rationale: {rationale}")
            print(f"Answer: {outputs}")
            print(f"GT: {answer}")
            print(f"Avg. Decoding Time (sec): {total_elapsed_time / counter}")
            print(f"Avg. Generated Tokens: {total_generated_tokens / counter}")
            print(f"Avg. Token / Sec: {total_token_per_sec / counter}")
            print("-----")

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": question,
                    "text": outputs,
                    "rationale": rationale,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        responses.append(
            {"question_id": idx, "category": category, "question": question, "prediction": outputs, "answer": answer}
        )
        # ans_file.flush()
        del model_kwargs, output
    ans_file.close()

    print(f"Average Decoding Time (sec): {total_elapsed_time / counter}")
    print(f"Avg. Prefill Time (sec): {total_prefill_time / counter}")
    print(f"Avg. Reasoning Time (sec): {total_reasoning_time / counter}")
    print(f"Avg. Generated Tokens: {total_generated_tokens / counter}")
    print(f"Average Token / Sec: {total_token_per_sec / counter}")

    # Summarize results
    print("### Task Results")
    scores = eval_results(args.evaluation, responses)
    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--assistant-model-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./data/gqa/")
    parser.add_argument("--question-file", type=str, default="./MIBench/multimodal_in_context_learning_val.jsonl")
    parser.add_argument("--answers-file", type=str, default="./test/answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_rationale_tokens", type=int, default=1024)
    parser.add_argument("--max_answer_tokens", type=int, default=1024)
    parser.add_argument("--cache_implementation", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--generation_config", action="store_true")
    parser.add_argument("--num_logits_to_keep", type=int, default=None)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--evaluation", type=str, default="gqa")
    parser.add_argument("--rationale_type", type=str, default="cot")
    parser.add_argument("--lambda_rat", type=float, default=0.5)
    parser.add_argument("--compile_strategy", type=str, default="poe")

    args = parser.parse_args()
    eval_model(args)
