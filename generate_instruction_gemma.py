from __future__ import annotations

import argparse
import gc
import io
import json
import os
import random
import time
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
import torch
import transformers
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def encode_prompt(prompt_instructions: list[dict[str]]):
    """Encode multiple prompt instructions into a single string."""
    prompt = "<bos><start_of_turn>user\n" + open("./prompt.txt").read() + "\n"
    prompt += "<start_of_turn>model\n"
    prompt += f"I see. List of 20 tasks:\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process(num_prompt_instructions: int, response: str):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1:
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks.jsonl",
    num_instructions_to_generate=100,
    model_name="google/gemma-7b-it",
    num_prompt_instructions=3,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    is_vllm=False,
    seed=0
):
    if seed is None:
        seed = int(time.time())
    transformers.set_seed(seed)
    
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r", encoding='utf-8')]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    # load model and tokenizer
    if is_vllm:
        model = LLM(
            model=model_name,
            tokenizer=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.99,
            seed=seed
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=4096,
            seed=seed
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
        prompt = encode_prompt(prompt_instructions)
        request_start = time.time()

        # generate instruction
        if is_vllm:
            outputs = model.generate(
                prompt,
                sampling_params=sampling_params,
                use_tqdm=False
            )
            result = outputs[0].outputs[0].text
            del outputs
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=4096,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )
            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=4096,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )

            output_ids = output_ids.to("cpu")
            result = tokenizer.decode(output_ids.tolist()[0]).replace(prompt, "").replace("<bos>", "").replace("<eos>", "")
            del input_ids, output_ids

        gc.collect()
        torch.cuda.empty_cache()
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = post_process(num_prompt_instructions, result)

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-7b-it")
    parser.add_argument("--seed_tasks_path", default="./seed_tasks.jsonl")
    parser.add_argument("--num_instructions_to_generate", default=100)
    parser.add_argument("--num_prompt_instructions", default=3)
    parser.add_argument("--is_vllm", default=False)
    parser.add_argument("--seed", default=None)
    args = parser.parse_args()

    generate_instruction_following_data(
        seed_tasks_path=args.seed_tasks_path,
        model_name=args.model,
        num_instructions_to_generate=args.num_instructions_to_generate,
        num_prompt_instructions=args.num_prompt_instructions,
        is_vllm=args.is_vllm,
        seed=args.seed
    )
