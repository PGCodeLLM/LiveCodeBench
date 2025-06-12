#!/usr/bin/env python
"""
Minimal playground for ByteDance-Seed/Seed-Coder-8B-Reasoning
(with vanilla Transformers + TextIteratorStreamer).

Usage:
    python playground.py            # one-shot generation
    python playground.py --stream   # stream tokens as they appear
"""

import argparse
import sys
import textwrap
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

MODEL_NAME = "ByteDance-Seed/Seed-Coder-8B-Reasoning"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load tokenizer & model (with trust_remote_code)."""
    print("Loading model … (may take a while)")
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else "auto",
        trust_remote_code=True,
    )
    return tok, model


def build_prompt(tok, question: str) -> str:
    """
    If the tokenizer ships with a .chat_template, use it; otherwise
    fall back to a plain prompt with an answer header.
    """
    if getattr(tok, "chat_template", None):
        messages = [
            {
                "role": "user",
                "content": question.strip() + "\n\n### Answer:",
            }
        ]
        return tok.apply_chat_template(messages, tokenize=False)

    return question.strip() + "\n\n### Answer:\n"


def prepare_inputs(tok, prompt: str):
    """
    Tokenise and **only** pass the keys the model expects
    (Seed-Coder is a decoder-only LM → input_ids + attention_mask).
    """
    encoded = tok(prompt, return_tensors="pt").to(DEVICE)
    return {
        "input_ids":      encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
    }, encoded["input_ids"].shape[-1]  # prompt length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true",
                        help="stream tokens as they are generated")
    args = parser.parse_args()

    question = textwrap.dedent(r"""
        There are three cards with letters $\texttt{a}$, $\texttt{b}$, $\texttt{c}$ placed in a row in some order. You can do the following operation at most once: 

 
        -  Pick two cards, and swap them.  Is it possible that the row becomes $\texttt{abc}$ after the operation? Output "YES" if it is possible, and "NO" otherwise.

        Input

        The first line contains a single integer $t$ ($1 \leq t \leq 6$) — the number of test cases.

        The only line of each test case contains a single string consisting of each of the three characters $\texttt{a}$, $\texttt{b}$, and $\texttt{c}$ exactly once, representing the cards.

        Output

        For each test case, output "YES" if you can make the row $\texttt{abc}$ with at most one operation, or "NO" otherwise.

        You can output the answer in any case (for example, the strings "yEs", "yes", "Yes" and "YES" will be recognized as a positive answer).Sample Input 1:
        6

        abc

        acb

        bac

        bca

        cab

        cba



        Sample Output 1:

        YES
        YES
        YES
        NO
        NO
        YES


        Note

        In the first test case, we don't need to do any operations, since the row is already $\texttt{abc}$.

        In the second test case, we can swap $\texttt{c}$ and $\texttt{b}$: $\texttt{acb} \to \texttt{abc}$.

        In the third test case, we can swap $\texttt{b}$ and $\texttt{a}$: $\texttt{bac} \to \texttt{abc}$.

        In the fourth test case, it is impossible to make $\texttt{abc}$ using at most one operation.

        ### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
        ```python
        # YOUR CODE HERE
        ```
    """).strip()

    tok, model = load_model()
    prompt = build_prompt(tok, question)

    print("\n--- PROMPT (first 300 chars) ---")
    print(prompt)#[:300] + (" …" if len(prompt) > 300 else ""))
    print()

    # generation cfg
    gen_cfg = dict(
        max_new_tokens=5000,
        temperature=0.0,   # deterministic
        top_p=0.9,
        do_sample=False,
    )

    gen_inputs, prompt_len = prepare_inputs(tok, prompt)

    #  generate
    if args.stream:
        streamer = TextIteratorStreamer(tok,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        _ = model.generate(**gen_inputs, streamer=streamer, **gen_cfg)

        for token in streamer:
            sys.stdout.write(token)
            sys.stdout.flush()
        print()
    else:
        with torch.no_grad():
            output = model.generate(**gen_inputs, **gen_cfg)

        completion = tok.decode(
            output[0, prompt_len:],
            skip_special_tokens=True,
        )
        print("--- COMPLETION ---")
        print(completion)


if __name__ == "__main__":
    main()
