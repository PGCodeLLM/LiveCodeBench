try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError:
    pass

from lcb_runner.runner.base_runner import BaseRunner


SYSTEM_MSG = "You are a helpful coding assistant."



class VLLMRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)

        model_path = model.model_name if args.local_model_path is None else args.local_model_path

        # tokenizer is now needed for the chat template
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=args.trust_remote_code
        )

        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            enable_prefix_caching=args.enable_prefix_caching,
            trust_remote_code=args.trust_remote_code,
        )

        self.sampling_params = SamplingParams(
            n=args.n,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["### Explanation"]
        )

        # In case we need a generic system prompt
        self.system_prompt = getattr(args, "system_prompt", "You are a helpful coding assistant.")

    def _build_chat_prompt(self, user_content: str) -> str:
        # messages = [
        #     {"role": "system",    "content": SYSTEM_MSG},
        #     {"role": "user",      "content": user_content},
        #     {"role": "assistant", "content": "\n<answer>\n```python\n"},
        # ]


        # return self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=False,   # last assistant turn already present
        # )
        
        if isinstance(user_content, str):
            return f"{user_content}"

        # list[dict] → chat models (Seed-Coder Instruct / Reasoning, Llama-3-Ins…)
        return self.tokenizer.apply_chat_template(
            user_content,
            tokenize=False,
            add_generation_prompt=True,   # inserts first <assistant>
        )

    def _run_single(self, prompt: str) -> list[str]:
        return self.run_batch([prompt])[0]

    def run_batch(self, raw_prompts: list[str]) -> list[list[str]]:
        prompts = [self._build_chat_prompt(p) for p in raw_prompts]

        outputs: list[list[str]] = [None] * len(prompts)
        todo, todo_idx = [], []

        for i, p in enumerate(prompts):
            if self.args.use_cache and p in self.cache and len(self.cache[p]) == self.args.n:
                outputs[i] = self.cache[p]
            else:
                todo.append(p)
                todo_idx.append(i)

        if todo:
            print(f"Running {len(todo)} prompts in batch...Sneak peek:\n {todo[0]}")
            print("--------------------------------------------------------------")
            results = self.llm.generate(todo, self.sampling_params)
            for i, p, res in zip(todo_idx, todo, results):
                texts = [o.text for o in res.outputs]
                cleaned = []
                for t in texts:
                    # -- keep only the first fenced block, drop a leading
                    #     “<assistant>” (and sentinel tokens) if it sneaks in
                    body = t.split("<answer>")[-1]              # after <answer>
                    body = body.split("</answer>", 1)[0]        # before closing
                    if body.startswith("```python"):
                        code = body[len("```python"):].lstrip()
                        if code.startswith("<assistant>"):
                            code = code[len("<assistant>"):]
                        code = code.lstrip()  
                        body = "```python\n" + code
                    cleaned.append(body.rstrip() + "\n```\n</answer>")
    
                if self.args.use_cache:
                    self.cache[p] = cleaned
                outputs[i] = cleaned

        return outputs
