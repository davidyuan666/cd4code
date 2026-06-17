"""CD4Code: LLM Code Generator using DeepSeek API."""
import os
import json
import time
import httpx
from openai import OpenAI
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS, HTTP_PROXY,
    PRICE_INPUT_PER_1M, PRICE_OUTPUT_PER_1M,
)


class TokenTracker:
    def __init__(self):
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0

    def record(self, usage):
        if usage is None:
            return
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        self.api_calls += 1
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.cost_usd += (prompt_tokens / 1e6) * PRICE_INPUT_PER_1M
        self.cost_usd += (completion_tokens / 1e6) * PRICE_OUTPUT_PER_1M

    def snapshot(self):
        return {
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }

    def reset(self):
        self.__init__()


def create_client():
    http_client = None
    if HTTP_PROXY:
        http_client = httpx.Client(proxy=HTTP_PROXY)
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        http_client=http_client,
    )


def generate_code(prompt, temperature=DEFAULT_TEMPERATURE,
                  top_p=DEFAULT_TOP_P, max_tokens=DEFAULT_MAX_TOKENS,
                  n=1, client=None, token_tracker=None):
    if client is None:
        client = create_client()

    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Write clean, correct, well-documented code. Return ONLY the Python code, no explanations."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
        )
        if token_tracker is not None:
            token_tracker.record(response.usage)
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"API error: {e}")
        return [""] * n


def generate_with_retry(prompt, max_retries=3, client=None):
    for attempt in range(max_retries):
        try:
            codes = generate_code(prompt, n=1, client=client)
            if codes and codes[0].strip():
                return codes[0]
        except Exception as e:
            print(f"Retry {attempt + 1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)
    return ""


def perturb_prompt(prompt, perturb_rate=0.3):
    import random
    words = prompt.split(' ')
    if len(words) < 5:
        return prompt
    random.seed(42)
    n_perturb = max(1, int(len(words) * perturb_rate))
    indices = random.sample(range(len(words)), n_perturb)
    for i in indices:
        if random.random() < 0.5 and len(words[i]) > 1:
            chars = list(words[i])
            j, k = random.sample(range(len(chars)), 2)
            chars[j], chars[k] = chars[k], chars[j]
            words[i] = ''.join(chars)
        else:
            words[i] = words[i].upper() if random.random() < 0.5 else words[i].lower()
    return ' '.join(words)


def load_humaneval(path=None):
    if path is None:
        from config import HUMANEVAL_PATH
        path = HUMANEVAL_PATH
    problems = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                problems.append(json.loads(line.strip()))
    return problems


def load_mbpp(path=None, n_samples=None):
    if path is None:
        from config import MBPP_PATH
        path = MBPP_PATH
    if n_samples is None:
        from config import MBPP_DEFAULT_SAMPLES
        n_samples = MBPP_DEFAULT_SAMPLES
    import random
    from config import MBPP_RANDOM_SEED

    problems = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                problems.append(json.loads(line.strip()))

    sample_info = {
        "total_available": len(problems),
        "n_samples": min(n_samples, len(problems)),
        "random_seed": MBPP_RANDOM_SEED,
        "sampled_ids": None,
    }

    if len(problems) > n_samples:
        random.seed(MBPP_RANDOM_SEED)
        all_ids = [p.get("task_id", str(i)) for i, p in enumerate(problems)]
        indices = random.sample(range(len(problems)), n_samples)
        problems = [problems[i] for i in indices]
        sample_info["sampled_ids"] = [all_ids[i] for i in indices]

    return problems, sample_info
