# -*- coding: utf-8 -*-

# !pip install --quiet auto-gptq transformers accelerate

import torch

if not torch.cuda.is_available():
    raise EnvironmentError("🚫 GPU not available. Please enable GPU from Runtime → Change runtime type.")

device = torch.device("cuda")
print("✅ CUDA is available. Device:", torch.cuda.get_device_name(0))

from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM

model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(
    model_id,
    use_safetensors=True,
    trust_remote_code=True,
    device_map="auto",  # ✅ Works well with Colab GPU
    use_triton=False,
    inject_fused_attention=False
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def python_exec(code: str):
    try:
        return str(eval(code, {"__builtins__": {}}))
    except Exception as e:
        return f"[Error]: {e}"

def noop(_): return None

def build_prompt(user_input):
    return f"""You are a helpful assistant that uses tools.

You can call:
- python.exec("...") → runs Python code
- noop("...") → when no code is needed

Always handle ONLY the most recent user query.
Do not invent additional turns.

Format:
TOOL: tool_name("argument")

Example:
User: How many 'a' are in 'banana'?
Assistant:
TOOL: python.exec("len([c for c in 'banana' if c == 'a'])")

Now handle this:
User: {user_input}
Assistant:"""

def call_model(prompt, max_new_tokens=100):
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return output[0]['generated_text'].strip()

import re

import re

def extract_tool_call(response):
    # Strip out everything before TOOL:
    if "TOOL:" in response:
        tool_line = response.split("TOOL:")[-1].strip()
        match = re.match(r"(\w+)\((['\"])(.*?)\2\)", tool_line)
        if match:
            tool_name, _, arg = match.groups()
            return tool_name, arg
    return "noop", ""

def run_agent(user_input):
    print(f"\n🧑 User: {user_input}")
    prompt = build_prompt(user_input)
    raw_response = call_model(prompt)

    print(f"\n🤖 LLM Response:\n{raw_response}")

    tool, arg = extract_tool_call(raw_response)
    print(f"\n🔧 Tool Used: {tool}('{arg}')")

    if tool == "python.exec":
        result = python_exec(arg)
        final = f"The answer is: {result}"
    elif tool == "noop":
        final = raw_response.split("TOOL:")[0].strip()
    else:
        final = "[❌ Unknown tool]"

    print(f"\n✅ Final Reply:\n{final}")

examples = [
    "How many 'r' are there in 'strawberry'?",
    "How often does 'the' appear in 'the theater is near the mall'?",
    "Count digits in 'abc12345xyz'.",
    "What is 4 * (5 + 2)?",
    "If speed is 80 and time is 1.5, what's the distance?"
]

for q in examples:
    run_agent(q)
    print("\n" + "-"*60)