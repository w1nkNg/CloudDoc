```python

import os
os.environ['HTTP_PROXY'] = 'http://192.168.2.170:7890'
os.environ['HTTPS_PROXY'] = 'http://192.168.2.170:7890'

# Load model directly
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set the random seed for reproducibility
torch.random.manual_seed(0)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Define the input messages
messages = [
    # {"role": "system", "content": "You are a helpful AI assistant."},
    # {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    # {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving a 2x + 3 = 7 equation?"}
]

# Initialize the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define generation arguments
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# Measure the time taken for generation
start_time = time.time()
output = pipe(messages, **generation_args)
end_time = time.time()

# Calculate the number of tokens generated
generated_text = output[0]['generated_text']
num_tokens = len(tokenizer.tokenize(generated_text))

# Calculate and print the tokens per second
time_taken = end_time - start_time
tokens_per_second = num_tokens / time_taken
print(f"Generated text: {generated_text}")
print(f"Time taken: {time_taken:.2f} seconds")
print(f"Tokens generated: {num_tokens}")
print(f"Tokens per second: {tokens_per_second:.2f}")



# import torch
# # x = torch.rand(5, 3)
# # print(x)

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

# prompt = "GPT2 is a model developed by OpenAI. How does it actually work?"

# # 将模型转移到CUDA设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

# gen_tokens = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=0.9,
#     max_length=500,
# )

# gen_text = tokenizer.batch_decode(gen_tokens)[0]

# print(gen_text)

####################################################################################################
#
# CPU Inference
# Generated text:  To solve the equation 2x + 3 = 7, follow these steps:

# Step 1: Isolate the term with the variable (2x) by subtracting 3 from both sides of the equation.

# 2x + 3 - 3 = 7 - 3

# This simplifies to:

# 2x = 4

# Step 2: Solve for x by dividing both sides of the equation by 2.

# 2x / 2 = 4 / 2

# This simplifies to:

# x = 2

# So, the solution to the equation 2x + 3 = 7 is x = 2.
# Time taken: 394.35 seconds
# Tokens generated: 150
# Tokens per second: 0.38
#
#
####################################################################################################
# GPU Inference
# Generated text:  To solve the equation 2x + 3 = 7, follow these steps:

# 1. Subtract 3 from both sides of the equation: 2x + 3 - 3 = 7 - 3
# 2. Simplify: 2x = 4
# 3. Divide both sides by 2: 2x/2 = 4/2
# 4. Simplify: x = 2

# The solution to the equation 2x + 3 = 7 is x = 2.
# Time taken: 5.55 seconds
# Tokens generated: 117
# Tokens per second: 21.08


# Generated text:  Certainly! Bananas and dragonfruits can be combined in various delicious ways. Here are some creative ideas to enjoy these fruits together:

# 1. Banana and Dragonfruit Smoothie:
#    - Blend together 1 ripe banana, 1/2 cup of dragon fruit, 1 cup of almond milk, and a handful of ice cubes.
#    - Add a tablespoon of honey or agave syrup for sweetness, if desired.
#    - Blend until smooth and enjoy a refreshing and nutritious smoothie.

# 2. Banana and Dragonfruit Salad:
#    - Slice 1 ripe banana and 1/2 cup of dragon fruit into bite-sized pieces.
#    - Toss the fruit with a handful of mixed greens, such as baby spinach or arugula.
#    - Add a sprinkle of chopped nuts, like almonds or walnuts, for extra crunch.
#    - Drizzle with a simple dressing made from olive oil, lemon juice, and a pinch of salt and pepper.

# 3. Banana and Dragonfruit Salsa:
#    - Dice 1 ripe banana and 1/2 cup of dragon fruit into small cubes.
#    - Combine the fruit with 1/2 cup of diced tomatoes, 1/4 cup of chopped red onion, 1 tablespoon of chopped cilantro, and the juice of 1 lime.
#    - Season with salt and pepper to taste.
#    - Serve the salsa with tortilla chips or as a topping for grilled chicken or fish.

# 4. Banana and Dragonfruit Wrap:
#    - Serve the salsa with tortilla chips or as a topping for grilled chicken or fish.

# 4. Banana and Dragonfruit Wrap:

# 4. Banana and Dragonfruit Wrap:
# 4. Banana and Dragonfruit Wrap:
#    - Slice a ripe banana and 1/2 cup of dragon fruit into thin slices.
#    - Spread a layer of cream cheese or Greek yogurt on a whole wheat tortilla.
#    - Slice a ripe banana and 1/2 cup of dragon fruit into thin slices.
#    - Spread a layer of cream cheese or Greek yogurt on a whole wheat tortilla.
#    - Spread a layer of cream cheese or Greek yogurt on a whole wheat tortilla.
#    - Layer the banana and dragon fruit slices on top of the cream cheese.
#    - Layer the banana and dragon fruit slices on top of the cream cheese.
#    - Add a handful of spinach or arugula leaves.
#    - Add a handful of spinach or arugula leaves.
#    - Roll up the
#    - Roll up the
# Time taken: 23.22 seconds
# Time taken: 23.22 seconds
# Tokens generated: 501
# Tokens per second: 21.58
```
