import openai
import asyncio
import json
from tqdm import tqdm


def generate_many(starts, engine, n, max_tokens, echo=False):
    async def generate(start, n):
        while True:
            try:
                return await openai.Completion.acreate(
                    engine=engine,
                    prompt=start,
                    logprobs=0,
                    n=n,
                    max_tokens=max_tokens,
                    temperature=1,
                    echo=echo,
                )
            except Exception as e:
                print(e)
                await asyncio.sleep(1)

    async def generate_one_prompt(start):
        ns = [128] * (n // 128) + [n % 128]
        r = await asyncio.gather(*[generate(start, n) for n in ns])
        return [c for c in r for c in c["choices"]]

    async def generate_all_prompts():
        return await asyncio.gather(*[generate_one_prompt(start) for start in starts])

    return asyncio.run(generate_all_prompts())


n_docs = 100
start_idx = 0
main_model = "davinci-002"
observer_model = "babbage-002"


start = "<|endoftext|>"
choices = generate_many(starts=[start], n=n_docs, max_tokens=100, engine=main_model)[0]

tokens_1 = [c["logprobs"]["tokens"] for c in choices]
lps_1 = [c["logprobs"]["token_logprobs"] for c in choices]

forbidden_substring = [R"\u", R"bytes:"]
valid_idxs = [
    i
    for i, choice in enumerate(choices)
    if not any(any(s in t for s in forbidden_substring) for t in choice["logprobs"]["tokens"])
]

forbidden_substring = [R"\u", R"bytes:"]
valid_idxs = [
    i
    for i, choice in enumerate(choices)
    if not any(any(s in t for s in forbidden_substring) for t in choice["logprobs"]["tokens"])
]
print(len(valid_idxs), "/", len(choices), "valid")

j = start_idx
for i in tqdm(valid_idxs):
    tokens = choices[i]["logprobs"]["tokens"]
    lps = choices[i]["logprobs"]["token_logprobs"]
    prefixes = [start + "".join(tokens[:t]) for t in range(0, len(tokens))]

    other_choices = generate_many(starts=prefixes, n=1, max_tokens=1, engine=main_model)
    try:
        other_tokens = [c[0]["logprobs"]["tokens"][0] for c in other_choices]
        other_lps = [c[0]["logprobs"]["token_logprobs"][0] for c in other_choices]

        prefix_with_main_choice = [p + t for p, t in zip(prefixes, tokens)]
        # print(prefix_with_main_choice[:10])
        observer_choices = generate_many(
            starts=prefix_with_main_choice, n=1, max_tokens=0, engine=observer_model, echo=True
        )
        # print([c[0]["logprobs"]["tokens"] for c in observer_choices[:10]])
        observer_lps = [c[0]["logprobs"]["token_logprobs"][-1] for c in observer_choices]

        prefix_other_choice = [p + t for p, t in zip(prefixes, other_tokens)]
        # print(prefix_other_choice[:10])
        observer_other_choices = generate_many(
            starts=prefix_other_choice, n=1, max_tokens=0, engine=observer_model, echo=True
        )
        # print([c[0]["logprobs"]["tokens"] for c in observer_other_choices[:10]])
        observer_other_lps = [c[0]["logprobs"]["token_logprobs"][-1] for c in observer_other_choices]

        z = zip(tokens, lps, observer_lps, other_tokens, other_lps, observer_other_lps)

        pairs = []
        for t, lp, oblp, ot, olp, obolp in z:
            pairs.append(
                [
                    {"t": t, "lp": lp, "olp": oblp},
                    {"t": ot, "lp": olp, "olp": obolp},
                ]
            )
        json.dump(pairs, open(f"pairs/{j}.json", "w"))
        j += 1
    except Exception as e:
        print(e)
        continue
# # %%
# print(r)
# # %%
# for c in r["choices"]:
#     print("========")
#     print(c["text"])
# # %%
# from tqdm import trange
# counts_d4 = 0
# seqs = []
# for _ in trange(100):
#     r = openai.Completion.create(
#         engine="babbage-002",
#         prompt="<|endoftext|>",
#         logprobs=0,
#         n=100,
#         max_tokens=100,
#         temperature=1,
#     )
#     for c in r["choices"]:
#         if " d4 " in c["text"] or " O-O " in c["text"] or " e4 " in c["text"]:
#             seqs.append(c["text"])
#             counts_d4 += 1
# print(counts_d4)
# # %%
# for seq in seqs:
#     print(seq)
#     print("========")
# # %%
# import numpy as np
# n = 100 * len(seqs)
# p = counts_d4 / n
# uncertainty =  np.sqrt(p * (1 - p) / n)
# print(f"{p:.3f} +/- {uncertainty:.3f}")
# # %%
# for c in r["choices"]:
#     print("========")
#     print(c["text"])

# # %%

# %%
