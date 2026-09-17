from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import FlyVocabulary


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 1.0
    max_tokens: int = 96
    max_chars: int = 240

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_chars <= 0 or self.max_chars > 240:
            raise ValueError("max_chars must lie in [1, 240]")


@dataclass(frozen=True, slots=True)
class GeneratedTweet:
    text: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    seed: int
    termination_reason: str
    checkpoint_id: str
    git_sha: str
    batch_id: str
    tweet_index: int


_SPECIAL_RENDER = {
    "<UNK>": "?",
    "<MASK>": "?",
    "<SEP>": "|",
    "<URL>": "url",
    "<USER>": "@user",
    "<NUMBER>": "0",
    "<INTEGER>": "0",
    "<REAL>": "0",
    "<MATH>": "math",
    "<CODE>": "code",
    "<QUOTE>": '"',
    "<EMPH>": "!",
    "<OPEN>": "(",
    "<CLOSE>": ")",
    "<LIST>": "-",
    "<EQ>": "=",
    "<NEQ>": "!=",
    "<LT>": "<",
    "<GT>": ">",
    "<LE>": "<=",
    "<GE>": ">=",
    "<ARROW>": "->",
    "<IFF>": "<=>",
    "<AND>": "and",
    "<OR>": "or",
    "<NOT>": "not",
    "<TRUE>": "true",
    "<FALSE>": "false",
}


def render_token_ids(vocabulary: FlyVocabulary, token_ids: tuple[int, ...] | list[int]) -> str:
    raw_tokens = [vocabulary.token_for(int(token_id)) for token_id in token_ids]
    no_output = {"<BOS>", "<EOS>", "<PAD>"}
    no_left_space = {".", ",", "?", "!", ":", ";", ")", "]", "}"}
    no_right_space = {"(", "[", "{"}

    result = ""
    previous = ""
    for raw in raw_tokens:
        if raw in no_output:
            continue
        if raw == "<NL>":
            result = result.rstrip() + "\n"
            previous = raw
            continue
        token = _SPECIAL_RENDER.get(raw, raw)
        if token in no_left_space:
            result = result.rstrip() + token
        elif not result or result.endswith((" ", "\n")) or previous in no_right_space:
            result += token
        else:
            result += " " + token
        previous = token
    return result.strip()


def generate_tweet(
    policy: FlyTextPolicy,
    vocabulary: FlyVocabulary,
    *,
    seed: int,
    config: GenerationConfig,
    checkpoint_id: str,
    git_sha: str,
    batch_id: str,
    tweet_index: int,
) -> GeneratedTweet:
    device = policy.input_gain.device
    bos_id = vocabulary.id_for("<BOS>")
    eos_id = vocabulary.id_for("<EOS>")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    was_training = policy.training
    policy.eval()
    token_ids: list[int] = []
    token_logprobs: list[float] = []
    termination_reason = "token_cap"
    try:
        state = policy.initial_state(batch_size=1)
        current = bos_id
        with torch.no_grad():
            for _ in range(config.max_tokens):
                input_ids = torch.tensor([current], dtype=torch.long, device=device)
                output = policy.step(input_ids, state)
                if not bool(torch.isfinite(output.logits).all().item()) or not bool(
                    torch.isfinite(output.state).all().item()
                ):
                    raise FloatingPointError("non-finite fly state or logits during generation")

                probabilities = torch.softmax(
                    output.logits / config.temperature, dim=-1
                ).detach().cpu()[0]
                if not bool(torch.isfinite(probabilities).all().item()):
                    raise FloatingPointError("non-finite fly sampling probabilities")
                sampled = int(torch.multinomial(probabilities, 1, generator=generator).item())
                probability = float(probabilities[sampled].item())
                logprob = math.log(max(probability, torch.finfo(probabilities.dtype).tiny))

                if sampled == eos_id:
                    token_ids.append(sampled)
                    token_logprobs.append(logprob)
                    termination_reason = "eos"
                    break

                candidate = [*token_ids, sampled]
                rendered_candidate = render_token_ids(vocabulary, candidate)
                if len(rendered_candidate) > config.max_chars:
                    termination_reason = "char_cap"
                    break

                token_ids.append(sampled)
                token_logprobs.append(logprob)
                state = output.state
                current = sampled
    finally:
        policy.train(was_training)

    return GeneratedTweet(
        text=render_token_ids(vocabulary, token_ids),
        token_ids=tuple(token_ids),
        token_logprobs=tuple(token_logprobs),
        seed=int(seed),
        termination_reason=termination_reason,
        checkpoint_id=str(checkpoint_id),
        git_sha=str(git_sha),
        batch_id=str(batch_id),
        tweet_index=int(tweet_index),
    )


def generate_batch(
    policy: FlyTextPolicy,
    vocabulary: FlyVocabulary,
    *,
    base_seed: int,
    config: GenerationConfig,
    checkpoint_id: str,
    git_sha: str,
    batch_id: str,
    count: int = 10,
) -> list[GeneratedTweet]:
    if count != 10:
        raise ValueError("Fly Tweets v1 batches must contain exactly 10 tweets")
    return [
        generate_tweet(
            policy,
            vocabulary,
            seed=int(base_seed) + index,
            config=config,
            checkpoint_id=checkpoint_id,
            git_sha=git_sha,
            batch_id=batch_id,
            tweet_index=index,
        )
        for index in range(10)
    ]
