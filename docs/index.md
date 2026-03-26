# PawBench

```text
       /\_/\
      ( o.o )   PawBench
       > ^ <    4-dimensional LLM inference benchmark
      /|   |\   "More bark than bite"
     (_|   |_)
```

**Because your model deserves a benchmark with more bark than bite.**

PawBench is a 4-dimensional LLM inference benchmark for OpenAI-compatible endpoints. It tests your model with realistic coding agent workloads -- multi-turn, multi-agent, parallel dispatch with tool calling -- not synthetic single-turn completions.

## Why PawBench?

Most LLM benchmarks measure single-turn completion speed or academic task accuracy. Real-world inference workloads look nothing like that. A coding assistant handles multi-turn conversations, calls tools, processes injected results, and adapts when requirements change mid-task.

PawBench measures what matters for production inference:

- **Throughput** -- How fast can your endpoint serve tokens under realistic multi-agent load?
- **Quality** -- Does the model follow instructions, call the right tools, and produce correct output?
- **Efficiency** -- How much of the output is useful code vs filler preamble?
- **Adaptability** -- Can the model handle mid-conversation steering events and cross-agent nudges?

## Meet Lola

PawBench is inspired by **Lola** ([@_justlolathings](https://www.instagram.com/_justlolathings/)) -- the most fashionable pup on Instagram. The built-in scenarios revolve around building her boutique dog apparel store, **PawStyle by Lola**. Every product, every size guide, every "Lola's Pick" badge traces back to this style icon on four legs.

Follow Lola: [https://www.instagram.com/_justlolathings/](https://www.instagram.com/_justlolathings/)

## Quick Start

```bash
pip install pawbench

# Benchmark your local vLLM
pawbench --endpoint http://localhost:8000

# Against any OpenAI-compatible endpoint
pawbench --endpoint https://api.openai.com/v1 --tag gpt4o
```

See the [Getting Started](getting-started.md) guide for detailed installation and usage instructions.

## What It Measures

| Dimension | Metrics |
|---|---|
| **Throughput** | Single-agent tok/s, parallel saturation curve (1->N), TTFT, peak concurrency |
| **Quality** | Tool call accuracy, instruction following, format compliance, keyword matching |
| **Efficiency** | Useful token ratio (code in tool args vs filler preamble), tokens per turn |
| **Adaptability** | Steering event response, mid-conversation context injection, nudge quality delta |

Learn more about each dimension in the [Dimensions](dimensions.md) guide.

## Server Metrics (optional)

If the endpoint exposes `/metrics` (vLLM, TGI), PawBench scrapes:

- KV cache usage and prefix cache hit rate
- Speculative decoding acceptance rate
- GPU cache pressure

## License

MIT
