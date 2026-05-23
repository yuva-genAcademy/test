"""
Side-by-side comparison: DFlash speculative decoding vs normal inference.

Runs a built-in suite of prompts covering different length/complexity profiles,
then writes a markdown report: compare_report.md

Usage:
    python compare.py                          # full suite, default model
    python compare.py --suite                  # same as above (explicit)
    python compare.py --prompt "Your prompt"   # single custom prompt
    python compare.py --model Qwen/Qwen3.5-4B --draft z-lab/Qwen3.5-4B-DFlash
    python compare.py --max-tokens 512 --temperature 0.6
    python compare.py --report my_report.md    # custom report filename
"""

import argparse
import datetime
import time
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Prompt suite
# Each entry explains *why* it was chosen so the report is self-documenting.
# ---------------------------------------------------------------------------
@dataclass
class TestPrompt:
    label: str          # short name shown in report headings
    category: str       # e.g. "short", "long", "code", "math", "reasoning"
    rationale: str      # why this prompt was chosen
    text: str           # the actual prompt
    max_tokens: int     # expected output length drives this

PROMPT_SUITE: List[TestPrompt] = [
    TestPrompt(
        label="One-liner factual",
        category="short / factual",
        rationale=(
            "Ultra-short prompts with a single-token-ish answer test the "
            "overhead floor of speculative decoding. If DFlash has per-block "
            "setup cost, it will show here as a slowdown or near-zero speedup."
        ),
        text="What is the capital of France?",
        max_tokens=32,
    ),
    TestPrompt(
        label="Short creative",
        category="short / creative",
        rationale=(
            "A short creative task produces a moderate, varied vocabulary output "
            "(20-60 tokens). Checks whether draft acceptance holds on open-ended "
            "text where the next token is less predictable than factual recall."
        ),
        text="Write a haiku about machine learning.",
        max_tokens=64,
    ),
    TestPrompt(
        label="Medium explanation",
        category="medium / explanation",
        rationale=(
            "A mid-length explanation (~150-300 tokens) with structured prose. "
            "Good balance point — DFlash typically shines here because the model "
            "produces fluent, repetitive sentence patterns that the draft can "
            "predict well in blocks."
        ),
        text="Explain how transformers work in deep learning. Keep it concise but complete.",
        max_tokens=300,
    ),
    TestPrompt(
        label="Code generation",
        category="medium / code",
        rationale=(
            "Code has highly repetitive token patterns (indentation, keywords, "
            "punctuation) which are easy for block diffusion to predict. "
            "This category typically shows the highest DFlash speedup ratio."
        ),
        text=(
            "Write a Python function that implements a binary search tree with "
            "insert, search, and in-order traversal methods. Include docstrings."
        ),
        max_tokens=512,
    ),
    TestPrompt(
        label="Step-by-step math",
        category="medium / math",
        rationale=(
            "Chain-of-thought math forces the model to produce structured "
            "intermediate steps. Tests whether DFlash preserves reasoning "
            "correctness while drafting multi-token arithmetic sequences."
        ),
        text=(
            "Solve step by step: A train leaves city A at 60 km/h. Another leaves "
            "city B (300 km away) toward A at 90 km/h at the same time. "
            "Where do they meet, and when?"
        ),
        max_tokens=300,
    ),
    TestPrompt(
        label="Long structured essay",
        category="long / structured",
        rationale=(
            "Long outputs (500+ tokens) stress-test sustained speedup. "
            "DFlash's advantage can compound over many blocks, but acceptance "
            "rate may drift as the response grows. This measures consistency."
        ),
        text=(
            "Write a detailed comparison of microservices vs monolithic architecture. "
            "Cover: definition, pros/cons, use cases, migration strategy, and a "
            "recommendation for a 10-person startup. Use headers and bullet points."
        ),
        max_tokens=700,
    ),
    TestPrompt(
        label="Long code with explanation",
        category="long / code + prose",
        rationale=(
            "Mixed code-and-prose output (comments, docstrings, explanation) "
            "combines two token distributions. Tests whether DFlash handles "
            "mode-switching between natural language and code mid-response."
        ),
        text=(
            "Implement a thread-safe LRU cache in Python using OrderedDict. "
            "Include full implementation, explain each design decision inline, "
            "and show example usage with 3 test cases."
        ),
        max_tokens=700,
    ),
    TestPrompt(
        label="Complex multi-constraint reasoning",
        category="long / logic",
        rationale=(
            "The original stress-test prompt added by the user. Dense constraint "
            "satisfaction with logical deduction chains produces long, highly "
            "structured output. Worst-case for draft acceptance since each token "
            "depends heavily on prior deductive state."
        ),
        text="""\
Multi-Agent Temporal Reasoning Stress Test

You are controlling a research station on Europa with 7 autonomous agents:
Atlas (engineering), Nova (medical), Echo (communications), Vega (navigation),
Orion (security), Luna (science), Helios (energy systems).

Each agent has exactly one specialization, works exactly one shift (Morning/Afternoon/Night),
and lives in exactly one sector (A-G). No two agents share the same specialization, shift, or sector.

Constraints:
- Atlas does not work Night shift.
- The Medical specialist lives adjacent to the Security specialist.
- Sector A and Sector G are connected only through Sector D.
- Vega lives in neither Sector A nor Sector G.
- The Communications specialist works immediately before the Science specialist in shift order.
- Helios is not the Energy specialist.
- Orion works later than Nova.
- Luna lives in a sector with an even ASCII distance from Atlas's sector letter.
- The Night shift agent lives alphabetically after the Morning shift agent.
- Atlas and the Engineering specialist are different people.
- The saboteur is not in Security.
- The Energy specialist works Afternoon shift.
- The person in Sector D works Afternoon shift.
- The Navigation specialist lives alphabetically before the Communications specialist.
- The Engineering specialist lives in a sector whose letter value (A=1, B=2...) is prime.
- Atlas's sector letter value plus Nova's equals 8.
- The saboteur lives in a sector whose letter value is Fibonacci.
- Vega said: "Helios works Night shift."
- Helios said: "Nova is the Medical specialist."
- Atlas said: "Orion is lying."
- If Atlas is truthful, Orion is the saboteur.

Determine the complete assignment and explain every deduction step-by-step without brute force.""",
        max_tokens=900,
    ),
]


# ---------------------------------------------------------------------------
# Core runners
# ---------------------------------------------------------------------------

def run_normal(model, tokenizer, formatted_prompt, max_tokens, sampler, silent=False):
    from mlx_lm import stream_generate

    if not silent:
        console.print(Rule("[bold blue]Normal inference[/bold blue]"))
    output = []
    tps = 0.0
    t0 = time.perf_counter()

    for r in stream_generate(model, tokenizer, formatted_prompt, max_tokens, sampler=sampler):
        output.append(r.text)
        tps = r.generation_tps
        if not silent:
            console.print(r.text, end="", highlight=False)

    elapsed = time.perf_counter() - t0
    total_tokens = len(tokenizer.encode("".join(output)))
    if not silent:
        console.print()
    return "".join(output), tps, elapsed, total_tokens


def run_dflash(model, draft, tokenizer, formatted_prompt, max_tokens, sampler, silent=False):
    from dflash.model_mlx import stream_generate

    if not silent:
        console.print(Rule("[bold green]DFlash (speculative)[/bold green]"))
    output = []
    accepted_lengths = []
    tps = 0.0
    t0 = time.perf_counter()

    for r in stream_generate(model, draft, tokenizer, formatted_prompt, max_tokens=max_tokens, sampler=sampler):
        output.append(r.text)
        accepted_lengths.append(r.accepted)
        tps = r.generation_tps
        if not silent:
            console.print(r.text, end="", highlight=False)

    elapsed = time.perf_counter() - t0
    total_tokens = len(tokenizer.encode("".join(output)))
    if not silent:
        console.print()
    return "".join(output), tps, elapsed, accepted_lengths, total_tokens


def print_prompt_summary(normal_tps, normal_elapsed, normal_tokens,
                         dflash_tps, dflash_elapsed, dflash_tokens, accepted,
                         temperature):
    avg_accepted = sum(accepted) / len(accepted) if accepted else 0
    speedup = dflash_tps / normal_tps if normal_tps > 0 else 0

    normal_stats = Text.assemble(
        ("Throughput:  ", "dim"), (f"{normal_tps:.1f} tok/s\n", "bold"),
        ("Time:        ", "dim"), (f"{normal_elapsed:.1f}s\n", "bold"),
        ("Tokens out:  ", "dim"), (f"{normal_tokens}", "bold"),
    )
    dflash_stats = Text.assemble(
        ("Throughput:  ", "dim"), (f"{dflash_tps:.1f} tok/s\n", "bold"),
        ("Time:        ", "dim"), (f"{dflash_elapsed:.1f}s\n", "bold"),
        ("Tokens out:  ", "dim"), (f"{dflash_tokens}\n", "bold"),
        ("Avg accepted:", "dim"), (f" {avg_accepted:.2f} tok/block\n", "bold"),
        ("Speedup:     ", "dim"), (f"{speedup:.2f}x", "bold green" if speedup > 1 else "bold red"),
    )
    console.print(
        Columns([
            Panel(normal_stats, title="[blue]Normal[/blue]", expand=True),
            Panel(dflash_stats, title="[green]DFlash[/green]", expand=True),
        ])
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    prompt: TestPrompt
    normal_tps: float
    normal_elapsed: float
    normal_tokens: int
    dflash_tps: float
    dflash_elapsed: float
    dflash_tokens: int
    avg_accepted: float
    speedup: float
    outputs_match: bool


def write_report(results: List[PromptResult], model_id: str, draft_id: str,
                 temperature: float, report_path: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append("# DFlash vs Normal Inference — Benchmark Report\n")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Model:** `{model_id}`  ")
    lines.append(f"**Draft:** `{draft_id}`  ")
    lines.append(f"**Temperature:** {temperature}  \n")

    # --- Overall summary table ---
    lines.append("## Overall Results\n")
    lines.append("| Prompt | Category | Normal (tok/s) | DFlash (tok/s) | Speedup | Avg accepted | Outputs match |")
    lines.append("|--------|----------|---------------|----------------|---------|--------------|---------------|")
    for r in results:
        match_str = "yes" if r.outputs_match else "no" if temperature == 0.0 else "n/a"
        lines.append(
            f"| {r.prompt.label} | {r.prompt.category} "
            f"| {r.normal_tps:.1f} | {r.dflash_tps:.1f} "
            f"| **{r.speedup:.2f}x** | {r.avg_accepted:.2f} | {match_str} |"
        )

    avg_speedup = sum(r.speedup for r in results) / len(results) if results else 0
    best = max(results, key=lambda r: r.speedup)
    worst = min(results, key=lambda r: r.speedup)
    lines.append(f"\n**Average speedup across all prompts: {avg_speedup:.2f}x**  ")
    lines.append(f"**Best speedup:** {best.prompt.label} ({best.speedup:.2f}x)  ")
    lines.append(f"**Lowest speedup:** {worst.prompt.label} ({worst.speedup:.2f}x)  \n")

    # --- Per-prompt detail ---
    lines.append("---\n")
    lines.append("## Per-Prompt Detail\n")
    for r in results:
        lines.append(f"### {r.prompt.label}\n")
        lines.append(f"**Category:** {r.prompt.category}  ")
        lines.append(f"**Why this prompt was chosen:**  ")
        lines.append(f"> {r.prompt.rationale}\n")
        lines.append("| Metric | Normal | DFlash |")
        lines.append("|--------|--------|--------|")
        lines.append(f"| Throughput (tok/s) | {r.normal_tps:.1f} | {r.dflash_tps:.1f} |")
        lines.append(f"| Time (s) | {r.normal_elapsed:.1f} | {r.dflash_elapsed:.1f} |")
        lines.append(f"| Tokens generated | {r.normal_tokens} | {r.dflash_tokens} |")
        lines.append(f"| Avg accepted tok/block | — | {r.avg_accepted:.2f} |")
        lines.append(f"| Speedup | — | **{r.speedup:.2f}x** |")
        if temperature == 0.0:
            lines.append(f"| Outputs match | {'yes' if r.outputs_match else 'no'} | |")
        lines.append("")

    # --- Interpretation ---
    lines.append("---\n")
    lines.append("## Interpretation Guide\n")
    lines.append(
        "- **Speedup > 1.5x** — DFlash draft model is a strong fit for this prompt type; "
        "block predictions are accepted frequently.\n"
        "- **Speedup 1.0–1.5x** — Modest gain; draft acceptance is partial. "
        "Still beneficial but not the sweet spot.\n"
        "- **Speedup < 1.0x** — Draft overhead exceeds benefit. "
        "Typically happens on very short outputs or highly unpredictable token sequences.\n"
        "- **Avg accepted** — How many tokens per block the verifier accepts on average. "
        "Higher = better draft quality for that prompt type.\n"
        "- **Outputs match** — At temperature=0 both approaches should produce identical text "
        "(DFlash is lossless). A mismatch indicates a bug.\n"
    )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"\n[bold green]Report saved →[/bold green] {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="Qwen/Qwen3.5-4B")
    p.add_argument("--draft",        default="z-lab/Qwen3.5-4B-DFlash")
    p.add_argument("--prompt",       default=None, help="Single custom prompt (skips the suite)")
    p.add_argument("--suite",        action="store_true", help="Run the full built-in prompt suite (default when --prompt is omitted)")
    p.add_argument("--max-tokens",   type=int,   default=None, help="Override max tokens (suite uses per-prompt defaults)")
    p.add_argument("--temperature",  type=float, default=0.0)
    p.add_argument("--report",       default="compare_report.md")
    return p.parse_args()


def main():
    args = parse_args()

    from mlx_lm.sample_utils import make_sampler
    from dflash.model_mlx import load, load_draft

    sampler = make_sampler(temp=args.temperature)

    console.print(f"\n[bold]Loading model:[/bold] {args.model}")
    model, tokenizer = load(args.model)
    console.print(f"[bold]Loading draft:[/bold]  {args.draft}")
    draft = load_draft(args.draft)

    # Warmup
    console.print("\n[dim]Warming up...[/dim]")
    warmup = tokenizer.encode("Hi")
    from mlx_lm import stream_generate as _bl
    list(_bl(model, tokenizer, warmup, 5, sampler=sampler))
    from dflash.model_mlx import stream_generate as _df
    list(_df(model, draft, tokenizer, warmup, max_tokens=5, sampler=sampler))

    # Decide what to run
    if args.prompt:
        # Single custom prompt — no report
        console.print(f"\n[bold]Prompt:[/bold] {args.prompt}\n")
        messages = [{"role": "user", "content": args.prompt}]
        fmt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        max_tok = args.max_tokens or 512

        normal_text, normal_tps, normal_elapsed, normal_tokens = run_normal(model, tokenizer, fmt, max_tok, sampler)
        console.print()
        dflash_text, dflash_tps, dflash_elapsed, accepted, dflash_tokens = run_dflash(model, draft, tokenizer, fmt, max_tok, sampler)

        console.print()
        console.print(Rule("[bold]Summary[/bold]"))
        print_prompt_summary(normal_tps, normal_elapsed, normal_tokens,
                             dflash_tps, dflash_elapsed, dflash_tokens,
                             accepted, args.temperature)
        if args.temperature == 0.0:
            match = normal_text.strip() == dflash_text.strip()
            console.print(f"\n[dim]Outputs match:[/dim] {'[green]yes[/green]' if match else '[yellow]no[/yellow]'}")
        return

    # --- Full suite ---
    results: List[PromptResult] = []
    total = len(PROMPT_SUITE)

    for i, tp in enumerate(PROMPT_SUITE, 1):
        console.print(f"\n[bold]━━ Prompt {i}/{total}: {tp.label}[/bold]  [dim]{tp.category}[/dim]")
        console.print(f"[dim italic]{tp.rationale}[/dim italic]\n")

        messages = [{"role": "user", "content": tp.text}]
        fmt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        max_tok = args.max_tokens or tp.max_tokens

        console.print(Rule("[bold blue]Normal[/bold blue]"))
        normal_text, normal_tps, normal_elapsed, normal_tokens = run_normal(
            model, tokenizer, fmt, max_tok, sampler
        )
        console.print()

        console.print(Rule("[bold green]DFlash[/bold green]"))
        dflash_text, dflash_tps, dflash_elapsed, accepted, dflash_tokens = run_dflash(
            model, draft, tokenizer, fmt, max_tok, sampler
        )

        avg_accepted = sum(accepted) / len(accepted) if accepted else 0
        speedup = dflash_tps / normal_tps if normal_tps > 0 else 0

        console.print()
        console.print(Rule("[bold]Prompt summary[/bold]"))
        print_prompt_summary(normal_tps, normal_elapsed, normal_tokens,
                             dflash_tps, dflash_elapsed, dflash_tokens,
                             accepted, args.temperature)

        results.append(PromptResult(
            prompt=tp,
            normal_tps=normal_tps,
            normal_elapsed=normal_elapsed,
            normal_tokens=normal_tokens,
            dflash_tps=dflash_tps,
            dflash_elapsed=dflash_elapsed,
            dflash_tokens=dflash_tokens,
            avg_accepted=avg_accepted,
            speedup=speedup,
            outputs_match=(normal_text.strip() == dflash_text.strip()),
        ))

    # Final aggregate table
    console.print()
    console.print(Rule("[bold]Final Report[/bold]"))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Prompt", style="dim")
    table.add_column("Category")
    table.add_column("Normal tok/s", justify="right")
    table.add_column("DFlash tok/s", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Avg accepted", justify="right")

    for r in results:
        speedup_str = f"{r.speedup:.2f}x"
        style = "green" if r.speedup >= 1.5 else ("yellow" if r.speedup >= 1.0 else "red")
        table.add_row(
            r.prompt.label, r.prompt.category,
            f"{r.normal_tps:.1f}", f"{r.dflash_tps:.1f}",
            f"[{style}]{speedup_str}[/{style}]",
            f"{r.avg_accepted:.2f}",
        )

    avg_speedup = sum(r.speedup for r in results) / len(results)
    table.add_row("", "[bold]AVERAGE[/bold]", "", "", f"[bold]{avg_speedup:.2f}x[/bold]", "")
    console.print(table)

    write_report(results, args.model, args.draft, args.temperature, args.report)


if __name__ == "__main__":
    main()
