import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def script_basename(relative_path: str) -> str:
    return relative_path.replace("\\", "/").split("/")[-1]


def normalize_flag_list(arg_obj: Dict[str, Any]) -> List[str]:
    option_strings = arg_obj.get("option_strings")
    flags = [str(x) for x in _as_list(option_strings) if x is not None and str(x).strip()]
    cleaned: List[str] = []
    for f in flags:
        f = f.strip()
        if f == "-":
            continue
        cleaned.append(f)
    seen = set()
    out: List[str] = []
    for f in cleaned:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def format_arg_block(title: str, args: List[Dict[str, Any]]) -> str:
    lines = [f"{title}:"]
    if not args:
        lines.append("- (none)")
        return "\n".join(lines)
    for a in args:
        flags = normalize_flag_list(a)
        name = a.get("name")
        if not name or str(name).strip() in {"-", "None"}:
            name = flags[0] if flags else ""
        choices = _as_list(a.get("choices"))
        default = a.get("default", None)
        help_text = a.get("help", None)
        parts = [f"- {name}"]
        if flags and flags != [name]:
            parts.append(f"flags={flags}")
        if choices:
            parts.append(f"choices={choices}")
        if default is not None:
            parts.append(f"default={default!r}")
        if help_text:
            parts.append(f"help={str(help_text).strip()!r}")
        lines.append("  ".join(parts))
    return "\n".join(lines)


def build_prompt(entry: Dict[str, Any], examples_per_script: int) -> str:
    rel = str(entry.get("relative_path", "")).strip()
    script = script_basename(rel)
    desc = str(entry.get("description", "")).strip()
    required_args = _as_list(entry.get("required_args"))
    optional_args = _as_list(entry.get("optional_args"))

    required_block = format_arg_block("REQUIRED_ARGS (must appear in every output command)", required_args)
    optional_block = format_arg_block("OPTIONAL_ARGS (include only when instruction implies them)", optional_args)

    prompt = f"""You are generating instruction->command training data for a CARLA script launcher.

TARGET_SCRIPT:
{script}

SCRIPT_DESCRIPTION:
{desc if desc else "(no description provided)"}

{required_block}

{optional_block}

Task:
- Output EXACTLY {examples_per_script} JSONL lines.
- Each line must be exactly: {{"instruction":"<natural language request>","output":"<command>"}}
- The output command MUST start with exactly "{script}" (no python/python3, no paths, no repo names).
- You MUST NOT output any other script name.
- You MUST NOT invent flags; you may only use flags that appear above.
- You MUST include all REQUIRED_ARGS in every command.
- If an instruction omits a required value, choose a valid value (respect choices if present).
- Optional args should appear only if the instruction explicitly asks for them (otherwise omit them so defaults apply).
- Vary wording, tone, structure, and argument values across examples (NLP augmentation).
- Line 1 MUST be a simple baseline example: a short instruction matching SCRIPT_DESCRIPTION and an output with only required args (or just "{script}" if there are no required args).

Important: Return ONLY the {examples_per_script} JSONL lines. No headers, no numbering, no extra text.
"""
    return prompt


def chunk(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> int:
    ap = argparse.ArgumentParser(description="Create per-script augmentation prompts from carla_commands_full.json")
    ap.add_argument(
        "--input",
        default=r"D:\Datasets\carla_python_script_dataset\carla_commands_full.json",
        help="Path to carla_commands_full.json",
    )
    ap.add_argument(
        "--output",
        default=r"D:\Datasets\carla_python_script_dataset\augmentation_prompts.jsonl",
        help="Output JSONL (one prompt per entry)",
    )
    ap.add_argument("--examples-per-script", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1, help="How many script entries per prompt (default: 1).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list of entries.")

    records: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        rel = str(entry.get("relative_path", "")).strip()
        if not rel:
            continue
        records.append(entry)

    with out_path.open("w", encoding="utf-8") as out:
        for batch_idx, batch in enumerate(chunk(records, max(1, args.batch_size))):
            if len(batch) != 1:
                raise SystemExit("batch-size > 1 is not implemented yet (use --batch-size 1).")
            entry = batch[0]
            rel = str(entry.get("relative_path", "")).strip()
            script = script_basename(rel)
            prompt = build_prompt(entry, args.examples_per_script)
            obj = {
                "id": f"{batch_idx:06d}",
                "script": script,
                "relative_path": rel,
                "prompt": prompt,
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} prompts to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
