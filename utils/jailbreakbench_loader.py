"""
Fetch JailbreakBench attack artifacts directly from the public GitHub repo, without
depending on the `jailbreakbench` PyPI package.

Why: the `jailbreakbench` package is only ever used in this repo for its
`read_artifact()` data fetch (never its judge/litellm-based scoring functionality),
but it unconditionally imports `litellm` at package load time. `litellm`'s pinned
transformers/protobuf ranges conflict with other packages in some environments
(e.g. vllm, used by the WildGuard competitive baseline), and separately, recently
resolved `litellm` versions have removed internal modules
(`litellm.llms.prompt_templates`) that `jailbreakbench`'s own code still imports,
breaking `import jailbreakbench` outright on Python 3.10+ regardless of other
packages. Fetching the same public JSON artifacts directly over HTTP (via
`requests`, already a repo dependency) sidesteps all of this.

Verified against the actual repo layout at
https://github.com/JailbreakBench/artifacts/tree/main/attack-artifacts: only GCG,
JBC, and PAIR have artifacts for vicuna-13b-v1.5 (the target model used everywhere
in this repo). Other method names historically listed in JBB_METHODS across this
codebase (AutoDAN, TAP, PAP-top5, DrAttack, Persuasive, Persuasive+Jailbreak) do not
exist in that repo and always 404'd — silently skipped by the old
jbb.read_artifact()-based loops, which is why every JailbreakBench result file in
this repo already totals 282 = 100 (GCG) + 100 (JBC) + 82 (PAIR), not more.
"""

import sys

# method -> attack_type, as laid out in the artifacts repo.
JBB_METHODS = {
    "GCG": "white_box",
    "JBC": "manual",
    "PAIR": "black_box",
}
JBB_TARGET_MODEL = "vicuna-13b-v1.5"
JBB_ARTIFACT_URL = (
    "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/"
    "attack-artifacts/{method}/{attack_type}/{model}.json"
)


def load_jailbreakbench_artifacts(model_name: str = JBB_TARGET_MODEL) -> list[dict]:
    """Return a list of {'method', 'behavior_id', 'behavior', 'prompt', 'jbb_success'}
    dicts — the same shape every evaluate_jailbreakbench()-style function in this
    repo has always built from jbb.read_artifact(), so this is a drop-in replacement:
    just add "ground_truth": "harmful" and your model's prediction when consuming it.
    """
    import requests

    entries = []
    for method, attack_type in JBB_METHODS.items():
        url = JBB_ARTIFACT_URL.format(method=method, attack_type=attack_type, model=model_name)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            artifact = resp.json()
            method_entries = []
            for jb in artifact["jailbreaks"]:
                prompt = jb.get("prompt")
                if prompt is None:
                    continue
                method_entries.append({
                    "method": method,
                    "behavior_id": jb.get("index"),
                    "behavior": jb.get("goal") or jb.get("behavior"),
                    "prompt": prompt,
                    "jbb_success": jb.get("jailbroken"),
                })
            entries.extend(method_entries)
            print(f"  {method}: {len(method_entries)} prompts loaded")
        except Exception as e:
            print(f"  Skipping '{method}': {e}", file=sys.stderr)
    return entries
