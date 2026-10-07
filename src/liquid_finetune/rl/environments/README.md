# RL environments (OpenEnv integration - advanced)

> **You probably don't need this.** For any task where a completion
> can be scored purely as a function of its text (math, code,
> grounding, format, schema), use the `rewards:` block instead. See
> [`rewards/README.md`](../../../../rewards/README.md).
>
> `rl_env:` is for **agentic, stateful, multi-turn** tasks where the
> environment evolves from agent actions: real tool use, game
> simulators, browsing, stateful feedback. It's an optional extra:
> install with `uv sync --extra rl-env` only if you need it.

`liquid-finetune` uses [OpenEnv](https://github.com/meta-pytorch/OpenEnv),
the Gym-style, HF-Hub-distributed environment standard co-built by
Meta PyTorch and HuggingFace, as its agentic environment layer.

Environments live on the HuggingFace Hub as Spaces, alongside the
models and datasets you already use. You reference them from your YAML
config by repo-id. We do NOT maintain a registry here.

## Quickstart

```bash
uv sync --extra rl-env
```

Copy
[`job_configs/grpo_openenv_example.yaml`](../../../../job_configs/grpo_openenv_example.yaml)
and launch it the same way as any other GRPO config:

```bash
uv run liquid-finetune job_configs/grpo_openenv_example.yaml
```

The example targets the hosted `qgallouedec/echo_env` Space over HTTPS
with no Docker required.

When `rl_env:` is set, the training loop:

1. Resolves the YAML block to a live OpenEnv client via `connect_openenv`.
2. Wraps it in a TRL `rollout_func` that uses the trainer's native
   generation path (Transformers, vLLM colocate, or vLLM server).
3. Auto-prepends `env_reward` to your reward list so the environment's
   per-step reward contributes to the GRPO objective. You can stack
   additional file-based rewards under `rewards:` as usual; they
   compose via `reward_weights`.

## YAML reference

```yaml
rl_env:
  source: "qgallouedec/echo_env" # HF Hub repo-id OR installed env name
  base_url: null # optional: connect to a running HTTP server
  docker_image: null # optional: override the default Docker image
  env_vars: {} # optional: env vars forwarded to the env container
  wait_timeout: 30.0 # seconds to wait for container startup
  skip_install: false # true means use GenericEnvClient (no package install)
  trust_remote_code: true # skip interactive confirmation (default true)
  max_turns: 1 # only 1 supported in the default adapter
  reset_kwargs: {} # kwargs forwarded to env.reset() each episode
  action_key: "message" # dict key for env.step({action_key: text})
```

## Picking a source

| Situation                                                                      | What to set                                                            |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Env is published as an HF Space and you want to talk to it directly over HTTPS | `source`, `base_url: "https://<space>.hf.space"`, `skip_install: true` |
| Env is published on the Hub and you want auto-install + auto-run               | `source: "org/env-name"` (needs Docker)                                |
| You're running the env locally via `docker run -p 8001:8001 ...`               | `base_url: "http://localhost:8001"`                                    |
| You have a custom Docker image                                                 | `docker_image: "my-env:latest"`                                        |

**On Modal**, prefer the hosted-Space path. Modal's training
container doesn't have Docker-in-Docker, so auto-install from Hub
would fail. Set `source` + `base_url` + `skip_install: true`.

**On SLURM / local multi-node**, Docker is usually available; `source`
with auto-start works.

## Action schema matters

Every OpenEnv environment declares its own action schema at
`GET /schema`. The default adapter sends
`env.step({action_key: completion_text})`. That works for simple
message-style envs (like `qgallouedec/echo_env`, action key `message`)
but fails for envs that expect structured actions (e.g. the
MCP-based `openenv/echo_env`, which expects
`{"tool_name": ..., "arguments": {...}}`).

Before pointing a config at a new env, check its schema:

```bash
curl https://<space>.hf.space/schema | python -m json.tool | head -30
```

If the schema is a single `{message: string}` field, set
`action_key: "message"`. If the action is structured, you'll need a
custom `rollout_func`; see the next section.

## Writing a custom rollout for multi-turn or structured actions

The default adapter supports single-turn rollouts with a flat
`{action_key: text}` action. For multi-turn episodes or envs that
expect structured JSON actions, write a Python `rollout_func` and
pass it to the trainer instead. The whole contract is: take
`(prompts, trainer)`, return a dict with `prompt_ids`,
`completion_ids`, `logprobs`, and any extra fields you want
forwarded to reward functions as kwargs.

See `src/liquid_finetune/rl/environments/adapter.py::build_openenv_rollout_func`
for the reference implementation.

## Publishing a new environment

Writing and publishing an OpenEnv environment is an OpenEnv-ecosystem
task, not a liquid-finetune one:

- [OpenEnv environment builder guide](https://meta-pytorch.org/OpenEnv/environment-builder/)
- [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)

Once your env is published as a Space, anyone can reference it from a
liquid-finetune YAML config via `rl_env.source`.

## Caveats

- TRL's `rollout_func` is marked **experimental**. The adapter isolates
  that dependency to a single file (`adapter.py`) so API changes stay local.
- OpenEnv itself is young (currently on v0.2). The contract may move.
