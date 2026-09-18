# Repo Notes

## Storage

- Do not use `/tmp` for large materializations such as model downloads, HF caches, tokenized datasets, checkpoint exports, vLLM assets, or package caches.
- Prefer repo-local caches for small/reusable artifacts and configured persistent shared storage for large model/checkpoint artifacts.
- `/tmp` is acceptable only when the tool requires node-local scratch space or short socket paths, such as Ray runtime/session directories.
