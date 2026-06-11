# AGENTS.md — Ultralytics Implementation Workspace

This repository is the implementation workspace for YOLO-based cell instance segmentation research.

## Scope

This file applies to all work under this Ultralytics implementation repo.

Current known locations:

- Local macOS workspace: `/Users/hung/Documents/research_ideas/repos/ultralytics`
- Remote server workspace, after sync: `/home/r13922151/research_ideas/repos/ultralytics`

The companion research workspace is `research_ideas`.

Current known research workspace locations:

- Local macOS: `/Users/hung/Documents/research_ideas`
- Remote server, after sync: `/home/r13922151/research_ideas`

Use the research workspace for research planning, method notes, literature/context files, prompt files, experiment plans, and result summaries. Use this Ultralytics repo for implementation code only.

## Required Reading Before Code Changes

Before modifying model, training, loss, evaluation, or utility code, read the relevant project instructions and method plan from the current `research_ideas` workspace.

For the current PCU-YOLO v1.5 work, read at minimum:

- `/Users/hung/Documents/research_ideas/AGENTS.md` locally, or `/home/r13922151/research_ideas/AGENTS.md` on the remote server
- `/Users/hung/Documents/research_ideas/research_archive/pcu_magss_yolo/pcv_mgss_yolo_proposal_v1.5_mechanism_simplification_review.md` locally, or the corresponding remote path under `/home/r13922151/research_ideas/`
- the active Codex prompt under `/Users/hung/Documents/research_ideas/prompts/active/` locally, or the corresponding remote path under `/home/r13922151/research_ideas/prompts/active/`

Do not chase obsolete historical research-workspace paths. Do not use historical project-specific instructions from older methods unless the active prompt explicitly requests them.

## Branch Policy

Use dedicated feature branches for research implementations. Do not implement new research methods directly on `main`.

Before switching branches or editing code:

1. Run `git status --short --branch`.
2. Preserve uncommitted work safely.
3. Do not discard, reset, clean, or overwrite changes unless explicitly instructed.

## Experiment Safety

Do not start any of the following unless explicitly approved by the human:

- training
- validation
- prediction
- export
- benchmarking
- long-running scripts
- GPU-heavy jobs
- dataset downloads
- checkpoint/weight downloads
- package installation

Allowed without extra approval:

- reading files
- editing code
- lightweight syntax checks
- import checks
- YAML/model parse checks
- synthetic CPU tensor forward smoke tests
- `python -m py_compile <file>`
- `python -m compileall <small explicit file-or-directory set>`
- `git status`, `git diff`, `git diff --stat`, `git log`
- writing implementation notes or reports under the research workspace when requested

If unsure whether an action counts as an experiment, do not run it.

## Implementation Policy

Prefer small, reviewable changes.

For new methods:

1. Add code behind explicit config flags or clearly named experimental modules.
2. Preserve baseline YOLOv8-seg behavior by default.
3. Avoid breaking existing YAMLs, trainers, validators, prediction paths, export paths, and metrics.
4. Keep experimental modules isolated and named clearly.
5. Do not delete legacy files unless explicitly instructed.
6. Do not silently change dataset splits, evaluation metrics, NMS behavior, mask post-processing, or training protocols.
7. Do not modify datasets, checkpoints, weights, `runs/`, `wandb/`, or experiment outputs unless explicitly instructed.

For PCU-YOLO work specifically:

- Implement only the active approved PCU mechanism described in the active prompt and method review.
- Do not redesign the research method inside this repo.
- Do not implement historical SSM-Net, BDFWarpUp, SGF, GateConcat, frequency decomposition, generic Mamba neck, or old PAN-like replacement ideas unless a current prompt explicitly asks for them.
- Do not add Readout B, boundary supervision, SAM, extra transformer branches, post-processing, or detection-head feedback unless explicitly approved.
- PCU should affect the segmentation mask path only unless the prompt states otherwise.

## Metrics and Evaluation

Cell instance segmentation work should preserve or support the following metrics when applicable:

- Dice
- AJI
- PQ
- mask mAP
- boundary F1 if available
- merge/split diagnostics if available

Do not change existing evaluation scripts or metric definitions without documenting the reason and receiving explicit approval.

## Documentation and Reports

Research implementation notes, risk audits, and experiment summaries should live under the `research_ideas` workspace, not scattered inside this repo, unless the file is directly part of code documentation.

Recommended local locations when requested:

- `/Users/hung/Documents/research_ideas/prompts/active/`
- `/Users/hung/Documents/research_ideas/research_archive/`
- `/Users/hung/Documents/research_ideas/experiments/`

Corresponding remote locations should be under:

- `/home/r13922151/research_ideas/`

## Git Hygiene

Before and after meaningful work, show:

- `git status --short --branch`
- `git branch --show-current`
- relevant `git diff --stat`

Do not commit, push, reset, clean, discard user changes, or open PRs unless explicitly instructed.
