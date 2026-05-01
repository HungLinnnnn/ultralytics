# AGENTS.md — Ultralytics Implementation Workspace

This repository is the implementation workspace for YOLO-based cell instance segmentation research.

## Scope

This file applies to all work under:

- `/home/r13922151/ultralytics`

For research planning, method discussion, literature notes, and experiment reports, use the companion research workspace:

- `/home/r13922151/research_team`

## Required Reading Before Code Changes

Before modifying model, training, loss, evaluation, or utility code, read the relevant research plan and project instructions from `/home/r13922151/research_team`.

For CUTR-YOLO work, read:

- `/home/r13922151/research_team/inputs/human_notes/260501/CUTR-YOLO.md`
- `/home/r13922151/research_team/projects/cutr_yolo/AGENTS.md` if present
- `/home/r13922151/research_team/AGENTS.md` if present
- `/home/r13922151/research_team/approvals/APPROVAL_POLICY.md` if present

## Branch Policy

Use dedicated feature branches for research implementations.

Current CUTR-YOLO branch:

- `feature/cutr-yolo`

Do not implement CUTR-YOLO on `main`.

Before switching branches:

1. Run `git status`.
2. Preserve uncommitted work safely.
3. Do not discard or overwrite changes unless explicitly instructed.

## Experiment Safety

Do not start any of the following unless explicitly approved by the human:

- training
- validation
- prediction
- export
- benchmarking
- long-running scripts
- GPU-heavy jobs

Allowed without extra approval:

- reading files
- editing code
- lightweight syntax checks
- import checks
- `python -m py_compile <file>`
- `git status`, `git diff`, `git log`
- writing design notes or reports

If unsure whether an action counts as an experiment, do not run it.

## Implementation Policy

Prefer small, reviewable changes.

For new methods:

1. Add code behind explicit config flags.
2. Preserve baseline YOLOv8-seg behavior by default.
3. Avoid breaking existing YAMLs, trainers, validators, and prediction paths.
4. Keep experimental modules isolated and named clearly.
5. Do not delete legacy files unless explicitly instructed.
6. Do not silently change metric definitions.

For CUTR-YOLO specifically:

- Do not replace the YOLOv8-seg prototype-coefficient mechanism in the first implementation.
- Prefer adding a lightweight mask-logit refinement path.
- Training-time UOT should initially be implemented as a teacher or auxiliary target builder, not as an inference-time dependency.
- Inference should not require GT masks or a UOT solver.

## Metrics and Evaluation

Cell instance segmentation work should preserve or support the following metrics when applicable:

- Dice
- AJI
- PQ
- mask mAP
- boundary F1 if available
- merge/split diagnostics if available

Do not change existing evaluation scripts without documenting the reason.

## Documentation and Reports

Implementation notes, risk audits, and experiment summaries should be written under `/home/r13922151/research_team`, not scattered inside this repo, unless the file is directly part of code documentation.

Recommended report locations:

- `/home/r13922151/research_team/outputs/discussions/decisions/`
- `/home/r13922151/research_team/outputs/result_summaries/`
- `/home/r13922151/research_team/projects/cutr_yolo/outputs/`

## Git Hygiene

Before and after meaningful work, show:

- `git status`
- `git branch --show-current`
- relevant `git diff --stat`

Commit messages should clearly state the scope, for example:

- `chore: restore reusable segmentation utilities`
- `feat(cutr): add mask tokenization utilities`
- `feat(cutr): add residual refinement head`
- `docs(cutr): add implementation plan`

Do not push broken or unreviewed large changes unless explicitly instructed.