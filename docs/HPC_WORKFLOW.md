# HPC WORKFLOW — nu-hpc (SLURM, H100 partition)

**CONFIRMED 2026-09-06 by direct recon — this section replaces the original (wrong) assumptions
below it. Read this first.**

- Login: `ssh nu-hpc`. Account `issai`. Partitions: `H100` (6 nodes g001-g006, 4×H100/node,
  770GB RAM, 192 CPUs) and `ZEN4` (CPU-only, 10 nodes, 192 CPUs, 385GB RAM each).
- QOS: account has `issai-h100` and `issai-zen4`. **The account default QOS is `issai-zen4`,
  which is NOT valid on the H100 partition** — every sbatch AND every interactive `srun`/`salloc`
  that touches H100 must pass `--qos=issai-h100` explicitly, or it sits PD forever with reason
  `Job's QOS not permitted to use this partition`.
- **GPU confirmed**: `nvidia-smi` inside an `sbatch --gres=gpu:1 --qos=issai-h100` job on H100
  shows an NVIDIA H100 80GB HBM3, driver 610.43.02 (CUDA UMD 13.3, compute capability 9.0) —
  new enough to run any cu121/cu124 PyTorch/vLLM wheel (driver is backward compatible).
- **No environment modules.** `module` is not a recognized command on this cluster — do not use
  `module load` anywhere (EXECUTION_PLAN.md's `module load Python/...` / `module load cuda` lines
  are wrong for this cluster, kept only as a generic fallback comment). There is also no conda/
  mamba/apptainer/singularity. Environment = bare system `python3` (3.9.25, at `/usr/bin/python3`)
  + a plain venv + pip. CUDA/nvcc are available on the GPU nodes themselves (not the login node);
  verify inside a job with `nvidia-smi` — don't try to load a cuda module.
- **No `/scratch` mount.** Storage is a single BeeGFS pool (`beegfs_nodev`, 84T total) mounted at
  `/shared/home`, used for both home and working data — there is no separate small-home/big-scratch
  split. Quota tool is `beegfs-ctl --getquota --uid $USER --csv` (NOT `lfs quota`, NOT `quota -s`
  — both fail on this filesystem).

  ⚠️ **Quota anomaly, empirically probed but still unresolved — check before running anything
  storage-heavy.** `beegfs-ctl --getquota --uid $USER --csv` reports a hard limit of ~400 GiB
  (429496729600 bytes) with tracked usage ~144 GiB, while `du -sh /shared/home/$USER/rsna-knee`
  shows **531 GB already on disk** (mostly `data/raw/train_series`). We wrote a 1 GiB test file
  into the tree and it succeeded instantly, and the quota tool's usage figure went up by exactly
  that 1 GiB — so the tool *is* live and accurate for new writes, it just does not reflect the
  531 GB that's already there. Most likely explanation: the quota daemon's usage figure was only
  ever synced against a periodic scan from before the competition data landed, and a future
  resync could suddenly show the account far over its 400 GiB hard limit. We could not verify this
  further without cluster-admin access. **Ask cluster admins to confirm the real enforced limit
  before Phase 2's cache build** (budgeted at 1.5-2x raw = another ~800GB-1TB) or before pulling
  down multiple 70B+ HF models for the label engine — either could trip a resync into ENOSPC
  mid-job. Until confirmed, treat `df -h /shared/home` (84T pool, 48T free across ALL users as of
  2026-09-06) as the only trustworthy signal, and prefer quantized/AWQ model checkpoints to keep
  our own footprint small regardless.

Layout convention used by all our scripts (single pool, no scratch/home split):

    /shared/home/$USER/rsna-knee/
      data/raw/          # competition data, already extracted here (train_series/, train.csv, ...)
      data/cache/        # preprocessed tensors (largest consumer, budget 1.5-2x raw — watch quota above)
      artifacts/         # checkpoints, OOF parquets, LLM label outputs
      code/              # synced from local git via scripts/push.sh
      hf_cache/          # HF_HOME for vLLM model downloads
      venv/              # python3 -m venv, no modules/conda involved
      logs/              # sbatch stdout/stderr

Export once in `~/.bashrc`: `export RSNA_ROOT=/shared/home/$USER/rsna-knee`

---

## Original assumptions (kept for reference, superseded by the confirmed section above)

The competition dataset is DICOM-heavy (expect 100–300 GB; verify below). Home dirs on
clusters are usually 20–100 GB — the data almost certainly needs scratch/project space.

    df -h $HOME
    quota -s                          # classic quota, may not exist
    lfs quota -h -u $USER /scratch    # Lustre quota, adjust path
    ls -d /scratch/$USER /raid /data /projects 2>/dev/null
    # If nothing obvious: check cluster docs or `echo $SCRATCH`, ask admins.

## 2. Kaggle CLI on the HPC

**Already done for this project** — data is downloaded and extracted at
`/shared/home/$USER/rsna-knee/data/raw/` (train_series/: 4407 study dirs confirmed, train.csv,
train_series.csv, test.csv, test_series.csv, test_series/, sample_submission.csv all present).
Kept below for reference / re-download if ever needed.

    python3 -m venv ~/rsna-knee/venv && source ~/rsna-knee/venv/bin/activate
    pip install kaggle
    # On your LOCAL machine: kaggle.com → Account → Create New API Token → kaggle.json
    # Copy it up:
    #   (local)  scp ~/.kaggle/kaggle.json nu-hpc:~/.kaggle/kaggle.json
    ssh nu-hpc "chmod 600 ~/.kaggle/kaggle.json"

    # ALWAYS check size first:
    kaggle competitions files -c rsna-knee-abnormality-detection | tail -5
    # Accept rules on the website once before the API will let you download.

    cd $RSNA_ROOT/data/raw
    # Run in tmux/screen or nohup — this is a long download:
    tmux new -s dl
    kaggle competitions download -c rsna-knee-abnormality-detection
    unzip -q rsna-knee-abnormality-detection.zip -d .
    # Verify:
    ls train_series | wc -l    # expect 4407
    find train_series -mindepth 2 -maxdepth 2 -type d | wc -l   # expect 24371

Notes:
- No `module` command and no conda on this cluster — the venv above uses the bare system
  `python3` (3.9.25). Confirm before installing large ML packages (torch/vllm) that a 3.9-compatible
  wheel exists for the version you pin; upgrade the venv's own pip/setuptools first.
- **Confirmed 2026-09-06: H100 compute nodes DO have outbound internet** (curl to pypi.org and
  huggingface.co both returned HTTP 200 from inside an sbatch job with `--qos=issai-h100`). So
  `pip install` and `huggingface-cli download` work fine either from the login node or inside a
  SLURM job — no need to pre-stage everything from the login node. (An earlier interactive
  `srun --pty ... curl` attempt hung and hit its walltime with no output; that was almost certainly
  the `--pty` flag over a non-interactive ssh session, not a real network block — a plain
  `sbatch --wait` script with `timeout N curl ...` and `--output` on the shared filesystem is the
  reliable way to test node-side network from here on. Don't use `--pty` for non-interactive checks.)
- Weights upload back to Kaggle also via CLI: `kaggle datasets create/version -p <dir>`
  (from login node or from local after rsync-ing weights down).
- HuggingFace models for vLLM: `huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ`
  on the login node; set `HF_HOME=$RSNA_ROOT/hf_cache`. Prefer AWQ/quantized checkpoints over bf16
  originals — see the quota anomaly warning above, a 70B+ bf16 model is ~140GB by itself.

## 3. Local ↔ HPC loop (Claude Code edits locally, HPC executes)

Windows: use `rsync` from Git Bash/WSL, or `scp` fallback. **Confirmed 2026-09-06: plain Git Bash
on this machine has no `rsync`** (only `scp`/`tar`/`ssh`) — `scripts/push.sh`/`pull.sh` auto-detect
this and fall back to a tar-over-ssh transfer (works fine, just no delete-sync: files removed
locally aren't removed remotely until you clean them up by hand). Install rsync via
`pacman -S rsync` (if using an MSYS2-based Git install) or use WSL if you want true delete-sync.
With `~/.ssh/config` already containing the `nu-hpc` host alias:

    # push code (from repo root, local):
    rsync -avz --delete --exclude venv --exclude artifacts --exclude logs \
        ./ nu-hpc:/shared/home/USERNAME/rsna-knee/code/
    # submit (note --qos is mandatory on this cluster's H100 partition):
    ssh nu-hpc "cd /shared/home/USERNAME/rsna-knee/code && sbatch slurm/train.sbatch configs/exp001.yaml"
    # watch:
    ssh nu-hpc "squeue -u \$USER; tail -50 /shared/home/USERNAME/rsna-knee/code/logs/*_latest.out"
    # pull results (OOF, metrics, small checkpoints):
    rsync -avz nu-hpc:/shared/home/USERNAME/rsna-knee/artifacts/exp001/ ./artifacts/exp001/

Wrap these as `scripts/push.sh`, `scripts/run.sh <config>`, `scripts/pull.sh <exp>` so the loop
is one command each. Telegram bot pings on job end (hook `sbatch --wrap` or trap in script).

## 3.5. KNOWN CLUSTER BUG — GPU isolation between co-located jobs (found 2026-09-07)

**Confirmed via direct probe: when two or more jobs (ours or another tenant's) land on the same
H100 node, ALL of them report `CUDA_VISIBLE_DEVICES=0`** regardless of which GRES index SLURM
logically assigned each one. A 4-task array job requesting `--gres=gpu:1` each, landing together
on one node, printed `CUDA_VISIBLE_DEVICES=0` for every single task — the cgroup/gres device
isolation that's supposed to give each job its own distinct physical GPU is not working on this
cluster. Small operations (a 1000x1000 tensor alloc) "succeed" for everyone since GPU 0 has
capacity to spare, masking the bug at trivial scale.

**Symptom in real training jobs:** co-located jobs eventually collide once combined GPU memory
pressure gets high enough — anywhere from immediately (exit code `0:53`, no application log at
all) to ~54 minutes in (exit code `120:0`, training runs fine until it doesn't, no Python
exception in the log — the process is killed externally). We hit this repeatedly on 2026-09-07
running exp002's 5-fold array job: 4 co-located tasks died together at ~54min; retries on other
nodes (including previously-unused ones) died again within seconds whenever co-located, on
every node we tried (g003, g004, g005) — this is not a single bad node, it's cluster-wide.

**This can also be triggered by OTHER users' jobs sharing a node**, not just our own — a job we
submitted to a node with zero other jobs of ours still failed instantly, most likely because
another tenant's job was already there.

**Mitigation (imperfect, no full fix available from user-space):**
- `--exclusive` (reserve the whole node) avoids the bug but is usually blocked by this account's
  QOS GRES-per-user cap once a few jobs are already running — don't rely on it for a 5-job sweep.
- `--nodelist=<node>` to manually spread jobs across distinct nodes helps but is NOT sufficient
  on its own, since another tenant can still land on "your" node afterward.
- The only fully reliable workaround is running one GPU job at a time (no concurrent submissions
  from us at all) — this sacrifices the fold-parallel-array-job speed advantage from §4 below, but
  is what actually avoids the collision reliably.
- Checkpointing (every epoch, `--resume auto`) makes retries cheap regardless — a killed job just
  needs resubmitting, no lost work beyond the current epoch.
- **This needs cluster-admin attention** (`gres.conf`/`cgroup.conf` likely missing
  `ConstrainDevices=yes` or an equivalent device cgroup setting) — flag it to HPC support with the
  probe evidence above (`CUDA_VISIBLE_DEVICES=0` for every co-located task) if it keeps costing
  wall-clock time.

## 4. SLURM patterns for this project

- **Fold-parallel arrays** beat multi-GPU: `--array=0-4`, one GPU per fold.
- Long trainings: request realistic `--time` (queue prioritizes shorter), make everything
  resumable, chain with `--dependency=afterany:<jobid>` if walltime caps are short.
- Dataloading: `--cpus-per-task=16` minimum for DICOM/cache reading; `--mem=64GB`+.
- vLLM labeling: 1 GPU, `--mem=128GB`, model pre-downloaded to HF_HOME.

Templates live in `slurm/` (train.sbatch, label_gen.sbatch, cache_build.sbatch).
Useful: `sacct -j <id> --format=Elapsed,MaxRSS,State`, `scontrol show job <id>`,
`sinfo -p H100 -o "%n %t %G"`.

## 5. First-session checklist (copy-paste order) — as actually run on this cluster

    ssh nu-hpc
    df -h /shared/home/$USER
    beegfs-ctl --getquota --uid $USER --csv
    sinfo -o "%P %n %G %m %c %t"
    squeue -p H100 | wc -l
    mkdir -p /shared/home/$USER/rsna-knee/{data/raw,data/cache,artifacts,hf_cache,code} \
             /shared/home/$USER/rsna-knee/logs
    python3 -m venv /shared/home/$USER/rsna-knee/venv
    source /shared/home/$USER/rsna-knee/venv/bin/activate
    pip install --upgrade pip kaggle
    # ...then §2 (data already present for this project), then §0.2 of EXECUTION_PLAN for the
    # full env — remember: no modules/conda, plain pip into the venv above; and every sbatch/srun
    # touching the H100 partition needs --qos=issai-h100 (the account default QOS, issai-zen4,
    # is not valid there).
