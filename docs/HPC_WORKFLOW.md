# HPC WORKFLOW — nu-hpc (SLURM, H100 partition)

Login: `ssh nu-hpc`. Partition `H100`, qos `issai-h100`. Modules: `Python/3.11.5-GCCcore-13.2.0`
(confirm cuda module name with `module avail cuda` / `module avail CUDA` before use).

## 1. Check storage BEFORE downloading anything

The competition dataset is DICOM-heavy (expect 100–300 GB; verify below). Home dirs on
clusters are usually 20–100 GB — the data almost certainly needs scratch/project space.

    df -h $HOME
    quota -s                          # classic quota, may not exist
    lfs quota -h -u $USER /scratch    # Lustre quota, adjust path
    ls -d /scratch/$USER /raid /data /projects 2>/dev/null
    # If nothing obvious: check cluster docs or `echo $SCRATCH`, ask admins.

Layout convention used by all our scripts:

    /scratch/$USER/rsna_knee/
      data/raw/          # competition zip + extracted DICOMs (read-only after extract)
      data/cache/        # preprocessed tensors (largest consumer, budget 1.5–2× raw)
      artifacts/         # checkpoints, OOF parquets
    ~/rsna_knee/         # code (small, in $HOME, synced from local git)
      code/  configs/  slurm/  logs/  venv/

Export once in `~/.bashrc`: `export RSNA_ROOT=/scratch/$USER/rsna_knee`

## 2. Kaggle CLI on the HPC

    source ~/rsna_knee/venv/bin/activate
    pip install kaggle
    # On your LOCAL machine: kaggle.com → Account → Create New API Token → kaggle.json
    # Copy it up:
    #   (local)  scp ~/.kaggle/kaggle.json nu-hpc:~/.kaggle/kaggle.json
    ssh nu-hpc "mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json"

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
- Downloads run on the **login node** (compute nodes often have no internet — test with
  `srun -p H100 --time=5 curl -sI https://pypi.org`). If login-node downloads are frowned upon,
  ask whether a data-transfer node exists.
- Weights upload back to Kaggle also via CLI: `kaggle datasets create/version -p <dir>`
  (from login node or from local after rsync-ing weights down).
- HuggingFace models for vLLM: `huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ`
  on the login node; set `HF_HOME=$RSNA_ROOT/hf_cache`.

## 3. Local ↔ HPC loop (Claude Code edits locally, HPC executes)

Windows: use `rsync` from Git Bash/WSL, or `scp` fallback. With `~/.ssh/config` already
containing the `nu-hpc` host alias:

    # push code (from repo root, local):
    rsync -avz --delete --exclude venv --exclude artifacts --exclude logs \
        ./ nu-hpc:~/rsna_knee/code/
    # submit:
    ssh nu-hpc "cd ~/rsna_knee/code && sbatch slurm/train.sbatch configs/exp001.yaml"
    # watch:
    ssh nu-hpc "squeue -u \$USER; tail -50 ~/rsna_knee/code/logs/*_latest.out"
    # pull results (OOF, metrics, small checkpoints):
    rsync -avz nu-hpc:/scratch/USERNAME/rsna_knee/artifacts/exp001/ ./artifacts/exp001/

Wrap these as `scripts/push.sh`, `scripts/run.sh <config>`, `scripts/pull.sh <exp>` so the loop
is one command each. Telegram bot pings on job end (hook `sbatch --wrap` or trap in script).

## 4. SLURM patterns for this project

- **Fold-parallel arrays** beat multi-GPU: `--array=0-4`, one GPU per fold.
- Long trainings: request realistic `--time` (queue prioritizes shorter), make everything
  resumable, chain with `--dependency=afterany:<jobid>` if walltime caps are short.
- Dataloading: `--cpus-per-task=16` minimum for DICOM/cache reading; `--mem=64GB`+.
- vLLM labeling: 1 GPU, `--mem=128GB`, model pre-downloaded to HF_HOME.

Templates live in `slurm/` (train.sbatch, label_gen.sbatch, cache_build.sbatch).
Useful: `sacct -j <id> --format=Elapsed,MaxRSS,State`, `scontrol show job <id>`,
`sinfo -p H100 -o "%n %t %G"`.

## 5. First-session checklist (copy-paste order)

    ssh nu-hpc
    df -h $HOME && (quota -s || lfs quota -h -u $USER /scratch) 
    module avail Python 2>&1 | grep -i python
    module avail cuda 2>&1 | grep -i cuda
    sinfo -p H100; squeue -p H100 | wc -l
    mkdir -p /scratch/$USER/rsna_knee/{data/raw,data/cache,artifacts} ~/rsna_knee/logs
    module load Python/3.11.5-GCCcore-13.2.0
    python -m venv ~/rsna_knee/venv && source ~/rsna_knee/venv/bin/activate
    pip install --upgrade pip kaggle
    # ...then §2 download, then §0.2 of EXECUTION_PLAN for the full env.
