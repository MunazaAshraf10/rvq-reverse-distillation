# rvq-reverse-distillation

Code for *Recovering a Withheld Audio Tokenizer from Its Generator: Reverse Distillation of the
MiniMax Music 3 Encoder, In and Out of Distribution*.

MiniMax Music 3 generates music as eight RVQ code streams at 25 Hz (one semantic, seven acoustic) but
ships no audio-to-code tokenizer, so it can only be prompted with text. We learn that tokenizer from the
generator's own outputs: every generated track comes with the codes and code distributions that produced
it, and an encoder learns the inverse map from the frozen DAV autoencoder latents. This repository holds
the encoder, its training, the condition replay metric reimplemented from the official components, and
every experiment, table and figure of the paper.

| | |
|---|---|
| Encoder weights | [Munaza10/rvq-reverse-distillation](https://huggingface.co/Munaza10/rvq-reverse-distillation) |
| MM3-OOD test set | [Munaza10/mm3-ood](https://huggingface.co/datasets/Munaza10/mm3-ood) |
| Training corpus | [bghira/minimax-music3-rvq-reverse-distillation](https://huggingface.co/datasets/bghira/minimax-music3-rvq-reverse-distillation) |

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/):

    uv sync                      # encoder and inference
    uv sync --all-extras         # plus training, the MiniMax components, evaluation and figures

## Encode audio

    uv run rvq-ae encode --audio song.flac --out codes.safetensors --topk 5

```python
from pathlib import Path

from rvq_ae.audio.io import load_audio
from rvq_ae.inference import CodeEncoder

codec = CodeEncoder.load(device="cuda")                 # subfolder="augmented" for lossy or noisy audio
result = codec.encode(*load_audio(Path("song.flac")), topk=5)
result.codes                                            # [frames, 8] at 25 Hz
```

The default encoder (169M parameters, causal depth decoder) is at the root of the model repository;
`augmented/` holds the same model trained on degraded latent views, which is robust to band limiting,
low-bitrate coding and noise at a cost of 0.01 replay cosine on clean audio.

## Train

One 24 GB GPU per run. Build the latent cache once (with two degraded views per track for the augmented
model), then train and evaluate:

    uv run rvq-ae cache --split train --split holdout --views 2
    uv run rvq-ae train --config configs/v4_169m.json --split-rule genre-ood --validation-split test-id \
        --seed 1 --output runs/genre-ood/v4_169m/seed-1
    uv run rvq-ae evaluate --run runs/genre-ood/v4_169m/seed-1 --checkpoint final \
        --split-rule genre-ood --split test-ood

| Config | Model |
|---|---|
| `v4_169m` | the encoder: 8-layer temporal transformer, causal depth decoder |
| `v4_169m_augmented` | the encoder trained on degraded latent views |
| `v1_41m`, `v2_155m` | independent-head baselines |
| `v2_169m_deep` | control: independent heads, parameter matched |
| `v4_157m_no_feedback` | control: depth decoder without code feedback |

Split rules: `published` (the corpus' held-out split) and `genre-ood` (jazz held out, with no prompt or
lyric shared with training).

## Reproduce the paper

Every number comes from a job list and every table and figure from a script reading `results/`:

    uv run python scripts/jobs.py                                      # runs/{train,experiments,evaluate}.jobs
    uv run python -m rvq_ae.experiments.queue runs/train.jobs --gpus 0,1,2,3
    uv run python -m rvq_ae.experiments.queue runs/experiments.jobs --gpus 0,1,2,3
    uv run python -m rvq_ae.experiments.queue runs/evaluate.jobs --gpus 0,1,2,3
    uv run python scripts/tables.py && uv run python scripts/figures.py   # into outputs/

The queue runs one job per GPU, lets several queues share a list, and retries only failed jobs; each
experiment shards over `RANK`/`WORLD_SIZE` and resumes from the rows it has written. `results/` already
holds the per-track rows of every run in the paper, so the last line works without a GPU. The full
study took about 110 GPU hours on four RTX 3090s.

## Layout

    src/rvq_ae/
      models/        encoder, transformer layers, losses, maximal update parametrisation
      audio/         audio I/O, the DAV autoencoder, the frame-to-latent timeline, degradations
      data/          corpus records, split rules, latent cache, training windows
      training/      trainer, learning rate schedules, token metrics
      minimax/       official MiniMax Music 3 components: condition replay, rendering, sampling
      experiments/   replay, calibration, MM3-OOD generation, resynthesis, codecs, use cases, statistics, queue
      inference.py   CodeEncoder: audio in, codes out
      hub.py         loading and publishing checkpoints
      cli.py         rvq-ae cache | train | evaluate | encode | push
    configs/         model configurations
    scripts/         job lists, tables, figures
    results/         per-track results of every experiment

## Citation

```bibtex
@article{ashraf2026reverse,
  title  = {Recovering a Withheld Audio Tokenizer from Its Generator: Reverse Distillation of the
            {MiniMax Music 3} Encoder, In and Out of Distribution},
  author = {Ashraf, Munaza and Pande, Kash and Musa, Muhammad and Ahmad, Azka and Islam, Khawar},
  year   = {2026},
  url    = {https://github.com/MunazaAshraf10/rvq-reverse-distillation}
}
```

Code under the GNU Affero General Public License v3.0. The weights, MM3-OOD and anything generated with
MiniMax Music 3 follow the MiniMax Music 3 Community License.
