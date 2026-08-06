# boltz-cdr

**Concentrating structural sampling where an antibody-structure model is actually uncertain.**

Tools for generating and ranking antibody–antigen interface ensembles with
[Boltz-2](https://github.com/jwohlwend/boltz): CDR-selective diffusion resampling,
template-masked loop re-prediction, a differentiable CDR–epitope guidance potential, and a
GPU-free evaluation suite for scoring the results.

---

## Why

AF3-class models predict antibody–antigen complexes far worse than they predict protein
complexes generally — roughly 30–40 % DockQ-acceptable against ~80 %. The reason is
structural. An antibody and its antigen share no evolutionary history, so the paired MSA
carries no coevolutionary contact signal; and the antibody's own MSA is dominated by
germline framework, which dilutes the hypervariable CDR3 that determines specificity. The
model's uncertainty is concentrated in the CDR loops.

Its sampler, however, spreads stochasticity uniformly over the whole complex, so drawing
more seeds mostly re-derives the parts that were already right. This package spends the
sampling budget on the loops instead, and then asks the question that matters just as much
in practice: given an ensemble, which scoring function picks the right member?

---

## Method

```
                     ┌──────────────────────────────────────────────┐
                     │   nanobody sequence   +   antigen sequence   │
                     │        (no complex structure known)          │
                     └───────────────────────┬──────────────────────┘
                                             │
 ╔═══════════════════════════════════════════▼════════════════════════════════════════╗
 ║  STAGE 0 · GLOBAL DOCK                              stock Boltz-2  ×  N seeds       ║
 ║  rank poses by  ipTM · interface-PAE · interface-pLDDT   ──────────►  ARM 0 BASELINE║
 ╚═══════════════════════════════════════════╤════════════════════════════════════════╝
                                             │  top-K docked poses
                          ┌──────────────────┴──────────────────┐
                          │                                     │
 ╔════════════════════════▼═══════════════════╗ ╔═══════════════▼════════════════════════╗
 ║  ARM A · TEMPLATE MASKING   (input space)  ║ ║  ARM B · DIFFUSION RESAMPLING          ║
 ║                                            ║ ║             (sampling space)           ║
 ╠════════════════════════════════════════════╣ ╠════════════════════════════════════════╣
 ║                                            ║ ║  patch AtomDiffusion.sample()          ║
 ║   docked complex                           ║ ║                                        ║
 ║        │                                   ║ ║  B1  eps *= 1 + (λ−1)·mask_CDR         ║
 ║        │  ✂ delete CDR residues            ║ ║      loop-only stochasticity; framework║
 ║        ▼                                   ║ ║      + antigen keep normal trajectory  ║
 ║   ┌────────────────────┐                   ║ ║                                        ║
 ║   │ ████████  ✂  ✂  ✂ │  framework kept   ║ ║  B2  re-noise ONLY CDR atoms to an     ║
 ║   │ ████████  antigen  │  antigen kept     ║ ║      intermediate σ, starting from a   ║
 ║   │           CDRs ✗   │  CDRs removed     ║ ║      docked pose → partial diffusion   ║
 ║   └──────────┬─────────┘                   ║ ║                                        ║
 ║              │  templates: force=true      ║ ║  B3  ∇E  differentiable CDR–epitope    ║
 ║              ▼                             ║ ║      interface potential, autograd,    ║
 ║   Boltz-2 × M seeds                        ║ ║      gradients masked to CDR atoms     ║
 ║   CDR loops rebuilt de novo inside a       ║ ║      → also gives epitope-directed     ║
 ║   FIXED, already-docked frame              ║ ║        generation, with no retraining  ║
 ╚════════════════════════╤═══════════════════╝ ╚═══════════════╤════════════════════════╝
                          └──────────────────┬──────────────────┘
                                             ▼
 ╔════════════════════════════════════════════════════════════════════════════════════╗
 ║  STAGE 3 · SCORE  &  SELECT      — two independent scorer families, compared —      ║
 ║                                                                                     ║
 ║   model confidence          │   interface physics (model-independent)               ║
 ║   ipTM · pTM · ipLDDT · iPAE│   shape complementarity Sc · interface H-bonds        ║
 ║                             │   salt bridges · buried SASA · clashes                ║
 ╚═══════════════════════════════════════════╤════════════════════════════════════════╝
                                             ▼
 ╔════════════════════════════════════════════════════════════════════════════════════╗
 ║  STAGE 4 · EVALUATE  vs. crystal structure                                          ║
 ║                                                                                     ║
 ║   ACCURACY      complex RMSD · ligand RMSD · interface RMSD · per-CDR RMSD · DockQ   ║
 ║   CONTACTS      fnat · fnonnat · precision/recall/F1 · epitope & paratope recall     ║
 ║   ENSEMBLE      diversity · best-of-N coverage                                       ║
 ║   RANKABILITY   Spearman(score, DockQ) · top-1 selected DockQ · enrichment           ║
 ╚════════════════════════════════════════════════════════════════════════════════════╝
```

Stage 0 finds *where* the binder sits. Arms A and B spend the remaining budget on *how the
CDR loops sit in that pocket* — Arm A by removing the loops from the model's input, Arm B
by reshaping the diffusion trajectory. Stage 3 asks which score picks the winner; Stage 4
checks everything against a crystal structure.

Boltz-2 is never modified on disk. Two attributes are rebound at runtime by
`boltz_cdr.patch`, reversibly, and the installer refuses to run against a Boltz version
other than the one the vendored sampler was derived from.

---

## Install

Everything except Boltz-2 inference runs on CPU with no model weights: CDR annotation,
template construction, the guidance potential and its gradient, all evaluation metrics, the
scorer comparison, and the test suite.

```bash
git clone https://github.com/bendorr/boltz-cdr.git
cd boltz-cdr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # 136 tests
```

For Boltz-2 inference, on a GPU:

```bash
pip install -r requirements-gpu.txt
```

Boltz-2 weights (~3 GB) download on first use to `~/.boltz` and are not vendored here.

---

## Quick start

On CPU:

```bash
# fetch and validate the benchmark complexes, with ground-truth interface statistics
python scripts/00_fetch_targets.py

# which loops get resampled, and how to define them yourself
python scripts/07_cdr_selection_demo.py --target 8QF4

# the guidance potential: energy, gradients, and a finite-difference check
python scripts/05_backward_pass_demo.py --target 8QF4 --displacement 4.0

# exercise the whole analysis path on a synthetic ensemble, without a model
python scripts/06_synthetic_ensemble.py --all
python scripts/04_evaluate.py --all --results results_synthetic
```

On a GPU:

```bash
python scripts/01_global_dock.py             --target 8QF4 --samples 8 --steered
python scripts/02_arm_a_template_resample.py --target 8QF4 --top-k 2 --samples 8
python scripts/03_arm_b_diffusion_resample.py --target 8QF4 --samples 8 \
    --lambda-cdr 1.5 --partial-sigma 8.0 --guidance-weight 0.2
python scripts/04_evaluate.py --target 8QF4
```

Roughly 25 GPU-minutes per target for all arms on an A100.
`notebooks/colab_boltz_cdr.ipynb` runs the whole pipeline end to end on Colab.

---

## Defining the CDRs

CDRs are annotated automatically by an IMGT annotator (no ANARCI or HMMER dependency), but
every stage accepts an explicit definition in **author residue numbering** — the numbers
you read off the structure in PyMOL. To see what the annotator picked and what the
alternatives would select:

```bash
python scripts/07_cdr_selection_demo.py --target 8QF4
```

Then use it:

```bash
--cdr-residues "E:28-35,53-60,99-117"      # positional -> CDR1, CDR2, CDR3
--cdr-residues "E:cdr3=99-117"             # named; omit loops you don't want resampled
--cdr-residues "E:cdr3=99-110+114-117"     # '+' joins discontinuous stretches
```

Per-target definitions go in `data/targets.yaml`:

```yaml
cdr_residues:
  chain: E
  cdr1: "28-35"
  cdr2: [53, 54, 55, 56, 57, 58, 59, 60]
  cdr3: "99-117"
```

Precedence: `--cdr-residues` > `cdr_residues` > `cdr_spans` (0-based, legacy) > automatic.

Author numbering is arbitrary — 8QF4's antibody starts at residue 2, 9EZU's at 0, 9HUR's
antigen at 108 — so every number is validated against the structure. Naming the wrong
chain, a residue that does not exist, ambiguous insertion-code numbering, and overlapping
loops are all errors rather than silent mis-selections.

---

## Benchmark targets

| PDB | Complex | Antigen fold | Antigen | Res (Å) | Released |
|---|---|---|---|---|---|
| `8QF4` | Nb H11 : Arc N-lobe | all-α helical bundle | 67 aa | 1.02 | 2024-03-13 |
| `9EZU` | VHH_h1 : VSIG4 | Ig β-sandwich | 118 aa | 1.05 | 2025-10-29 |
| `9HUR` | Sybody LA4 : CD63 LEL | all-α tetraspanin | 102 aa | 1.65 | 2025-11-19 |

All three post-date Boltz-2's PDB training cutoff (2023-06-01), so results are not
confounded by memorization. All have fully ordered CDR loops, no chain breaks, and
CDR-dominated paratopes: 17–20 of the ~20 contacting antibody residues are CDR residues.
`1ZVH` (2005) is available as an opt-in pre-cutoff control.

---

## Layout

```
boltz-cdr/
├── data/targets.yaml      benchmark complexes and selection criteria
├── src/boltz_cdr/
│   ├── pdb_io.py          mmCIF fetch/parse; numpy-native Chain/Complex types
│   ├── cdr.py             IMGT CDR annotation for VHH, no ANARCI dependency
│   ├── cdr_spec.py        user CDR definitions in author residue numbering
│   ├── masks.py           (chain, residue) → Boltz token mask → atom mask
│   ├── featurize.py       Boltz-shaped feature dicts built from a structure
│   ├── templates.py       CDR-masked template mmCIF                   [Arm A]
│   ├── yaml_io.py         Boltz-2 input YAML generation
│   ├── potentials.py      differentiable CDR–epitope potential        [Arm B3]
│   ├── sampler.py         CDR-selective noise + partial diffusion     [Arm B1/B2]
│   ├── patch.py           runtime installation into boltz==2.2.1, version-guarded
│   ├── run.py             in-process Boltz driver and output collection
│   ├── scoring.py         confidence + physics scorers, ranking comparison
│   ├── ensemble.py        ensemble diversity, clustering, coverage
│   └── metrics/           RMSD family, contacts, DockQ, interface physics
├── scripts/               00 fetch · 01 dock · 02 Arm A · 03 Arm B · 04 evaluate
│                          05 gradient demo · 06 synthetic ensemble · 07 CDR selection
├── notebooks/             end-to-end Colab runner
└── tests/                 136 tests, all CPU; a stub Boltz exercises the patch,
                           sampler, and runner without model weights
```

---

## Evaluation

All metrics are pure numpy/scipy and need neither a GPU nor Boltz, so results can be
re-analyzed long after a GPU session has ended.

**Accuracy** — DockQ and CAPRI class; ligand RMSD (antigen-superimposed, the docking
number); interface RMSD; complex RMSD; per-CDR RMSD after *framework* superposition, which
separates loop-conformation error from rigid-body pose error.

**Contacts** — fnat, fnonnat, precision/recall/F1, epitope and paratope recall, contact-map
Jaccard, and a per-residue-pair breakdown into recovered, missed, and spurious.

**Interface physics** — shape complementarity on a grid-derived solvent-excluded surface,
buried SASA, hydrogen bonds and their density, salt bridges, clashes, and buried
unsatisfied polar atoms.

**Rankability** — Spearman correlation with DockQ, top-1 selected DockQ, and enrichment
against oracle and random selection, for every scorer. Best-of-N is not actionable on its
own; what matters is whether a score can find the good member.

Calibration against the benchmark crystal structures: `evaluate(native, native)` returns
DockQ 1.0000 with all RMSDs at 0.000; shape complementarity comes out at 0.66–0.70 against
a published 0.64–0.68 range for antibody–antigen interfaces; no refined structure registers
a clash.

---

## Maturity

A working proof of concept, not a validated benchmark. Three targets at ~8 samples per arm
cannot reach statistical significance, and no hyperparameter — λ, σ_start, guidance weight,
composite scorer weights — has been tuned against them. Control arms exist for this reason:
`armA_control` templates the complete complex, so a diversity gain cannot be attributed to
template conditioning in general, and `baseline_steered` runs Boltz-2's own potentials, so
it cannot be attributed to steering in general.

Boltz-2 inference has not been run against this code; stages 0/A/B need a GPU. Every
component that does not require model weights is executed and tested, including the runtime
patch and the vendored sampler, which run against a stub Boltz on CPU.

---

## License

MIT — see [LICENSE](LICENSE).
