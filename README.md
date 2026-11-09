# SIGHTLINE

SIGHTLINE is a research prototype for studying whether vision-language models can choose useful viewpoints for fine-grained object recognition. A stationary camera photographs an object on a motorized turntable; a Streamlit application coordinates capture, model requests, and saved evidence.

**This is ongoing research. Experimental collection is paused while I address mechanical repeatability, camera control, and evaluation methodology. The retained runs are exploratory evidence, not a completed study or an accuracy ranking.**

## Why choose the next view?

A recognition model normally receives whichever image a camera captures. That view can hide a logo, connector, marking, or distinctive shape. I started SIGHTLINE to investigate whether a model could inspect the views already available and choose another angle that reveals useful identifying information, without retraining it.

The research question is: **Can AI-guided viewpoint selection improve fine-grained object recognition accuracy and image efficiency compared with uniform and random scanning, within individual models and across different models?**

The purpose is to test whether it helps, when it helps, and where it fails. Some objects may need only one image. Extra images may add little, or leave a model with the same mistaken interpretation. Different models may also find different evidence useful: recognizing a category, brand, product family, and exact variant are separate achievements.

![Turntable, controller, driver and LCD assembled for acquisition](docs/assets/prototype-assembly.jpg)

*Prototype assembly with the platform, object, Arduino, driver, and LCD. The phone camera is outside the frame. This photograph documents construction, not angular precision. Rotated for readability; metadata removed.*

## Implemented system

The active application is [dashboard.py](dashboard.py), with [camera.py](camera.py), [motor_controller.py](motor_controller.py), and the [LCD controller firmware](firmware/Sightline_Controller/Sightline_Controller.ino). It registers objects, collects separate scans, asks for viewpoints during guided acquisition, evaluates saved image prefixes, preserves raw responses and failures, supports retrying recognition, and exports records for review.

| Evaluator | Requested model ID in code and retained records | Execution |
|---|---|---|
| Qwen | `Qwen3-VL-4B-Instruct` | Local llama.cpp; cached GGUF is Q4_K_M |
| Gemini | `gemini-3.5-flash-lite` | Google API |
| NVIDIA Nemotron | `nvidia/nemotron-3.5-super-vl-120b-a12b` | NVIDIA hosted API |

These are requested IDs, not independently verified immutable weight revisions. The local server's alias alone cannot prove its loaded weights. Different representations and training may produce different interpretations and view choices; model size is not an assumed accuracy ordering. Earlier provider probes are outside the active research application.

The hardware uses an ELEGOO Mega 2560, a 28BYJ-48 5V motor and ULN2003 driver, an approximately 150 mm printed platform and mount, and an LCD1602A display with breadboard wiring. The reported acquisition setup uses a Samsung Galaxy S25 Ultra through Plugcam and a Windows laptop with an RTX 4060 Laptop GPU. Photographs and firmware support the physical construction and control interface; phone/GPU model names and platform diameter are creator-supplied specifications. See [hardware verification and startup](docs/methodology.md#hardware-and-startup) for what remains unverified.

## Experimental design and present evidence

Five strategies each receive five photographs: uniform, random, Qwen-guided, Gemini-guided, and NVIDIA-guided. Each starts at operator-aligned 0°. Uniform spacing is rounded to the 5° grid: `0, 70, 145, 215, 290`. Random scanning samples unused grid angles after 0°, with its seed recorded.

Recognition checkpoint *k* contains photos 1 through *k* from that trial in a fresh request. Guided selection receives only its own scan's photographs and angles, without registered identity or recognition answers. All three models evaluate uniform/random baselines; each evaluates its own guided scan. **Cross-model transfer is not implemented for the three current guided methods.** Completing it requires evaluating each guided scan with the other two recognizers.

The private data audit on 9 November 2026 found:

| Retained records | Testing | Research |
|---|---:|---:|
| Registered object IDs | 4 | 20 |
| Registered IDs with captures | 3 | 4 |
| Images / capture records | 64 | 95 |
| Trials with five captured views | 11 | 16 |
| Complete five-strategy capture batches | 1 | 3 |
| Recognition checkpoint records marked `ok` | 138 | 90 |
| Eligible retained checkpoints never attempted | 24 | 65 |
| Failed recognition records | 0 | 0 |
| Selector attempts, including failures | 12, none failed | 51, one failed |

Capture completion is not evaluation completion. The `Research` folder name does not certify final results. All manifest paths resolved at audit time, all recorded image checksums matched, and no duplicate capture or evaluation keys were found. Neither workspace contained saved manual reviews. Older Testing prompts and logging formats differ from Research and should not be pooled. [Inventory and source digests](evidence/inventory.json) make the counts inspectable; full private datasets are excluded.

![Research collection coverage](docs/assets/collection-coverage.png)

*Counts from the retained Research manifests: 20 registrations, four IDs with captures, three complete capture batches. This is collection coverage, not a count of scientifically validated observations.*

The public [evidence excerpt](evidence/provenance.json) contains one vehicle batch: 25 photographs, 45 cumulative recognition records, 12 selector attempts, and 45 recognition-attempt records. On the same five uniform images, Qwen returned “silver metallic SUV toy model,” Gemini “Infiniti QX80 toy car model,” and NVIDIA “Infiniti QX60 toy car.” NVIDIA's intermediate predictions changed between QX60 and QX80. These traceable disagreements illustrate category versus specific identification; they do not establish comparative accuracy. Exact product ground truth still requires an independent product reference.

## Run and return to the project

Validated with Python **3.12.14 on Windows**, Streamlit **1.67.0**, and the pinned dependencies. Other Python versions and platforms have not been tested. The camera implementation uses Windows DirectShow.

```powershell
# From the repository directory, using an installed Python 3.12:
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:SIGHTLINE_PORT = "COM10"  # Replace with the actual Arduino port
$env:GEMINI_API_KEY = "your-key"
$env:NVIDIA_API_KEY = "your-key"
# Optional: point to a private directory containing both historical workspaces.
$env:SIGHTLINE_DATA_ROOT = "D:\SIGHTLINE-private"
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Use session environment variables or your usual local secret manager. [.env.example](.env.example) documents the variables; the application does **not** load `.env` files automatically. Missing cloud keys block that evaluator. Guided Research acquisition requires its selector to be available. Without `SIGHTLINE_DATA_ROOT`, new workspaces are created beside the source. The public excerpt is not automatically loaded as live Research data.

Start llama.cpp separately with the cached model and vision projector; the [continuation instructions](docs/methodology.md#local-qwen-and-software-startup) give the explicit filenames, checked runtime version, and recovery procedure. On the audited machine, the generic `python` and `py -3` Store launchers failed; checks used an isolated Python 3.12 environment. Verify your interpreter before following the commands.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\evidence.py figures
# Read-only inventory of private data; prints aggregates, never invokes a model:
.\.venv\Scripts\python.exe tools\evidence.py audit D:\SIGHTLINE-private
```

## What needs to happen next

The acquisition and model-request paths are implemented and supported by retained captures and responses. Software regression tests and a firmware build are verified; this review performed no physical runs or new model inference.

1. Characterize wobble, angular return repeatability, coupling, and object stability. Inspect support geometry and use fiducial/reference measurements before choosing mechanical changes.
2. Fix camera focus, exposure, lighting, placement, and framing; record the actual delivered resolution and software/model configurations.
3. Freeze prompts and an independent category/brand/family/exact-identity rubric; review labels before computing accuracy. Define repeats and balance method order.
4. Collect the remaining controlled object trials and repeated random scans; evaluate missing checkpoints only after their inputs are accepted.
5. Add cross-model transfer, then analyze paired object-level differences and uncertainty. Check accuracy versus image count, failure rates, and the first reviewed correct checkpoint without treating cumulative checkpoints as independent observations.

The disc has visible/reported slight tilt and no encoder. Commanded angles are not measured angles, and shared hardware does not guarantee equal mechanical effects across strategies. Coupling inspection, bearing support, or an encoder are possible improvements, not installed features. [Methodology and continuation](docs/methodology.md) records the controls, limitations, recovery steps, and reliability changes needed to resume the work.

Code is [MIT licensed](LICENSE). Hardware photographs are original project images. Model weights, camera software, SDKs, and Arduino libraries retain their own licenses and service terms and are not redistributed here. Printable CAD/STL files were not present in the supplied project.
