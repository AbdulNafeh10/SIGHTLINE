# SIGHTLINE

I built SIGHTLINE to investigate whether vision-language models can choose useful viewing angles for object recognition. An object rotates on a motorized turntable while a fixed camera photographs it. The model uses the images to decide which angle to capture next.

The project is currently in prototype testing. I have paused new collection while I work on turntable repeatability, camera consistency, and how recognition should be scored. The examples here are early observations, not results from a completed study.

![Arduino Mega, motor driver, LCD and turntable](docs/assets/prototype-assembly.jpg)

*Prototype setup: Arduino Mega, motor driver, LCD, and turntable.*

## Research question

**Can AI-guided viewpoint selection improve fine-grained object recognition accuracy and image efficiency compared with uniform and random scanning, within individual models and across different models?**

A single image can hide the detail needed to identify an object. A logo may be on the other side, or two similar products may have different ports or markings. I wanted to test whether a model could use what it has already seen to choose a more useful next view, without retraining it.

The comparison needs to consider both recognition and the number of photographs used. Some objects may be identifiable from one view. For others, more images may help, add little information, or lead to a different prediction.

## How it works

A 28BYJ-48 stepper motor rotates an approximately 150 mm 3D-printed platform while the camera stays fixed. An Arduino Mega controls the motor through a ULN2003 driver. The LCD displays the current operation and commanded angle. A Samsung Galaxy S25 Ultra provides the camera feed through Plugcam.

The [Streamlit dashboard](dashboard.py) coordinates rotation and capture, sends images to the models, and saves photographs, predictions, selected angles, and API errors. Recognition can be run again from saved images without operating the turntable.

The [hardware and methodology notes](docs/methodology.md) explain the circuit, experiment design, startup, and data handling.

## Models

| Model | Setup |
|---|---|
| Qwen3-VL-4B-Instruct | Runs locally through llama.cpp on my RTX 4060 laptop |
| Gemini 3.5 Flash Lite | Accessed through Google's API |
| NVIDIA Nemotron 3.5 Super VL | Accessed through NVIDIA's hosted API |

Cost and access influenced these choices. Qwen can run on my laptop, while the free API access available to me from Google and NVIDIA makes small tests with hosted models practical. I plan runs around each account's request allowances. Check [Google's limits](https://ai.google.dev/gemini-api/docs/rate-limits) and [NVIDIA's API access](https://docs.api.nvidia.com/nim/docs/product) for the current allowances.

Using different models also lets me compare how they interpret the same photographs and which views they choose. A model might recognize the category but miss the brand or exact variant. I want to understand whether useful viewpoints depend on the model selecting them.

## Experiment design

The intended comparison uses five photographs for each of five strategies: uniform, random, Qwen-guided, Gemini-guided, and NVIDIA-guided scanning.

Uniform scanning uses evenly spaced angles rounded to the turntable's 5° grid. Random scanning selects unused angles. In a guided scan, the model sees its own photographs so far and chooses the next angle.

Recognition uses the images cumulatively: the third checkpoint includes the first three photographs. Every checkpoint is a fresh request, without earlier answers or the registered object identity. Each model evaluates the uniform and random scans, along with its own guided scan.

Testing whether one model's selected views help another model is a separate comparison that I still need to add.

## Early observations

The [example records](evidence/trials.csv) include a vehicle test where the same five uniform views produced different descriptions:

| Model | Prediction after five photographs |
|---|---|
| Qwen | silver metallic SUV toy model |
| Gemini | Infiniti QX80 toy car model |
| NVIDIA | Infiniti QX60 toy car |

NVIDIA called the vehicle a QX80 after three images and a QX60 after five. Qwen's description stayed at the general category level. This is a useful example of why extra images and high reported confidence should not automatically be treated as better recognition.

The saved viewpoint sequences also differ: some choices cover several sides, while others make small changes around nearby angles. These examples help identify what to investigate next, but they do not establish which strategy or model performs best.

## Running the prototype

The current setup uses Windows and Python 3.12. The camera module uses DirectShow.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:SIGHTLINE_PORT = "COM10"  # Set the Arduino's actual port
$env:GEMINI_API_KEY = "your-key"
$env:NVIDIA_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Start the local Qwen server in a separate terminal:

```powershell
llama-server -hf Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M --alias Qwen3-VL-4B-Instruct --port 9931
```

The [startup instructions](docs/methodology.md#startup) cover firmware, power checks, camera selection, local model files, and recovery after an interruption. [.env.example](.env.example) lists the configuration variables; the app reads environment variables directly and does not load this file automatically.

## Next steps

The turntable has a slight wobble and no encoder, so its recorded angles are commands rather than independent measurements. I need to check its coupling, support, and return-to-zero repeatability before relying on repeated scans.

I also need to control focus, exposure, lighting, and object placement; settle on a consistent scoring rubric; and collect repeated trials with balanced method order. After those checks, I can compare recognition by image count and extend the evaluation to views selected by other models.

[MIT license](LICENSE).
