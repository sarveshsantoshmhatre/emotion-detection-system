# AI-Based Emotion Detection System

A college-project web application that uses a pre-trained model to estimate facial-expression categories from an uploaded image or a live webcam stream. It now includes a standalone responsive website that can be deployed with Docker to Render or Railway.

> **Important limitation:** This application estimates visible facial-expression categories. It cannot reliably determine a person's internal emotional state, and must not be used for medical diagnosis, hiring, discipline, surveillance, or other high-impact decisions.

## Features

- Upload JPG, JPEG, PNG, or WebP images
- Detect one or more faces and display expression labels
- Show the model's expression score distribution
- Export one selected face's score table as CSV
- Analyze periodic frames from a browser webcam stream
- Automated tests and a GitHub Actions test workflow
- College project report template

## Technology stack

- Python
- FastAPI + responsive HTML/CSS/JavaScript website (Docker deploy)
- Streamlit interface (local alternative)
- DeepFace (pre-trained inference)
- OpenCV
- TensorFlow/Keras backend as required by the selected model
- streamlit-webrtc
- NumPy and Pandas
- pytest

## Requirements

- Ubuntu 22.04/24.04 or another supported Linux distribution
- Python 3.10 or 3.11 recommended; check dependency compatibility for your platform
- Internet access for installing packages and the first model-weight download
- Webcam for live video (optional)
- A CPU is sufficient for a demonstration; inference may be slow. A compatible GPU is optional.

## Install on Ubuntu

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev \
  build-essential libgl1 libglib2.0-0 git

git clone https://github.com/sarveshsantoshmhatre/emotion-detection-system.git
cd emotion-detection-system

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (normally http://localhost:8501).

The first model inference can download model weights and take longer than subsequent predictions.

## Deploy the standalone website to Render or Railway

The repository now includes a separate website with a live camera preview, face bounding boxes, expression labels, and score bars. It sends selected camera frames to the Python server for analysis; it does not record a video file.

### Deploy to Render

1. Sign in to [Render](https://render.com/) and connect your GitHub account.
2. Create a **New Web Service** and select `sarveshsantoshmhatre/emotion-detection-system`.
3. Choose the `main` branch and select **Docker** as the runtime (Render can build the root `Dockerfile`).
4. Deploy. The included `render.yaml` describes the service and health check.
5. Open the generated HTTPS URL, press **Start Camera**, and allow browser camera permission.

### Deploy to Railway

1. Sign in to [Railway](https://railway.com/) and create a project from GitHub.
2. Select `sarveshsantoshmhatre/emotion-detection-system`.
3. Railway detects the root `Dockerfile`; the included `railway.json` sets the health-check path.
4. Deploy and generate a public domain in the service's networking settings.
5. Open the HTTPS URL and press **Start Camera**.

### Run the standalone website locally

Install the requirements, then run:

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. The first model inference may download model weights and take longer. The deployment image needs enough memory and disk for TensorFlow and DeepFace; free hosting plans may be too limited or slow for this workload.

## Use the app

1. Open **Image Analysis** and upload a clear image with a visible face.
2. Select **Analyze Image**.
3. Review the annotated image, predicted category, score chart, and CSV export.
4. Open **Live Webcam**, select Start, and grant browser camera permission.
5. Select Stop to end the webcam session.

### Webcam and deployment notes

Camera access is generally available on localhost or HTTPS sites. Remote deployments may require WebRTC network configuration and a TURN server if peer-to-peer connections cannot be established. The configured public STUN server is a demonstration setting, not a guaranteed production setup.

## Tests

Install dependencies, then run:

```bash
pytest -v
python -m compileall -q app.py emotion_engine.py tests
```

The unit tests mock model inference; they do not validate real-world model accuracy. Test the live application with consented sample images before presenting it.

## Project structure

```text
.
├── app.py                 # Streamlit interface
├── web_app.py             # FastAPI website backend
├── static/index.html      # Responsive live-camera website
├── Dockerfile             # Render/Railway deployment
├── render.yaml
├── railway.json
├── emotion_engine.py
├── requirements.txt
├── tests/
│   └── test_emotion_engine.py
├── docs/
│   └── PROJECT_REPORT.md
└── .github/
    └── workflows/
        └── tests.yml
```

## Evaluation plan

Record performance on your own test setup; do not invent results.

- Mean and median inference latency after warm-up
- Number of images tested and faces detected
- Face-detection failure rate
- A confusion matrix and accuracy only when compared with a clearly defined labelled test dataset and its ground-truth labels
- Hardware, operating system, dependency versions, and model details

Do not treat the model's score output as the probability that a person truly feels a particular emotion. If you evaluate classification accuracy, disclose class imbalance, dataset limitations, and the fact that facial-expression labels are imperfect proxies.

## Privacy and responsible use

- Use images only with appropriate consent.
- Do not commit personal photos, downloaded model weights, secrets, or large datasets.
- This project does not intentionally write uploaded images or a prediction history to disk.
- Predictions are sensitive and uncertain; do not use them to make decisions about people.

## References

- DeepFace: https://github.com/serengil/deepface
- Streamlit documentation: https://docs.streamlit.io/
- streamlit-webrtc: https://github.com/whitphx/streamlit-webrtc
- OpenCV documentation: https://docs.opencv.org/
- FER-2013 dataset (possible research/evaluation source): https://www.kaggle.com/datasets/msambare/fer2013

See [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for the report template.
