# AI-Based Emotion Detection System
## College Project Report Template

> Replace every bracketed placeholder with your own information. Add screenshots and measured results from your own run. Do not report invented accuracy, timing, or test results.

**Student name:** [Your name]  
**Roll number:** [Your roll number]  
**Department / Institution:** [Department and institution]  
**Guide:** [Project guide]  
**Academic year:** [Academic year]

## Abstract

This project implements a web-based facial-expression analysis system using Python and a pre-trained model. Users can upload an image or activate a browser webcam to view detected faces and predicted expression categories. The system uses DeepFace for model inference, OpenCV for image processing, and Streamlit for the user interface. Results include a predicted category, a score distribution, annotated face regions, and optional CSV export. The project focuses on integrating existing AI technology into an accessible demonstration rather than training a new model. Model predictions are uncertain and describe facial-expression categories; they do not establish a person's internal emotional state.

## Chapter 1 — Introduction

### 1.1 Background
Computer vision systems can detect faces and classify visual patterns in images. Facial-expression classification applies machine-learning models to estimate labels associated with visible facial configurations.

### 1.2 Problem statement
Running expression-analysis models often requires programming and model-integration knowledge. This project provides a simple interface for demonstrating image-based and webcam-based inference.

### 1.3 Objectives
- Integrate a pre-trained facial-expression model.
- Accept supported image uploads.
- Provide live webcam processing.
- Visualize results and offer CSV export.
- Test input validation, output normalization, and basic application components.
- Document limitations and responsible-use considerations.

### 1.4 Scope
The project is an educational prototype. It is not a clinical tool and does not determine actual feelings, intent, honesty, personality, or mental health.

## Chapter 2 — Literature Review

Describe face detection, image preprocessing, convolutional neural networks, transfer learning, facial-expression datasets, and pre-trained facial-analysis software. Compare at least three relevant papers or documented systems. For each source, summarize its method, dataset, findings, limitations, and relevance. Verify references rather than copying unverified claims.

Potential starting points:
- DeepFace library documentation: https://github.com/serengil/deepface
- FER-2013 dataset: https://www.kaggle.com/datasets/msambare/fer2013
- OpenCV documentation: https://docs.opencv.org/

## Chapter 3 — System Analysis and Design

### 3.1 Functional requirements
- Upload a JPEG, PNG, or WebP image.
- Identify face regions and estimate expression categories.
- Display category scores and annotated output.
- Start and stop webcam video.
- Download score values as CSV.

### 3.2 Non-functional requirements
- Clear error messages for invalid input.
- Modular code and automated tests.
- Reasonable responsiveness on CPU hardware.
- Responsible handling of image data and explicit acknowledgement of model uncertainty.

### 3.3 Architecture
Image upload or webcam frame → image preprocessing → face detection and pre-trained expression inference → result normalization → annotation and dashboard display.

### 3.4 Hardware and software
Record your computer model, CPU, RAM, OS version, Python version, package versions, and webcam details.

## Chapter 4 — Implementation

### 4.1 Modules
- `app.py`: Streamlit user interface, charts, webcam session, and CSV export.
- `emotion_engine.py`: input validation, DeepFace inference, result normalization, and bounding-box annotation.
- `tests/test_emotion_engine.py`: unit tests with mocked model responses.
- `.github/workflows/tests.yml`: automated compile and test checks.

### 4.2 Model method
The application calls the pre-trained DeepFace emotion-analysis interface with OpenCV face detection. Record the actual installed DeepFace/TensorFlow versions and any model configuration used in your experiment. Do not claim to have trained a model as part of this implementation.

### 4.3 Webcam processing
The browser supplies video to the WebRTC component. The app samples frames to reduce computation, runs inference on selected frames, and displays the latest annotation.

## Chapter 5 — Testing and Results

### 5.1 Test cases

| Test | Expected result | Actual result |
|---|---|---|
| Missing image | Input validation error | [Fill in] |
| Empty image | Input validation error | [Fill in] |
| Grayscale input | Rejected with clear message | [Fill in] |
| Mock model output | Normalized prediction | [Fill in] |
| Multiple mock faces | Both results preserved | [Fill in] |
| Annotation | Output image retains expected dimensions | [Fill in] |
| Valid image in UI | Results displayed | [Fill in] |
| Webcam access | Stream starts with permission | [Fill in] |
| CSV export | CSV can be downloaded and opened | [Fill in] |

### 5.2 Performance evaluation

Run at least 20 image analyses after warming up the model. Record:

| Metric | Measured result |
|---|---|
| Number of analysed images | [Fill in] |
| Mean inference latency | [Fill in] |
| Median inference latency | [Fill in] |
| Minimum / maximum latency | [Fill in] |
| Detected faces / expected faces | [Fill in] |
| Failed detections | [Fill in] |
| Hardware and software environment | [Fill in] |

Do not use illustrative numbers as results. Note that inference time depends on CPU/GPU, image size, package versions, and model initialization.

### 5.3 Classification evaluation (optional)
To report accuracy, use a suitable labelled held-out dataset and compare predicted labels against its defined reference labels. Report class-wise precision, recall, F1-score, a confusion matrix, dataset split, preprocessing, and limitations. Do not infer accuracy from model score values alone. Dataset labels are not perfect ground truth for internal emotion.

### 5.4 Screenshots
Insert actual screenshots of:
1. Application home page.
2. Image upload.
3. Annotated result and category scores.
4. CSV output.
5. Webcam stream.
6. Successful automated test run.

## Chapter 6 — Limitations, Privacy, and Responsible Use

Lighting, pose, occlusion, image quality, population differences, and model training data can affect performance. Visible expression is not a reliable or complete measure of a person's internal state. Do not use this prototype for high-impact decisions. Obtain consent for images and avoid storing or sharing personal face imagery.

## Chapter 7 — Conclusion and Future Scope

Summarize what you actually implemented and tested. Possible future work includes a carefully validated model comparison, accessibility improvements, benchmarking on labelled data, and configurable privacy-preserving deployment.

## References

Include complete citations for research papers and datasets actually consulted. At minimum, cite the DeepFace, Streamlit, streamlit-webrtc, and OpenCV documentation links provided in the README.

## Appendix

Include installation commands, dependency versions, selected test output, and any additional diagrams.
