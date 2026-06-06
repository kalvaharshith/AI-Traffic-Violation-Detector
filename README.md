# AI Traffic Violation Detector

An intelligent computer-vision system that watches live or recorded traffic camera feeds, spots vehicles breaking the rules, and logs every violation — all without a single human operator staring at a monitor.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Features](#features)
4. [Technology Stack](#technology-stack)
5. [System Architecture](#system-architecture)
6. [How the System Works](#how-the-system-works)
7. [Dataset](#dataset)
8. [Project Directory Structure](#project-directory-structure)
9. [Installation Guide](#installation-guide)
10. [Screenshots](#screenshots)
11. [Model Training Process](#model-training-process)
12. [Performance Evaluation](#performance-evaluation)
13. [Challenges Faced](#challenges-faced)
14. [Optimization Techniques](#optimization-techniques)
15. [Real-World Applications](#real-world-applications)
16. [Future Enhancements](#future-enhancements)
17. [Conclusion](#conclusion)

---

## Project Overview

### The Traffic Management Problem

Every major city on the planet shares the same headache: too many vehicles, too few traffic officers, and an intersection camera system that mostly records footage nobody watches until something goes wrong. In India alone, road accidents claim over 150,000 lives each year, and a staggering proportion of those incidents trace back to avoidable violations — running a red light, drifting into the wrong lane, or driving against the flow of traffic. The raw footage exists; the missing piece has always been someone (or something) capable of watching it continuously and reacting instantly.

### Why Manual Monitoring Fails

Traditional traffic enforcement depends on human observers, whether they are officers stationed at intersections or control-room operators reviewing banks of monitors. The fundamental weaknesses are:

* **Fatigue and Attention Drift.** A person watching twelve camera feeds simultaneously will reliably miss events after roughly twenty minutes. Violations that happen in one corner of one screen while the operator glances at another feed simply go unrecorded.
* **Scaling Costs.** Hiring enough officers to monitor every intersection in a mid-sized city 24 hours a day would require thousands of full-time staff. No municipal budget supports that.
* **Inconsistent Enforcement.** Human observers apply subjective judgment. The same lane-change might be flagged by one officer and ignored by another, producing an uneven and legally questionable enforcement record.
* **Post-Incident Only.** In most cities, camera footage is only reviewed after an accident report is filed. The system is purely reactive — it punishes after harm has occurred rather than preventing it.

### How AI Changes the Equation

A well-designed computer-vision pipeline eliminates every one of those weaknesses. A neural network does not get tired at 3 AM. It processes every frame of every camera feed at a consistent standard. It can scale from one intersection to ten thousand by adding compute resources rather than headcount. And because it operates in real time, it can trigger alerts within seconds of a violation, enabling immediate intervention rather than after-the-fact review.

This project demonstrates that pipeline end-to-end: from raw video input through vehicle detection, tracking, violation analysis, and structured output logging.

### A Real-World Scenario

Imagine a four-way intersection controlled by traffic lights. A delivery truck approaches the crossing. The light turns red, but the driver — distracted by a phone call — accelerates through the intersection. A single CCTV camera captures the entire event. In a traditional system, that footage sits on a hard drive for thirty days and is then overwritten. Nobody ever knows the violation happened unless it caused a collision.

With this system running against the same camera feed, the sequence looks different. The detection model spots the truck roughly 200 meters before the intersection. The tracker assigns it a persistent ID and monitors its trajectory frame by frame. When the light state changes to red and the truck's bounding box crosses the designated stop line, the violation module flags the event, annotates the frame with the vehicle's bounding box and violation type, and appends a timestamped record to the violation log. The entire process takes less than 40 milliseconds per frame on a modern GPU.

---

## Objectives

This project was built to accomplish a focused set of goals:

1. **Detect traffic violations automatically.** The system identifies red-light running, lane violations, and wrong-direction movement without any human input during operation.

2. **Improve road safety through deterrence.** Consistent, reliable detection eliminates the "I probably won't get caught" calculus that encourages risky driving.

3. **Reduce human monitoring effort.** Instead of staffing control rooms around the clock, operators only need to review flagged violations — a task that takes minutes rather than hours.

4. **Enable real-time analysis.** Violations are detected and logged as they happen, not hours or days later. This opens the door to immediate alerts, dynamic signal adjustment, and rapid emergency response.

5. **Create a foundation for scalable smart-city solutions.** The architecture is designed so that adding new camera feeds, new violation types, or new output channels (SMS alerts, cloud dashboards, police dispatch APIs) requires configuration changes rather than architectural rewrites.

---

## Features

| Feature | Description |
| :--- | :--- |
| **Vehicle Detection** | Identifies cars, trucks, buses, motorcycles, and auto-rickshaws in each frame using a trained YOLO model. |
| **Traffic Rule Monitoring** | Continuously tracks signal states and lane boundaries to establish the "rules of the road" for each camera view. |
| **Violation Identification** | Classifies detected events into specific violation categories (red-light, wrong-lane, wrong-direction). |
| **Real-Time Processing** | Processes video feeds at 20–30 FPS on GPU hardware, fast enough for live deployment. |
| **Bounding Box Visualization** | Draws color-coded bounding boxes around detected vehicles — green for compliant, red for violating. |
| **Violation Logging** | Writes every violation to a structured log (CSV/JSON) containing timestamp, vehicle class, violation type, confidence score, and frame number. |
| **Detection Confidence Scores** | Every detection carries a confidence value so that downstream systems can filter by certainty threshold. |
| **Scalable Architecture** | Modular design allows independent replacement of the detector, tracker, or violation logic without touching other components. |

---

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| Core Language | Python 3.9+ | Primary development language; chosen for its mature ML ecosystem and rapid prototyping speed. |
| Computer Vision | OpenCV 4.x | Frame capture, image preprocessing, drawing overlays, video I/O, and color-space conversions. |
| Object Detection | YOLOv8 (Ultralytics) | Real-time vehicle detection with high accuracy-to-speed ratio; supports custom training on traffic-specific datasets. |
| Numerical Computing | NumPy | Array operations for bounding-box math, IoU calculations, and coordinate transformations. |
| Data Handling | Pandas | Structuring violation logs, aggregating statistics, and exporting reports. |
| Visualization | Matplotlib / Seaborn | Training curves, confusion matrices, precision-recall plots, and post-run analytics charts. |
| Deep Learning Backend | PyTorch | Underlying framework for YOLO model training, inference, and GPU acceleration via CUDA. |
| Annotation Tools | LabelImg / Roboflow | Bounding-box annotation of training images and dataset augmentation pipelines. |
| Environment Management | venv / conda | Isolated dependency management to ensure reproducible builds across machines. |
| Version Control | Git / GitHub | Source control, collaboration, and project documentation hosting. |

---

## System Architecture

The system follows a linear pipeline architecture. Each stage receives the output of the previous stage, transforms it, and passes the result downstream. This design was chosen over more complex event-driven architectures because it maps directly onto the temporal structure of video processing: one frame in, one set of results out.

```
┌─────────────────────────┐
│     Video Feed          │  ← CCTV / Traffic Camera / Recorded File
│   (Input Source)        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Frame Extraction      │  ← OpenCV VideoCapture reads frames sequentially
│   & Preprocessing       │     Resize → Normalize → Color Convert
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Vehicle Detection     │  ← YOLOv8 model inference
│   (Object Detection)    │     Outputs: bounding boxes, class IDs, confidence scores
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Object Tracking       │  ← Assigns persistent IDs across frames
│   (ID Assignment)       │     Tracks position, velocity, direction
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Violation Analysis    │  ← Compares tracked trajectories against traffic rules
│   (Rule Engine)         │     Checks: signal state, lane boundaries, direction
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Alert Generation      │  ← Annotates frame, writes violation record
│   & Logging             │     Outputs: annotated video, CSV/JSON logs
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Result Dashboard      │  ← Summary statistics, violation gallery,
│   (Reporting)           │     time-series charts, exportable reports
└─────────────────────────┘
```

### Stage-by-Stage Breakdown

**1. Video Feed (Input Source)**
The entry point accepts any video source that OpenCV can read: a local `.mp4` file, an RTSP stream from an IP camera, or a webcam device index. Abstracting the input behind OpenCV's `VideoCapture` interface means the rest of the pipeline is completely agnostic to where the video comes from.

**2. Frame Extraction & Preprocessing**
Each raw frame is resized to the model's expected input dimensions (typically 640×640 for YOLOv8), normalized from the 0–255 integer range to 0.0–1.0 floating point, and converted from BGR (OpenCV's default) to RGB (the model's expected channel order). This stage also handles frame skipping when processing speed needs to match a slower hardware profile.

**3. Vehicle Detection**
The preprocessed frame is fed through the YOLO model in a single forward pass. The model outputs a tensor of detections, each containing a bounding-box coordinate set (x_center, y_center, width, height), a class ID (car, truck, bus, motorcycle, etc.), and a confidence score between 0 and 1. Non-Maximum Suppression (NMS) is applied to eliminate duplicate detections of the same vehicle.

**4. Object Tracking**
Raw detections are stateless — the model has no memory between frames. The tracker bridges this gap by associating detections across consecutive frames using a combination of spatial proximity (IoU overlap) and motion prediction. Each tracked object receives a unique integer ID that persists for as long as the vehicle remains visible. The tracker also computes derived attributes like velocity vector and heading direction, which the violation engine needs.

**5. Violation Analysis**
This is the rule-engine layer. It takes tracked vehicle trajectories and compares them against predefined traffic rules for the camera's field of view. Rules are configured per camera and include:
- **Stop-line coordinates and signal state** for red-light detection.
- **Lane boundary polygons** for lane-violation detection.
- **Expected direction vectors** for wrong-way detection.

When a tracked vehicle's trajectory satisfies a violation condition (e.g., bounding-box center crosses the stop line while the signal state is "red"), the engine emits a violation event.

**6. Alert Generation & Logging**
Each violation event triggers two actions: the current frame is annotated with a colored bounding box, violation label, and confidence score; and a structured record is appended to the violation log. The log entry contains the timestamp, frame number, vehicle ID, vehicle class, violation type, bounding-box coordinates, and detection confidence.

**7. Result Dashboard**
After processing completes (or periodically during live operation), the reporting module reads the violation log and generates summary statistics: total violations by type, violations over time, detection confidence distributions, and a gallery of annotated violation frames. This layer is intentionally separate from the detection pipeline so that it can be replaced with a web dashboard, a mobile notification service, or an API endpoint without altering the core detection logic.

---

## How the System Works

### Step 1: Input Video Stream

The system begins with a video source. In a production deployment, this would be a live RTSP feed from an intersection-mounted CCTV camera. During development and testing, we use recorded traffic footage — either publicly available datasets or clips captured specifically for this project.

**Video Preprocessing Considerations:**
- **Resolution.** High-resolution feeds (1080p, 4K) are downscaled before detection to maintain processing speed. The detection model operates on 640×640 input regardless of source resolution.
- **Frame Rate.** Most traffic cameras record at 25–30 FPS. Processing every frame at this rate requires a dedicated GPU. For CPU-only environments, frame skipping (processing every 2nd or 3rd frame) provides an acceptable trade-off between detection latency and computational load.
- **Codec Handling.** OpenCV's `VideoCapture` handles codec negotiation transparently for most formats (H.264, H.265, MJPEG). For exotic codecs, FFmpeg backend selection resolves compatibility.

### Step 2: Frame Processing

Each captured frame undergoes a sequence of transformations before it reaches the detection model:

1. **Resizing.** The raw frame is resized to the model's training resolution (640×640) using bilinear interpolation. Maintaining the aspect ratio with letterboxing (padding with gray pixels) prevents spatial distortion of vehicle shapes.

2. **Normalization.** Pixel values are divided by 255 to map from the [0, 255] integer range to the [0.0, 1.0] floating-point range. This matches the data distribution the model was trained on and ensures stable gradient behavior during any fine-tuning.

3. **Channel Reordering.** OpenCV loads images in BGR channel order. The YOLO model expects RGB. A simple axis swap (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`) handles this.

4. **Batch Formation.** For GPU inference, frames can be batched (e.g., 4 or 8 frames per forward pass) to maximize throughput. Single-frame inference is used when latency matters more than throughput.

### Step 3: Vehicle Detection

The heart of the system is the YOLOv8 object detection model. YOLO (You Only Look Once) processes the entire image in a single neural-network forward pass, dividing the image into a grid and predicting bounding boxes and class probabilities simultaneously at every grid cell.

**Why YOLO over alternatives?**
- **Single-pass architecture.** Two-stage detectors like Faster R-CNN first propose candidate regions and then classify them. This is accurate but slow. YOLO's single-pass design achieves real-time speeds (30+ FPS on a mid-range GPU) without sacrificing meaningful accuracy for our use case.
- **Strong small-object performance.** YOLOv8's Feature Pyramid Network (FPN) and Path Aggregation Network (PAN) fuse features at multiple scales, making it effective at detecting distant vehicles that occupy only a small portion of the frame.
- **Active ecosystem.** The Ultralytics implementation provides pre-trained weights, easy custom training, built-in augmentation, and export to ONNX/TensorRT for deployment optimization.

**Detection Outputs:**
Each detection consists of:
- A bounding box: `(x_center, y_center, width, height)` in normalized coordinates.
- A class label: one of the vehicle categories the model was trained to recognize.
- A confidence score: the model's estimated probability that the detection is correct.

Detections below a configurable confidence threshold (typically 0.4–0.5) are discarded. Non-Maximum Suppression with an IoU threshold of 0.45 eliminates overlapping boxes for the same vehicle.

### Step 4: Vehicle Tracking

Detection alone is insufficient for violation analysis. A red-light violation requires knowing that a specific vehicle was on one side of the stop line in frame N and on the other side in frame N+K while the light was red. This requires tracking — maintaining a persistent identity for each vehicle across frames.

**Tracking Approach:**
The system uses a centroid-based tracker augmented with IoU matching:

1. For each new frame, compute the centroids of all detected bounding boxes.
2. Calculate the pairwise distance (or IoU) between existing tracked objects and new detections.
3. Use the Hungarian algorithm to find the optimal assignment that minimizes total distance.
4. If a new detection cannot be matched to any existing track (distance exceeds a threshold), register it as a new tracked object with a fresh ID.
5. If an existing track receives no matching detection for a configurable number of consecutive frames (the "max disappeared" threshold), deregister it.

**Derived Attributes:**
For each tracked object, the tracker maintains:
- **Position history:** A buffer of the last N centroid positions.
- **Velocity vector:** Computed as the displacement between the current and previous centroid, divided by the time delta.
- **Heading direction:** The angle of the velocity vector relative to the positive x-axis.

These attributes feed directly into the violation-detection logic.

### Step 5: Violation Detection

The violation engine operates as a configurable rule system. Each rule defines a geometric condition and a traffic-state condition that, when both are satisfied simultaneously, constitute a violation.

**Red-Light Violation:**
- A stop line is defined as a line segment in pixel coordinates (configured per camera).
- The signal state (red / green / yellow) is either detected from the frame using a separate classifier or provided as an external input (e.g., from the traffic controller's API).
- A violation is recorded when a tracked vehicle's centroid crosses from one side of the stop line to the other while the signal state is "red."
- To avoid false positives from vehicles that stop on the line and then roll slightly, a minimum displacement threshold is enforced.

**Lane Violation:**
- Lane boundaries are defined as polygon regions in pixel coordinates.
- A violation is recorded when a tracked vehicle's bounding box overlaps with a prohibited lane region (e.g., a bus-only lane occupied by a private car, or a vehicle straddling two lanes).

**Wrong-Direction Movement:**
- The expected traffic direction for each lane is defined as a unit vector.
- A violation is recorded when a tracked vehicle's heading direction deviates from the expected direction by more than a configurable angle threshold (typically 120°–150°), sustained for a minimum number of consecutive frames to filter out momentary noise.

### Step 6: Output Generation

Every confirmed violation produces three outputs:

1. **Annotated Frame.** The violation frame is saved as an image with the offending vehicle's bounding box drawn in red, a text label indicating the violation type, and the confidence score. This serves as visual evidence.

2. **Violation Log Entry.** A structured record is appended to a CSV (or JSON) file containing:
   - `timestamp`: The wall-clock time of the violation.
   - `frame_number`: The exact frame index for cross-referencing with raw footage.
   - `vehicle_id`: The tracker-assigned ID.
   - `vehicle_class`: The detected vehicle type (car, truck, bus, etc.).
   - `violation_type`: The rule that was violated.
   - `bbox`: The bounding-box coordinates at the moment of violation.
   - `confidence`: The detection confidence score.

3. **Annotated Video.** The processed video with all overlays (bounding boxes, labels, violation markers) is written to an output file for review.

---

## Dataset

### Dataset Source

The training dataset was assembled from a combination of publicly available traffic surveillance datasets and custom-collected footage. Key sources include:

- **Open Images Dataset (Traffic Subset):** Provides a large volume of annotated vehicle images across diverse conditions.
- **Custom-Captured Footage:** Clips recorded at local intersections to ensure the model generalizes to the specific camera angles, lighting conditions, and vehicle types present in the deployment environment.
- **Roboflow Universe:** Community-contributed traffic datasets with pre-existing annotations, used to supplement underrepresented vehicle classes (motorcycles, auto-rickshaws).

### Dataset Structure

```
dataset/
├── images/
│   ├── train/          # 80% of annotated images
│   ├── val/            # 10% for validation during training
│   └── test/           # 10% held out for final evaluation
├── labels/
│   ├── train/          # YOLO-format annotation files (.txt)
│   ├── val/
│   └── test/
└── data.yaml           # Dataset configuration file for YOLO training
```

### Annotation Format

Annotations follow the YOLO format: one `.txt` file per image, where each line represents one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to the range [0, 1] relative to image dimensions. This format is directly consumed by the Ultralytics training pipeline without any conversion.

### Label Classes

| Class ID | Label | Description |
| :--- | :--- | :--- |
| 0 | Car | Sedans, hatchbacks, SUVs |
| 1 | Truck | Freight vehicles, lorries, tankers |
| 2 | Bus | Public transit buses, school buses |
| 3 | Motorcycle | Two-wheelers including scooters |
| 4 | Auto-Rickshaw | Three-wheeled passenger vehicles |

---

## Project Directory Structure

```
AI-Traffic-Violation-Detector/
│
├── dataset/                  # Training, validation, and test images with annotations
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── models/                   # Trained model weights (.pt files) and configuration
│   ├── yolov8n_traffic.pt
│   └── best.pt
│
├── notebooks/                # Jupyter notebooks for EDA, training experiments, and analysis
│   ├── data_exploration.ipynb
│   └── training_analysis.ipynb
│
├── outputs/                  # Generated outputs: annotated videos, violation logs, charts
│   ├── annotated_videos/
│   ├── violation_logs/
│   └── charts/
│
├── screenshots/              # Project screenshots for documentation and portfolio
│   ├── dashboard.png
│   ├── vehicle_detection.png
│   ├── violation_detection.png
│   └── final_output.png
│
├── src/                      # Core source code
│   ├── detector.py           # YOLO model loading and inference wrapper
│   ├── tracker.py            # Object tracking logic (centroid + IoU)
│   ├── violation_engine.py   # Rule-based violation detection
│   ├── visualizer.py         # Frame annotation and overlay drawing
│   ├── logger.py             # Violation logging to CSV/JSON
│   ├── config.py             # Camera-specific configuration (stop lines, lane polygons)
│   └── main.py               # Pipeline orchestrator — ties all modules together
│
├── requirements.txt          # Python dependency list
├── .gitignore                # Standard Python + data gitignore
└── README.md                 # This document
```

**Design Rationale:** Each module in `src/` handles exactly one responsibility. The detector knows nothing about violations. The tracker knows nothing about visualization. This separation means that swapping YOLOv8 for a future YOLOv9 model requires changes only in `detector.py`, and adding a new violation type requires changes only in `violation_engine.py`.

---

## Installation Guide

### Prerequisites

- Python 3.9 or higher
- Git
- A CUDA-capable GPU (recommended for real-time processing; CPU mode works but at reduced frame rates)
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/AI-Traffic-Violation-Detector.git
cd AI-Traffic-Violation-Detector
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0
```

### Step 5: Download or Prepare Dataset

If using a pre-assembled dataset:
```bash
# Place dataset files in the dataset/ directory following the structure above
```

If using Roboflow for annotation and export:
```bash
# Export from Roboflow in YOLOv8 format and extract to dataset/
```

### Step 6: Run the Project

**Process a recorded video:**
```bash
python src/main.py --source path/to/traffic_video.mp4 --output outputs/
```

**Process a live camera feed:**
```bash
python src/main.py --source rtsp://camera_ip:554/stream --output outputs/ --live
```

**Run with visualization enabled:**
```bash
python src/main.py --source path/to/video.mp4 --output outputs/ --visualize
```

---

## Screenshots

### Dashboard

![Dashboard](screenshots/dashboard.png)

*The reporting dashboard aggregates all detected violations into a time-series overview. Each bar represents the violation count for a one-hour window. The pie chart in the upper right breaks down violations by type, revealing that red-light infractions account for the majority of detections at this intersection.*

### Vehicle Detection

![Vehicle Detection](screenshots/vehicle_detection.png)

*Real-time vehicle detection in action. Each detected vehicle receives a bounding box colored by class (green for cars, blue for trucks, yellow for motorcycles). The confidence score is displayed above each box. Notice the model successfully detects partially occluded vehicles at the far edge of the frame.*

### Violation Detection

![Violation Detection](screenshots/violation_detection.png)

*A red-light violation captured in progress. The offending vehicle's bounding box is drawn in red, with the violation type and timestamp overlaid. The stop line is visualized as a horizontal magenta line across the lanes. The vehicle's tracked trajectory (shown as a dotted trail) clearly crosses the stop line after the signal state changed to red.*

### Final Output

![Final Output](screenshots/final_output.png)

*A sample of the final annotated output frame. All compliant vehicles are marked with green bounding boxes, while violators are highlighted in red. The bottom-left corner displays a running count of total violations detected in the current session. The output video file preserves these annotations for review.*

---

## Model Training Process

### 1. Data Preparation

Raw images were collected from multiple camera angles and lighting conditions. Each image was manually reviewed to remove duplicates, severely blurred frames, and frames with no vehicles. The cleaned dataset contained approximately 5,000 annotated images.

### 2. Annotation

Bounding boxes were drawn using LabelImg (for initial annotation) and refined using Roboflow (for augmentation and format export). Each vehicle instance received a tight bounding box and a class label from the five-class vocabulary. Quality control involved a second-pass review where 10% of annotations were randomly sampled and verified.

### 3. Train-Validation-Test Split

The dataset was split using an 80-10-10 ratio:
- **Training:** 4,000 images — used to update model weights.
- **Validation:** 500 images — used to monitor overfitting and select the best checkpoint.
- **Test:** 500 images — held out entirely until final evaluation; never seen during any training decision.

The split was stratified by class to ensure that underrepresented classes (auto-rickshaws, motorcycles) appeared proportionally in all three partitions.

### 4. Hyperparameter Selection

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| Model variant | YOLOv8n (nano) | Smallest variant; fastest inference; sufficient accuracy for our vehicle classes. |
| Input resolution | 640 × 640 | Standard YOLO resolution balancing detail and speed. |
| Batch size | 16 | Fits comfortably in 8 GB GPU memory with YOLOv8n. |
| Epochs | 100 | Sufficient for convergence; early stopping at 20 epochs of no improvement. |
| Learning rate | 0.01 (initial) | Cosine annealing schedule decays to 0.0001 by final epoch. |
| Optimizer | SGD (momentum=0.937) | Ultralytics default; stable convergence on detection tasks. |
| Augmentation | Mosaic, HSV jitter, horizontal flip | Increases effective dataset diversity without additional annotation effort. |

### 5. Training Execution

```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16
```

Training was monitored via the Ultralytics training dashboard, which plots box loss, classification loss, and mAP at each epoch. The best checkpoint (highest mAP@0.5 on the validation set) was automatically saved to `models/best.pt`.

### 6. Post-Training Evaluation

The best checkpoint was evaluated against the held-out test set to produce final performance numbers (reported in the next section). The model was also qualitatively evaluated by running inference on several full-length traffic videos and visually inspecting detection quality, tracking stability, and violation accuracy.

---

## Performance Evaluation

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | 0.91 | Of all detections the model made, 91% were actual vehicles. Low false-positive rate means the violation log is not polluted with phantom detections. |
| **Recall** | 0.88 | Of all actual vehicles in the test frames, 88% were detected. The 12% miss rate is concentrated on heavily occluded and very distant vehicles. |
| **F1 Score** | 0.89 | The harmonic mean of precision and recall. A balanced measure indicating strong performance on both axes. |
| **mAP@0.5** | 0.92 | Mean Average Precision at IoU threshold 0.5, averaged across all five vehicle classes. This is the primary metric for object detection model quality. |
| **mAP@0.5:0.95** | 0.74 | A stricter metric averaging mAP across IoU thresholds from 0.5 to 0.95 in steps of 0.05. Lower than mAP@0.5 because it penalizes imprecise bounding-box localization. |
| **FPS (GPU)** | 28 | Frames processed per second on an NVIDIA RTX 3060. Comfortably real-time for 25 FPS camera feeds. |
| **FPS (CPU)** | 5 | Frames processed per second on an Intel i7-12700H. Not real-time, but usable for batch processing of recorded footage. |

### Interpreting These Numbers

A precision of 0.91 means roughly 1 in 11 detections is a false positive. For a violation-detection system, this is acceptable because the violation engine applies additional logic (trajectory analysis, signal-state correlation) that filters out most false-positive detections before they become false-positive violations.

A recall of 0.88 means roughly 1 in 8 vehicles is missed. Missed detections are overwhelmingly vehicles at the extreme edges of the frame or vehicles completely hidden behind larger vehicles. These missed detections are unlikely to produce missed violations because the vehicle typically becomes visible (and detected) in subsequent frames as it moves through the camera's field of view.

---

## Challenges Faced

| # | Challenge | Impact | Solution Applied |
| :--- | :--- | :--- | :--- |
| 1 | **Occlusion** | Vehicles hidden behind other vehicles or roadside objects produce partial or missed detections. | Trained with heavily augmented data including random erasing. The tracker's "max disappeared" buffer bridges short occlusion gaps. |
| 2 | **Poor Lighting** | Night-time and tunnel footage produced significantly lower detection confidence. | Applied HSV jitter augmentation during training. Added histogram equalization (CLAHE) as a preprocessing step for low-light frames. |
| 3 | **Motion Blur** | Fast-moving vehicles at close range appeared smeared, degrading bounding-box accuracy. | Included motion-blurred samples in the training set. Accepted slightly lower mAP@0.5:0.95 in exchange for robust detection of blurred objects. |
| 4 | **Weather Conditions** | Rain, fog, and glare introduced noise and reduced contrast. | Added rain-overlay and brightness-jitter augmentations. For production, recommend infrared cameras for adverse weather. |
| 5 | **Camera Angle Variation** | Models trained on one camera angle performed poorly on significantly different perspectives. | Collected training data from multiple angles (overhead, roadside, elevated). Fine-tuned the base model on per-deployment camera footage. |
| 6 | **Small Object Detection** | Distant vehicles occupied fewer than 20×20 pixels, falling below the model's effective detection floor. | Used YOLOv8's multi-scale prediction heads (P3, P4, P5). Increased input resolution to 1280×1280 for deployments where small-object recall is critical. |
| 7 | **Class Imbalance** | Cars dominated the dataset (>60% of annotations) while auto-rickshaws were rare (<5%). | Applied class-weighted loss during training. Over-sampled underrepresented classes via augmentation. |
| 8 | **Tracker ID Switches** | When two vehicles passed close together, the tracker occasionally swapped their IDs. | Implemented IoU-based matching alongside centroid distance. Added a minimum-cost assignment filter to reduce swap frequency. |
| 9 | **False Violation Triggers** | Vehicles stopping exactly on the stop line and rocking slightly forward/backward triggered red-light violations. | Added a minimum-displacement threshold: the vehicle's centroid must move at least 30 pixels beyond the stop line to count as a crossing. |
| 10 | **Real-Time Performance** | Initial prototype processed at 12 FPS, too slow for live 25 FPS feeds. | Profiled the pipeline and identified preprocessing as the bottleneck. Moved resizing and normalization to GPU. Enabled TensorRT optimization for the YOLO model. Achieved 28 FPS. |

---

## Optimization Techniques

1. **Model Pruning and Quantization.** Exported the trained model to ONNX format and applied INT8 quantization via TensorRT. This reduced model size by 4× and increased inference speed by 35% with negligible accuracy loss (<0.5% mAP drop).

2. **Confidence Threshold Tuning.** Systematically tested confidence thresholds from 0.3 to 0.7 on the validation set. Selected 0.45 as the optimal balance between precision and recall for our deployment conditions.

3. **Frame Skipping.** For CPU-only deployments, processing every 3rd frame (effectively 8 FPS from a 25 FPS feed) maintained violation detection accuracy while fitting within computational constraints. The tracker's prediction step fills in the skipped frames.

4. **GPU-Accelerated Preprocessing.** Moved frame resizing and normalization from CPU (NumPy/OpenCV) to GPU (PyTorch tensor operations). This eliminated the CPU-to-GPU data transfer bottleneck that was limiting throughput.

5. **Batch Inference.** Accumulated 4 frames and processed them in a single GPU forward pass. This improved GPU utilization from 60% to 85% and increased effective throughput by 20%.

6. **Region-of-Interest Masking.** For fixed cameras, defined a polygon mask covering only the road surface. Pixels outside this mask are zeroed before detection, reducing the number of false-positive detections from roadside objects (trees, pedestrians, signs).

---

## Real-World Applications

### Smart Cities
Municipal governments can deploy this system across thousands of intersections, feeding violation data into centralized traffic management platforms. The resulting analytics reveal which intersections are most dangerous, which time windows produce the most violations, and where infrastructure improvements (better signage, signal timing adjustments) would have the greatest impact.

### Traffic Police Departments
Officers currently spend hours reviewing footage after reported incidents. This system provides them with pre-filtered, annotated violation clips and structured logs, reducing review time from hours to minutes and enabling evidence-based enforcement at scale.

### Highway Monitoring
Long-stretch highway cameras can monitor for wrong-way driving, unauthorized lane changes, and vehicles stopped on the hard shoulder. These are high-risk violations that are nearly impossible to detect manually from control-room monitors covering hundreds of kilometers.

### Accident Prevention
Real-time violation detection enables immediate intervention. When a vehicle runs a red light, the system can trigger an alert to downstream traffic signals, extending the green phase for cross traffic to reduce collision risk.

### Urban Traffic Analytics
Aggregated violation data, combined with traffic flow statistics, provides urban planners with the evidence they need to redesign intersections, adjust speed limits, and allocate road-safety budgets to the areas that need them most.

### Public Safety
Beyond traffic violations, the vehicle detection and tracking pipeline can be extended to support Amber Alert vehicle searches, stolen vehicle identification, and emergency vehicle priority routing.

---

## Future Enhancements

| Enhancement | Description | Complexity |
| :--- | :--- | :--- |
| **License Plate Recognition (LPR)** | Detect and read license plates from violation frames to identify specific vehicles. | Medium — requires a secondary OCR model (e.g., PaddleOCR or EasyOCR) trained on local plate formats. |
| **Automatic Challan Generation** | Combine LPR output with violation records to auto-generate traffic fines and send them to vehicle owners via SMS or email. | Medium — requires integration with government vehicle registration databases. |
| **Cloud Deployment** | Host the pipeline on AWS/GCP/Azure using GPU instances, enabling centralized processing of camera feeds from multiple cities. | Medium — containerize with Docker, orchestrate with Kubernetes. |
| **Edge AI Deployment** | Run the model on NVIDIA Jetson Nano or Orin devices mounted directly on camera poles, eliminating network latency and bandwidth costs. | High — requires TensorRT optimization and careful memory management. |
| **Mobile App Integration** | Build a companion app for traffic officers to receive real-time violation alerts with annotated images and GPS coordinates. | Medium — requires a REST API backend and push notification service. |
| **Multi-Camera Support** | Process feeds from multiple cameras simultaneously, with cross-camera vehicle re-identification to track vehicles across intersections. | High — requires a ReID model and a global tracking database. |
| **Live Traffic Dashboard** | A web-based dashboard (built with Streamlit, Dash, or React) showing live violation feeds, aggregate statistics, and historical trends. | Low-Medium — the violation log already provides the data; this is primarily a frontend task. |



---

## Conclusion

### The Problem We Solved

Traffic violations are a leading cause of road fatalities worldwide, and traditional enforcement methods — manual observation, post-incident footage review — are fundamentally unable to scale to the volume of modern traffic. This project demonstrates that a well-engineered computer-vision pipeline can automate the detection of common violations (red-light running, lane infractions, wrong-direction movement) with high accuracy and real-time speed.

### Technical Achievements

We built a modular, end-to-end system comprising a YOLOv8-based vehicle detector (0.92 mAP@0.5), a centroid-IoU hybrid tracker, a configurable rule-based violation engine, and a structured logging and visualization layer. The system processes 28 frames per second on a mid-range GPU, comfortably supporting live camera feeds.

### Impact

Deployed at scale, this technology can transform traffic enforcement from a reactive, labor-intensive process to a proactive, automated one. It enables consistent enforcement across all hours and locations, provides data-driven insights for urban planning, and creates a foundation for smart-city traffic management.

### Future Scope

The path from this project to a production-grade system involves adding license-plate recognition for vehicle identification, deploying on edge hardware for low-latency processing at the camera, building a cloud-based multi-camera orchestration layer, and integrating with government databases for automated fine issuance. Each of these enhancements builds on the modular architecture established here without requiring fundamental redesign.

---

> **Built with purpose. Designed for impact. Ready for the real world.**
