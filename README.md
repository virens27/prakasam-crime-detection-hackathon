# AI-Powered Crime Detection in Crowded Public Areas

**Mission Y4 – Prakasam Police Hackathon 2026**
**Team:** [Your Team Name] | **Challenge:** 03 – AI-Powered Crime Detection in Crowded Public Areas

---

## Problem Statement

Crowded public areas (markets, bus stands, festivals, public events) make it difficult for human CCTV operators to manually monitor every feed for criminal activity such as weapon use, physical assault, or abandoned/suspicious objects. By the time an incident is noticed, response is often delayed. This project aims to assist (not replace) human monitoring by automatically flagging high-risk events in real time so operators and patrol units can respond faster.

## Proposed Solution

A lightweight, CPU-friendly video analytics system that processes CCTV/sample footage and flags three specific high-risk events:

1. **Weapon Detection** — detects visible knives/guns in frame using object detection.
2. **Fight/Assault Detection** — detects rapid, erratic close-proximity body movement between individuals using pose estimation.
3. **Abandoned Object Detection** — detects objects left stationary and unattended for a defined time threshold.

Each detection triggers a logged alert with a timestamped snapshot, shown on a simple monitoring dashboard.

## Technology Stack

| Component | Technology |
|---|---|
| Weapon detection | YOLOv8n (Ultralytics) |
| Pose / fight detection | MediaPipe Pose |
| Abandoned object detection | OpenCV (background subtraction) |
| Backend | Python, Flask / FastAPI |
| Dashboard | Streamlit (or Flask + HTML) |
| Data | Sample CCTV-style video clips (public datasets) |

## System Architecture

```
[Video Input (sample clips)]
        |
        v
[Frame Sampling Layer] -- (1 frame every ~0.5-1s for CPU efficiency)
        |
        v
   ----------------------------------------------------
   |                  |                                |
   v                  v                                v
[Weapon Module]  [Fight Module]                [Abandoned Object Module]
(YOLOv8n)        (MediaPipe Pose)               (OpenCV bg subtraction)
   |                  |                                |
   ----------------------------------------------------
        |
        v
[Alert Aggregator] -- (timestamp, type, confidence, snapshot)
        |
        v
[Dashboard] -- (live alert feed + flagged frame gallery)
```

## Implementation Details

See `/src/modules/` for individual detection module code and `/src/main.py` for the orchestration pipeline that ties them together.

## Key Features

- Runs on CPU only — no GPU dependency, deployable on standard police-station hardware
- Modular design — each detection type is an independent module, easy to extend with more event types later
- Real-time-style alerting with visual evidence (snapshot + timestamp) for faster human verification
- Simple dashboard built for non-technical operators

## Impact

- Reduces reliance on constant manual CCTV monitoring across multiple feeds
- Faster detection-to-response time for weapon/assault incidents in crowded areas
- Provides timestamped visual evidence to support rapid investigation

## Future Scope

- Expand to additional event types (theft detection, snatching)
- Integrate with live RTSP camera feeds instead of recorded clips
- Add multi-camera support with a centralized control-room dashboard
- Deploy on edge devices (Jetson Nano) for low-cost real-time field use
- Fine-tune models on India-specific CCTV footage for better accuracy

---

## Project Structure

```
crime-detection-mvp/
├── src/
│   ├── modules/
│   │   ├── weapon_detection.py
│   │   ├── fight_detection.py
│   │   └── abandoned_object.py
│   ├── dashboard/
│   │   └── app.py
│   └── main.py
├── data/
│   └── sample_clips/        # place test video clips here
├── docs/                     # PPT, architecture diagrams, notes
├── tests/
├── requirements.txt
└── README.md
```

## Setup Instructions

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd crime-detection-mvp

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add sample video clips to data/sample_clips/

# 5. Run the pipeline
python src/main.py

# 6. Run the dashboard
streamlit run src/dashboard/app.py
```

## Team

| Name | Role |
|---|---|
| [Member 1] | Development |
| [Member 2] | Development |
| [Member 3] | Documentation, PPT, Demo Video |

## License

This project was built for Mission Y4 – Prakasam Police Hackathon 2026.
