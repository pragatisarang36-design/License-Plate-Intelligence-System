# License Plate Intelligence System

A Python computer-vision application that processes a webcam or video file to detect supported vehicle classes, locate license plates, read plate characters with EasyOCR, and associate readings with short-lived vehicle tracks. It displays annotated frames in a desktop dashboard and saves a processed video, plate-reading CSV, and periodic screenshots.

Developer: **Pragati Sarang**

## 1. Project Overview

The system is a local desktop application for processing live camera input or recorded video. It combines object detection, simple vehicle tracking, plate localization, and OCR, then presents and records the resulting annotations. It is intended as a computer-vision demonstration and does not perform identity verification or registration lookups.

## 2. Problem Statement and Objectives

Reviewing traffic footage manually to find vehicles and transcribe visible plate characters can be time-consuming. This project demonstrates an automated pipeline for assisting with that review.

Its objectives are to:

- Detect cars, motorcycles, buses, and trucks in video frames.
- Assign temporary IDs to detected vehicles as they move through a scene.
- Detect plate regions and run OCR on plate crops associated with vehicles.
- Show detections, readings, and processing statistics while a source is processed.
- Save recognized readings and annotated output for later inspection.

The application does not validate a plate against a registration database or establish a vehicle owner's identity.

## 3. Technologies and Libraries Used

- **Python** for application logic and command-line arguments.
- **Ultralytics YOLOv8** for vehicle detection (`yolov8n.pt`) and license-plate detection (`license_plate_detector.pt`).
- **OpenCV** for video capture, image processing, annotation, and video/image writing.
- **EasyOCR** for English alphanumeric plate text recognition.
- **PyTorch** to select CUDA when available and run the models on the selected device.
- **NumPy** for image arrays and preprocessing.
- **Tkinter** and **Pillow** for the desktop live-view window.
- **tqdm** and **Colorama** for terminal progress and colored status output.
- Python standard-library modules for CSV, timestamps, threading, command-line parsing, and downloading model weights.
- **Pandas** is installed by the script's dependency bootstrap, but is not used by the current processing implementation.

## 4. System Architecture and Working Methodology

The processing flow is:

1. Parse the input source and processing options.
2. Load YOLOv8 vehicle weights and license-plate weights; select CUDA when PyTorch reports it available, otherwise use CPU.
3. Read frames from the camera or video file through OpenCV.
4. Run vehicle detection on a configurable frame interval and update the IoU-based tracker.
5. When an active track is eligible for OCR, run the plate detector on the full frame, associate a plate by its center point, and submit the expanded plate crop to EasyOCR.
6. Draw vehicle and plate annotations plus a statistics panel, then display and save the processed frames.
7. When processing ends, write the collected OCR records to `output_anpr/plates.csv`.

Video processing runs on a worker thread while Tkinter refreshes the display. Pressing Space pauses or resumes frame processing; Q exits, and S writes a snapshot of the latest frame.

## 5. Vehicle Detection and Tracking

Vehicle detection uses the pretrained Ultralytics `yolov8n.pt` model and filters detections to the COCO class IDs for car (2), motorcycle (3), bus (5), and truck (7), with a confidence threshold of 0.35. By default, vehicle detection runs every two frames; the most recent detections are reused on intervening frames. The optional `--half-res` flag runs vehicle detection at half the source dimensions.

`VehicleTracker` matches each new detection to an unmatched existing track when bounding-box intersection-over-union (IoU) exceeds 0.3. Otherwise it creates a new sequential ID. A track is removed after it has not been matched for more than 30 frames. This is a lightweight bounding-box association method, not a persistent identity system; IDs can change when a vehicle leaves view or matching fails.

## 6. License Plate Detection

A separate YOLO model detects plate boxes on the full frame, using a default confidence threshold of 0.25. A plate is associated with a vehicle when the center of the plate box falls inside that vehicle's current bounding box. The code selects the first matching plate returned for that track.

The plate detector runs only when at least one active track is eligible for OCR. A maximum OCR-call budget per frame controls how many tracks can be processed. The detected plate region is padded before OCR, while the displayed plate box uses the detector's original coordinates.

## 7. Automatic Number Plate Recognition (ANPR) and EasyOCR

EasyOCR is initialized for English (`en`) and uses CUDA when available. Before recognition, the plate crop is resized to 80 pixels tall, converted to grayscale, bilateral-filtered, and adaptively thresholded. EasyOCR is restricted to uppercase letters A–Z and digits 0–9. Returned text is uppercased and stripped of other characters; OCR fragments with confidence at or below 0.25 are discarded, and remaining fragments are concatenated.

The application accepts a reading for recording only when the cleaned result has at least four characters. Recognition quality depends on plate visibility, image resolution, motion, lighting, camera angle, and the OCR/model performance; there is no jurisdiction-specific plate-format validation.

## 8. Plate Text Confidence and Confirmation Logic

Each OCR result has a confidence calculated as the mean confidence of the accepted EasyOCR fragments. The result is recorded when it contains at least four characters. For each active track, the displayed confirmed text is replaced only when a new accepted reading has a higher confidence than the track's currently stored confidence. The overlay labels this state `LOCKED`; an available reading before that state is labeled `READING`.

This is a highest-confidence comparison, not a multi-frame consensus or a guarantee that the text is correct. Each accepted OCR attempt is appended to the CSV records, including repeated readings of the same plate. Once a track has a confirmed plate, OCR is normally retried every 45 frames; unconfirmed tracks can be retried subject to the eight-frame minimum interval and per-frame OCR budget.

## 9. Vehicle Identification and Tracking

Vehicle IDs are sequential IDs maintained by the in-process IoU tracker. Plate readings are linked to the track ID active at recognition time. The ID only describes that temporary track in one run: there is no cross-video identity matching, database lookup, face/person recognition, or long-term vehicle registry. The dashboard's vehicle count is the greatest ID allocated during the run, not the number of vehicles currently visible.

## 10. Live Dashboard and Monitoring

The Tkinter window displays the annotated video and supports pause/resume, snapshot, and quit controls. Frame annotations include vehicle class, temporary ID, vehicle detection confidence, plate boxes, and a plate panel with text and OCR confidence. A dashboard overlay reports processing FPS, frame number, current time, total allocated vehicle IDs, OCR records, unique recognized text strings, and the most frequently recorded text string.

The command-line process also shows a `tqdm` progress indicator when frame count is available and prints run status and output locations. FPS is the measured processing rate, not a guarantee about the source or display frame rate.

## 11. CSV Export and Recorded Plate Information

At the end of processing, the program writes `output_anpr/plates.csv` with these columns:

| Column | Meaning |
| --- | --- |
| `plate_text` | Cleaned OCR text accepted by the four-character minimum |
| `confidence` | Mean confidence of accepted OCR fragments, written to three decimal places |
| `vehicle_id` | Temporary tracker ID at the time of the OCR reading |
| `timestamp` | Local system date/time in ISO format when the reading was made |
| `frame_number` | One-based frame number where OCR was performed |

The CSV is rewritten at the end of each completed processing run. It contains OCR readings rather than a deduplicated registry; repeated text and repeated readings may appear. The dashboard's unique count is the number of distinct text strings in the current run.

Additional output:

- `output_anpr/output.mp4` — annotated processed video, encoded with `mp4v` at the input-reported FPS and dimensions.
- `output_anpr/screenshots/` — annotated JPEG frames saved at approximately five-second intervals based on the input-reported FPS.
- `output_anpr/snap_<timestamp>.jpg` — a manual snapshot saved when S is pressed in the GUI.

## 12. Installation and Dependencies

Use Python 3.10 or newer on Windows. Create and activate a virtual environment, then install the runtime libraries:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python easyocr numpy tqdm colorama pandas Pillow
```

Tkinter is included with most standard Windows Python installations. PyTorch is required by Ultralytics and EasyOCR and is generally installed as a dependency; install a PyTorch build appropriate for the machine if dependency installation does not provide a usable CPU or CUDA build.

The script also attempts to install the listed packages (`ultralytics`, `opencv-python`, `easyocr`, `numpy`, `tqdm`, `colorama`, `pandas`, and `Pillow`) every time it starts, suppressing installation errors. Installing them in advance in the active environment makes failures easier to diagnose. An internet connection may be needed for package and model downloads.

## 13. Windows Setup and Execution

Run commands from the repository directory containing `License Plate Intelligence System.py`:

```powershell
python ".\License Plate Intelligence System.py"
```

With no source argument, the program opens camera index 0. To process a video file:

```powershell
python ".\License Plate Intelligence System.py" "C:\path\to\video.mp4"
```

Available options:

```text
--every N          Run vehicle detection every N frames (default: 2)
--half-res         Run vehicle detection on half-size frames
--ocr-recheck N    Recheck confirmed plates every N frames (default: 45)
--max-ocr N        Maximum OCR calls per frame (default: 3)
--plate-conf FLOAT Plate detector confidence threshold (default: 0.25)
--width N          GUI display width (default: 1280)
--height N         GUI display height (default: 720)
```

For example:

```powershell
python ".\License Plate Intelligence System.py" "C:\path\to\traffic.mp4" --every 2 --max-ocr 3 --plate-conf 0.25
```

The GUI keys are Q to quit, S to save a snapshot, and Space to pause or resume. Output is written to `output_anpr` relative to the current working directory.

## 14. Model Download and Setup

The vehicle model is loaded by Ultralytics from the name `yolov8n.pt`. Ultralytics downloads the pretrained weights when they are not already available in its model cache.

The script checks for `license_plate_detector.pt` beside the Python script and accepts it if its size exceeds 1 MB. Otherwise, it attempts to download that file from the URL configured in the source. If automatic download fails, download the weights from the configured [license-plate model source](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) and place the file beside `License Plate Intelligence System.py` with the exact name `license_plate_detector.pt`.

The plate model is a required external weight file; it is not included in the source code. EasyOCR may also download its recognition model files the first time it initializes.

## 15. Limitations

- Results depend on the pretrained models and video quality; small, distant, obscured, angled, blurred, or poorly lit plates may be missed or misread.
- Only car, motorcycle, bus, and truck classes are selected by vehicle detection.
- Vehicle IDs use simple IoU matching and can change or be assigned incorrectly during occlusion, fast movement, close interactions, or missed detections.
- Plate-to-vehicle association checks only whether a plate center lies inside a vehicle box; it does not resolve ambiguous matches robustly.
- OCR is limited to English letters and digits, strips punctuation, requires at least four characters for recording, and does not check jurisdiction-specific formats.
- The highest-confidence reading is displayed as confirmed without repeated-reading consensus or external validation.
- The processing loop is sequential, with vehicle detection skipped on intervening frames; high-resolution or busy videos may process slowly, particularly on CPU.
- The output writer assumes the reported source dimensions and FPS are usable by OpenCV. Camera/backend differences can affect output creation.
- There is no database, alerting, web service, access control, or persistent tracking between runs.

## 16. Disclaimer

This project is provided for educational and portfolio purposes. It demonstrates computer-vision techniques and should not be treated as an authoritative identification or enforcement system. Plate text can be inaccurate, and human review is necessary before relying on a result. Users are responsible for complying with applicable privacy, data-protection, surveillance, and recording laws when capturing or processing video.

---

Built by **Pragati Sarang**.
