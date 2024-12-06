# Real-Time Traffic Monitoring with ROI Selection

A Python application for real-time traffic monitoring using customizable Regions of Interest (ROI) and YOLOv8 object detection. This tool allows users to select specific areas in video streams for focused traffic analysis.

## Features

- **Dynamic ROI Selection**: Click and drag to define monitoring areas
- **Real-time Vehicle Detection**: Using YOLOv8 for accurate vehicle detection
- **Multi-threaded Processing**: Optimized performance with background processing
- **Vehicle Counting**: Automatic counting of vehicles passing through ROI
- **Screenshot Capability**: Save monitoring snapshots with timestamp
- **Live Statistics**: Real-time display of FPS, vehicle count, and ROI information
- **YouTube Stream Support**: Monitor live traffic cameras from YouTube

## Requirements

- Python 3.7 or higher
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- yt-dlp

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   # or
   .\env\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. Controls:
   - Click and drag to select ROI
   - Press 'R' to reset ROI
   - Press 'S' to take a screenshot
   - Press 'Q' to quit

## Project Structure

```
├── main.py              # Main application file
├── requirements.txt     # Project dependencies
├── screenshots/         # Directory for saved screenshots
└── README.md           # Project documentation
```

## Features in Detail

### ROI Selection
- Click and drag with the mouse to define a Region of Interest
- Real-time visual feedback during selection
- Ability to reset and redefine ROI at any time

### Vehicle Detection
- Detects cars, trucks, buses, and motorcycles
- Real-time tracking within the selected ROI
- Automatic vehicle counting

### Information Display
- FPS counter
- Vehicle count
- ROI dimensions and position
- Active/Paused status

### Screenshot Function
- Save current view with all overlays
- Automatic timestamp in filename
- Saved in 'screenshots' directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team