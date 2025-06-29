# Traffic Monitoring System

A computer vision system that counts vehicles in different regions from drone video footage using YOLO object detection and tracking.

## Demo Videos

### Input Video
📹 **Original Drone Footage**: [Download Input Video](https://drive.google.com/file/d/1ShkuRXyxfJYIA-2UNSazPAygDz0jRqgT/view?usp=drive_link)

### Output Video
🎯 **Processed Result**: [Download Output Video](https://drive.google.com/file/d/1_J8vKGNZbm2SQm0mutIfKx6lrWOd2iKE/view?usp=drive_link)

## Features

- **Multi-Region Vehicle Tracking**: Define multiple polygonal regions and track vehicles entering/exiting each region
- **Real-time Vehicle Counting**: Count vehicles entering and leaving each defined region
- **Visual Feedback**: Color-coded regions with transparency overlay and real-time count display
- **Video Output**: Save processed video with annotations, bounding boxes, and tracking information
- **Persistent Tracking**: Maintains vehicle IDs across frames for accurate counting

## Requirements

```bash
pip install ultralytics opencv-python numpy
```

## Project Structure

```
traffic_monitoring/
├── tmp.py                          # Main tracking implementation
├── input_vid.mov                   # Input drone video
├── traffic_monitoring_output.mp4   # Output processed video
├── runs/detect/train2/weights/     # Trained YOLO model weights
├── dataset/                        # Training dataset (ignored in git)
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## Usage

### Basic Usage

```python
from tmp import CarRegionTracking

# Define regions of interest (polygon coordinates)
regions = {
    "region-01": [[8, 594], [716, 594], [716, 984], [8, 984]],
    "region-02": [[8, 1115], [682, 1132], [670, 1565], [4, 1544]],
    # ... more regions
}

# Initialize tracker
tracker = CarRegionTracking(
    model_path="runs/detect/train2/weights/best.pt",
    regions=regions,
    vid_src="input_vid.mov",
    save_path="output_video.mp4"
)

# Process video
tracker.process()
```

### Running the Script

```bash
python tmp.py
```

## Configuration

### Region Definition

Regions are defined as lists of coordinate points forming polygons:

```python
regions = {
    "region-name": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    # ... more regions
}
```

### Model Path

Update the model path to point to your trained YOLO weights:

```python
model_path = "runs/detect/train2/weights/best.pt"
```

## Key Components

### CarRegionTracking Class

Main class that handles:
- **Vehicle Detection**: Uses YOLO for object detection
- **Multi-Object Tracking**: Maintains consistent IDs across frames
- **Region Management**: Tracks which vehicles are in which regions
- **Counting Logic**: Counts entries and exits for each region
- **Visualization**: Draws regions, bounding boxes, and statistics

### Key Methods

- `isInRegion(point, region)`: Point-in-polygon detection using ray casting algorithm
- `drawRegions(frame)`: Draws colored regions with transparency
- `drawBoundingBox(frame, box, track_id)`: Draws vehicle bounding boxes and IDs
- `compareRegions()`: Compares current vs previous frame to count entries/exits
- `process()`: Main processing loop for video analysis

## Output

The system generates:

1. **Processed Video**: Shows original video with overlays including:
   - Color-coded regions (9 different colors)
   - Semi-transparent region fills
   - Vehicle bounding boxes
   - Track IDs for each vehicle
   - Real-time entry/exit counts per region

2. **Console Output**: Real-time statistics and processing information

## Region Colors

The system uses 9 predefined colors for regions:
- Green, Blue, Red, Cyan, Magenta, Yellow, Purple, Orange, Light Blue

## Algorithm Details

### Point-in-Polygon Detection

Uses ray casting algorithm to determine if vehicle center point is inside a polygonal region.

### Vehicle Tracking

- Utilizes YOLO's built-in tracking capabilities with `persist=True`
- Maintains consistent track IDs across frames
- Tracks vehicle center points for region detection

### Counting Logic

- Compares vehicle presence in regions between consecutive frames
- Increments "In" count when vehicle appears in region
- Increments "Out" count when vehicle disappears from region

## Performance Considerations

- Processing speed depends on video resolution and model complexity
- GPU acceleration recommended for real-time processing
- Adjust region complexity based on accuracy requirements

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure YOLO model path is correct
2. **Video not opening**: Check input video path and format
3. **Poor tracking**: Adjust confidence thresholds or use better trained model
4. **Inaccurate regions**: Verify region coordinate definitions

### Tips

- Use high-resolution videos for better detection accuracy
- Ensure good lighting conditions in source video
- Fine-tune region boundaries for optimal counting
- Monitor console output for debugging information

## License

This project is for educational purposes. Please ensure compliance with relevant licenses for YOLO and other dependencies.

## Contributing

Feel free to submit issues and enhancement requests!
