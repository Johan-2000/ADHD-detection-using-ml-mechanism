import cv2
import easyocr
import pandas as pd

VIDEO_PATH = 'Disruptive Dan.avi'
CSV_OUTPUT_PATH = 'behavior_classification.csv'
FPS_SAMPLE_RATE = 1.0  

VALID_LABELS = ['focused', 'Disruptive', 'Turning', 'Disfocused']

def get_behavior_class(label):
    """Maps an extracted label to a behavior class."""
    if label == 'focused':
        return 'Non-ADHD'
    elif label in ['Disruptive', 'Turning', 'Disfocused']:
        return 'ADHD'
    else:
        return 'Unknown'

def parse_ocr_text(text):
    """Cleanly extracts the label and score from the OCR output."""
    for label in VALID_LABELS:
        if label in text:
            parts = text.split(label)
            if len(parts) > 1:
                try:
                    score_part = parts[1].strip().split()[0].replace(',', '.')
                    score = float(score_part)
                    return label, score
                except (ValueError, IndexError):
                    continue  
    return None, None

def extract_behavior_data(video_path, csv_output_path, fps_sample_rate):
    """Processes video, classifies behavior, and saves to CSV."""
    
    print("Loading EasyOCR model... (This may take a moment on first run)")
    reader = easyocr.Reader(['en']) 
    print("Model loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps * fps_sample_rate))
    all_data = []
    frame_count = 0

    print(f"Video FPS: {fps:.2f}. Sampling every {fps_sample_rate} second(s).")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp_seconds = frame_count / fps
            timestamp_formatted = f"{int(timestamp_seconds // 60):02d}:{int(timestamp_seconds % 60):02d}"
            
            results = reader.readtext(frame, detail=1, paragraph=False)

            for (bbox, text, prob) in results:
                label, score = parse_ocr_text(text)
                
                if label and score is not None:
                   
                    behavior_class = get_behavior_class(label)
                    
                    all_data.append({
                        'Timestamp': timestamp_formatted,
                        'Second': round(timestamp_seconds, 2),
                        'Label': label,
                        'Confidence_Score': score,
                        'Behavior': behavior_class  
                    })

        frame_count += 1
        
    cap.release()
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(csv_output_path, index=False)
        print(f"\n Successfully generated CSV file at: **{csv_output_path}**")
        print(f"Total entries: {len(df)}")
    else:
        print("\n No relevant labels were successfully extracted from the video.")


extract_behavior_data(VIDEO_PATH, CSV_OUTPUT_PATH, FPS_SAMPLE_RATE)