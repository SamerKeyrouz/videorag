import json
import os
import cv2

VIDEO_PATH = "../data/video.mp4"
TRANSCRIPT_PATH = "../data/transcript.json"
KEYFRAME_DIR = "../data/keyframes"
CHUNKED_OUTPUT_PATH = "../data/segments.json"

FRAME_INTERVAL = 5
SEGMENT_LENGTH = 10


def load_transcript():
    with open(TRANSCRIPT_PATH, 'r') as f:
        return json.load(f)


def segment_transcript(transcript, chunk_duration=SEGMENT_LENGTH):
    segments = []
    current_chunk = []
    current_start = transcript[0]['start']
    current_end = current_start + chunk_duration

    for entry in transcript:
        if entry['start'] < current_end:
            current_chunk.append(entry['text'])
        else:
            if current_chunk:
                segments.append({
                    'start': round(current_start, 2),
                    'end': round(current_end, 2),
                    'text': " ".join(current_chunk)
                })
            current_chunk = [entry['text']]
            current_start = entry['start']
            current_end = current_start + chunk_duration


    if current_chunk:
        segments.append({
            'start': round(current_start, 2),
            'end': round(current_end, 2),
            'text': " ".join(current_chunk)
        })

    return segments


def extract_keyframes(video_path, frame_interval=FRAME_INTERVAL):
    os.makedirs(KEYFRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved = []
    for sec in range(0, int(total_frames / fps), frame_interval):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = os.path.join(KEYFRAME_DIR, f"frame_{sec}.jpg")
        cv2.imwrite(frame_path, frame)
        saved.append((sec, frame_path))

    cap.release()
    return saved


def associate_keyframes(segments, keyframes):
    for seg in segments:
        seg_mid = (seg['start'] + seg['end']) / 2
        closest = min(keyframes, key=lambda x: abs(x[0] - seg_mid))
        seg['image_path'] = closest[1]
    return segments


if __name__ == "__main__":
    print("Loading transcript...")
    transcript = load_transcript()

    print("Segmenting transcript...")
    segments = segment_transcript(transcript)

    print("Extracting keyframes...")
    keyframes = extract_keyframes(VIDEO_PATH)

    print("Associating frames with text...")
    paired_segments = associate_keyframes(segments, keyframes)

    print(f" Saving {len(paired_segments)} segments to: {CHUNKED_OUTPUT_PATH}")
    with open(CHUNKED_OUTPUT_PATH, "w") as f:
        json.dump(paired_segments, f, indent=2)

    print("Done.")
