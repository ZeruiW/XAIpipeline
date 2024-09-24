import os
import av
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import json

def process_video(video_path, output_dir, extractor):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    fps = video_stream.average_rate
    total_frames = video_stream.frames
    
    frames = []
    attention_data = []
    frame_count = 0
    all_logits = []
    attention_buffer = []
    temporal_smoothing_window = 3  # 减小时间平滑窗口

    for frame in tqdm(container.decode(video=0), desc="Processing frames", total=total_frames):
        frame_rgb = frame.to_rgb().to_ndarray()
        frames.append(frame_rgb)
        
        if len(frames) == 8:
            spatial_attention, logits = extractor.extract_attention(frames)
            all_logits.append(logits)
            
            for i in range(8):
                attention = spatial_attention[0, i+1]
                attention_buffer.append(attention)
                
                if len(attention_buffer) >= temporal_smoothing_window:
                    smoothed_attention = np.mean(attention_buffer[-temporal_smoothing_window:], axis=0)
                    heatmap_frame = extractor.apply_attention_heatmap(frames[i], smoothed_attention)
                    
                    frame_filename = f"frame_{frame_count+1}_spatial_attention.png"
                    cv2.imwrite(os.path.join(frames_dir, frame_filename), cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
                    
                    attention_data.append({
                        "frame_index": frame_count,
                        "max_attention": float(smoothed_attention[1:].max()),
                        "min_attention": float(smoothed_attention[1:].min()),
                        "mean_attention": float(smoothed_attention[1:].mean())
                    })
                
                frame_count += 1
            
            frames = frames[7:]

    # Process remaining frames
    if frames:
        padding = [frames[-1]] * (8 - len(frames))
        spatial_attention, logits = extractor.extract_attention(frames + padding)
        all_logits.append(logits)
        
        for i in range(len(frames)):
            attention = spatial_attention[0, i+1]
            attention_buffer.append(attention)
            
            smoothed_attention = np.mean(attention_buffer[-temporal_smoothing_window:], axis=0)
            heatmap_frame = extractor.apply_attention_heatmap(frames[i], smoothed_attention)
            
            frame_filename = f"frame_{frame_count+1}_spatial_attention.png"
            cv2.imwrite(os.path.join(frames_dir, frame_filename), cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
            
            attention_data.append({
                "frame_index": frame_count,
                "max_attention": float(smoothed_attention[1:].max()),
                "min_attention": float(smoothed_attention[1:].min()),
                "mean_attention": float(smoothed_attention[1:].mean())
            })
            
            frame_count += 1

    # Save attention data
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(attention_data, f)

    # Calculate overall logits and predicted label
    overall_logits = np.mean(all_logits, axis=0)
    predicted_label = int(np.argmax(overall_logits))

    return predicted_label

def visualize_saliency(video_name, num_segments=8, results_dir=''):
    try:
        # Load data
        with open(os.path.join(results_dir, "results.json"), 'r') as f:
            attention_data = json.load(f)
        
        # Extract temporal attention
        temporal_attention = np.array([frame['mean_attention'] for frame in attention_data])
        
        # Normalize temporal attention
        temporal_attention = (temporal_attention - temporal_attention.min()) / (temporal_attention.max() - temporal_attention.min())
        
        # Select frames to visualize
        total_frames = len(attention_data)
        frame_indices = np.linspace(0, total_frames-1, num_segments, dtype=int)
        
        # Create figure with adjusted dimensions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 0.7]})
        fig.suptitle(f"Temporal Saliency and Key Frames", fontsize=12)
        
        # Plot temporal saliency
        ax1.plot(range(total_frames), temporal_attention, color='blue', alpha=0.5)
        ax1.scatter(frame_indices, temporal_attention[frame_indices], color='red', s=30, zorder=5)
        for idx in frame_indices:
            ax1.axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel("Frame Number", fontsize=10)
        ax1.set_ylabel("Temporal Saliency", fontsize=10)
        ax1.set_xlim(0, total_frames-1)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        
        # Display key frames
        for i, frame_idx in enumerate(frame_indices):
            # Read the frame
            frame_path = os.path.join(results_dir, 'frames', f"frame_{frame_idx+1}_spatial_attention.png")
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Frame not found: {frame_path}")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add frame to the plot
            ax_sub = ax2.inset_axes([i/num_segments, 0.1, 1/num_segments - 0.01, 0.8], transform=ax2.transAxes)
            ax_sub.imshow(frame)
            ax_sub.axis('off')
            ax_sub.set_title(f"Frame {frame_idx+1}", fontsize=8)
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # Adjust space between subplots
        output_path = os.path.join(results_dir, f"{video_name}_temporal_saliency_visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal saliency visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error in visualize_saliency for video {video_name}: {str(e)}")
        raise

def create_heatmap_video(output_dir, video_name):
    frames_dir = os.path.join(output_dir, 'frames')
    output_video_path = os.path.join(output_dir, f"{video_name}_heatmap.mp4")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('_spatial_attention.png')])
    if not frame_files:
        raise ValueError("No frames found to create video")
    
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        out.write(frame)
    
    out.release()
    return output_video_path
