#!/usr/bin/env python3
"""
Story Teller Web Interface for Soprano TTS + Gemini (Pipeline Streaming)
"""

import argparse
import socket
import time
import queue
import threading
import re
import gradio as gr
import numpy as np
import google.generativeai as genai
from soprano import SopranoTTS
from soprano.utils.streaming import play_stream

# Parse arguments for Soprano
parser = argparse.ArgumentParser(description='Soprano Story Teller')
parser.add_argument('--model-path', '-m', help='Path to local model directory (optional)')
parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cuda', 'cpu', 'mps'], help='Device to use for inference')
parser.add_argument('--backend', '-b', default='auto', choices=['auto', 'transformers', 'lmdeploy'], help='Backend to use for inference')
parser.add_argument('--cache-size', '-c', type=int, default=100, help='Cache size in MB (for lmdeploy backend)')
parser.add_argument('--decoder-batch-size', '-bs', type=int, default=1, help='Batch size when decoding audio')
args = parser.parse_args()

# Initialize Soprano
print("Loading Soprano TTS model...")
model = SopranoTTS(
    backend=args.backend,
    device=args.device,
    cache_size_mb=args.cache_size,
    decoder_batch_size=args.decoder_batch_size,
    model_path=args.model_path
)
device = model.device
backend = model.backend
print(f"Model loaded successfully on {device}!")

SAMPLE_RATE = 32000

def generate_and_narrate(
    api_key: str,
    description: str,
    duration: str,
    vibe: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float
):
    """
    True Pipeline Streaming:
    Gemini Stream -> Text Buffer -> Sentence Splitter -> Soprano Stream -> Audio Player
    """
    if not api_key.strip():
        yield "Please provide a valid Gemini API Key.", None, "✗ Error: Missing API Key"
        return

    if not description.strip():
        yield "Please enter a story description.", None, "✗ Error: Missing Description"
        return

    yield "Thinking... (Connecting to Gemini)", None, "⏳ Starting..."

    # State
    full_text = ""
    sentence_buffer = ""
    sentence_queue = queue.Queue()
    finished_generating_text = False
    
    # --- Thread 1: Gemini Text Generation ---
    def generate_text_thread():
        nonlocal full_text, sentence_buffer, finished_generating_text
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            prompt = f"""
            Write a story based on description: "{description}".
            Target Duration: {duration}. Vibe: {vibe}.
            Write for TTS (simple, evocative, linear). No markdown.
            """
            
            response = gemini_model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    text_chunk = chunk.text
                    full_text += text_chunk
                    sentence_buffer += text_chunk
                    
                    # Split into sentences (simple heuristic)
                    sentences = re.split(r'(?<=[.!?])\s+', sentence_buffer)
                    if len(sentences) > 1:
                        # We have complete sentences, keep the last incomplete one in buffer
                        for s in sentences[:-1]:
                            if s.strip():
                                sentence_queue.put(s.strip())
                        sentence_buffer = sentences[-1]
            
            # Flush remaining buffer at end
            if sentence_buffer.strip():
                sentence_queue.put(sentence_buffer.strip())
                
        except Exception as e:
            print(f"Gemini Error: {e}")
        finally:
            finished_generating_text = True

    # Start Text Gen Thread
    t = threading.Thread(target=generate_text_thread)
    t.start()

    # --- Main Loop: Consume Sentences and Narrate ---
    
    yield "", None, "⏳ Waiting for first sentence..."
    
    start_time = time.time()
    first_audio_played = False
    
    while not finished_generating_text or not sentence_queue.empty():
        try:
            # Get next sentence (wait briefly if needed)
            sentence = sentence_queue.get(timeout=0.1)
            
            # Yield update text to UI
            yield full_text, None, f"🎙️ Narrating: \"{sentence[:20]}...\""
            
            # Synthesize & Play immediately
            stream = model.infer_stream(
                sentence,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                chunk_size=1
            )
            
            # Check latency on first chunk
            if not first_audio_played:
                latency = time.time() - start_time
                yield full_text, None, f"✓ Audio Started ({latency*1000:.0f}ms latency)..."
                first_audio_played = True

            play_stream(stream)
            
        except queue.Empty:
            continue
        except Exception as e:
            yield full_text, None, f"Error: {e}"

    yield full_text, None, "✓ Story Complete"

def find_free_port(start_port=7860, max_tries=100):
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise OSError("Could not find a free port")

# --- UI Setup ---
with gr.Blocks(title="Soprano Story Teller") as demo:
    gr.Markdown(
        f"""# 📖 Soprano Story Teller (Turbo Mode)
        **Gemini 2.0 Flash -> Pipe -> Soprano TTS**
        
        **Device:** {device.upper()} | **Backend:** {backend}
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                placeholder="Enter your Google Gemini API Key",
                type="password"
            )
            description_input = gr.Textbox(
                label="Story Description",
                placeholder="e.g., A cyberpunk detective in neon rain...",
                lines=3
            )
            with gr.Row():
                duration_input = gr.Textbox(
                    label="Duration / Length",
                    value="Short (approx 30s)"
                )
                vibe_input = gr.Dropdown(
                    label="Vibe",
                    choices=["Whimsical", "Spooky", "Melancholic", "Action", "Sci-Fi", "Cozy", "Dramatic", "Funny"],
                    value="Sci-Fi"
                )
            
            with gr.Accordion("TTS Settings", open=False):
                temperature = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Temperature")
                top_p = gr.Slider(0.5, 1.0, 0.95, step=0.05, label="Top P")
                repetition_penalty = gr.Slider(1.0, 2.0, 1.2, step=0.1, label="Repetition Penalty")
            
            generate_btn = gr.Button("🚀 Generate & Narrate", variant="primary", size="lg")

        with gr.Column(scale=1):
            story_output = gr.Textbox(label="Live Story Text", lines=10, interactive=False)
            audio_output = gr.Audio(label="Audio Output", interactive=False, visible=False) # Hidden, simple placeholder
            status_output = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("🔊 **Audio plays directly from server speakers (for ultra-low latency).**")

    generate_btn.click(
        fn=generate_and_narrate,
        inputs=[api_key_input, description_input, duration_input, vibe_input, temperature, top_p, repetition_penalty],
        outputs=[story_output, audio_output, status_output]
    )

def main():
    port = find_free_port(7860)
    print(f"Starting Story Teller on port {port}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo"), 
    )

if __name__ == "__main__":
    main()
