"""Record five fixed-seed episodes of the trained baseline policy."""

import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont


root = Path(__file__).resolve().parent
encoder = shutil.which("ffmpeg")
if encoder is None:
    raise RuntimeError("Install FFmpeg and make sure ffmpeg is on your PATH.")
Q = np.load(root / "results/baseline_q.npy", allow_pickle=False)
assert Q.shape == (64, 4) and np.isfinite(Q).all()
output = root / "results/trained_agent.mp4"
font = ImageFont.load_default(size=16)
episode_results = []

with gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True,
              success_rate=1.0 / 3.0, reward_schedule=(1, 0, 0),
              render_mode="rgb_array") as env:
    env.reset(seed=10000)
    height, width, _ = env.render().shape
    process = subprocess.Popen([
        encoder, "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height + 88}",
        "-r", "10", "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output)
    ], stdin=subprocess.PIPE)

    def write_frame(episode, seed, steps, outcome, repeats=1):
        image = Image.new("RGB", (width, height + 88), "#f4f6f9")
        image.paste(Image.fromarray(env.render()), (0, 56))
        draw = ImageDraw.Draw(image)
        draw.text((12, 7), "FrozenLake 8x8 | trained baseline", font=font, fill="#172e44")
        draw.text((12, 30), f"Episode {episode}/5 | seed {seed} | epsilon = 0", font=font, fill="#172e44")
        draw.text((12, height + 64), f"Steps: {steps} | {outcome}", font=font, fill="#172e44")
        for _ in range(repeats):
            process.stdin.write(image.tobytes())

    try:
        for episode, seed in enumerate(range(10000, 10005), start=1):
            state, info = env.reset(seed=seed)
            terminated = truncated = False
            steps = 0
            total_reward = 0
            write_frame(episode, seed, steps, "Running", repeats=10)
            while not (terminated or truncated):
                action = int(np.argmax(Q[state]))
                state, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total_reward += reward
                outcome = ("Goal reached" if total_reward > 0 else
                           "Step limit reached" if truncated else
                           "Fell into a hole" if terminated else "Running")
                write_frame(episode, seed, steps, outcome)
            write_frame(episode, seed, steps, outcome, repeats=15)
            episode_results.append({"seed": seed, "steps": steps,
                                    "reward": total_reward, "outcome": outcome})
    finally:
        process.stdin.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {return_code}.")

assert len(episode_results) == 5
assert output.stat().st_size > 0
(root / "results/video_episodes.json").write_text(json.dumps(episode_results, indent=2) + "\n")
print(f"Saved {output}")
print(json.dumps(episode_results, indent=2))
