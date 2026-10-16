# CPSC 4420 Assignment 2: Q-learning on FrozenLake

Rohit Vummadi

Repository: https://github.com/RohitVummadi/CPSC-4420-Assignment2

This project implements tabular Q-learning for slippery FrozenLake on an 8 by 8 map. The notebook covers Parts A through H: the environment, Q-learning, baseline training, three parameter experiments, greedy evaluation, and added action noise.

## Files

- `assignment_2.ipynb`: implementation, comments, checks, saved outputs, plots, and observations.
- `results/plots/`: exported figures from the saved notebook.
- `results/*.csv`: numeric summaries from those saved outputs.
- `results/baseline_q.npy` and `results/selected_q.npy`: Part G's trained Q-tables, recreated with seed 42 and checked against the saved evaluation results.
- `results/trained_agent.mp4`: five consecutive fixed-seed episodes of the trained baseline policy.
- `results/video_episodes.json`: the seeds, outcomes, and lengths of the recorded episodes.
- `record_video.py`: recreates the video from the saved baseline Q-table.
- `output/pdf/assignment_2_report.pdf`: the submission report.

## Setup

Tested with Python 3.14 on macOS. Create an environment and install the pinned dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run the notebook

```sh
jupyter lab assignment_2.ipynb
```

Choose Restart Kernel and Run All Cells to run the whole assignment. The experiments each use 20,000 training episodes, so the complete notebook takes several minutes. Assertions check the episode counts, Q-values, evaluation bookkeeping, and action-noise behavior.

Parts C through F contain unseeded training runs. Their saved results describe the runs already shown in the notebook. Running them again changes the numbers and plots; update their written observations and regenerate exported figures and tables before reporting a new run. Parts G and H reset the training seed to 42 and evaluate on seeds 10000 through 10099. Their results can still vary with software versions.

## Settings and results

The baseline uses alpha 0.1, gamma 0.99, initial epsilon 1.0, epsilon decay 0.9995, minimum epsilon 0.05, and 20,000 episodes, with a 100-step episode limit. Training includes epsilon-greedy exploration. Evaluation uses greedy actions, breaks ties with the first maximum, and never changes the Q-table.

The parameter experiments compare alpha 0.1/0.5/0.9, gamma 0.50/0.90/0.99, and epsilon decay 0.99/0.9995/0.9999. The selected configuration combines alpha 0.1, gamma 0.99, and decay 0.99 based on the training comparisons. In Part G's fresh seeded comparison, the baseline achieved mean reward 0.370 and 37% success over 100 episodes; the selected agent achieved 0.230 and 23%. The baseline performed better in this evaluation, so the video uses that policy.

Part H adds action-replacement probabilities of 0, 0.10, and 0.25 while keeping slippery movement enabled. Policies trained and evaluated at the matching noise levels achieved 37%, 21%, and 8% success. The unchanged policy trained at zero added noise achieved 37%, 14%, and 7% under those same evaluation conditions. These are results from one training seed, not averages across multiple training runs.

## Recreate the video

Install FFmpeg separately and ensure `ffmpeg` is on your PATH. On macOS with Homebrew:

```sh
brew install ffmpeg
python record_video.py
```

The script records seeds 10000 through 10004 in order, with no retries or selection of successful episodes. It uses greedy actions, slippery movement, and zero added action noise. The frames are actual Gymnasium renders. The title and outcome labels are added to the video. The script overwrites the existing MP4 and episode log.

The saved models match the saved Part G results. Re-running the notebook does not automatically replace the saved models, CSV summaries, exported figures, or report.

## Submission

Public repository: https://github.com/RohitVummadi/CPSC-4420-Assignment2. The report includes this URL. Upload only `output/pdf/assignment_2_report.pdf` to Canvas. The repository contains the notebook, this README, requirements, results and plots, trained models, video, and video script.
