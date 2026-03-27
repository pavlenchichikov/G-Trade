import time
import subprocess

def run_training():
    print(">>> STARTING V29 AUTO-TRAINER CYCLE <<<")
    # Run the training script and capture output
    process = subprocess.Popen(['python', 'train_all.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    metrics = {}

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            # Parse metrics: --- V29 BTC: Acc 52.27% | AUC 0.4779 ...
            if "--- V29" in line and "Acc" in line:
                try:
                    parts = line.split()
                    name = parts[2].replace(":", "")
                    acc = float(parts[4].replace("%", ""))
                    metrics[name] = acc
                except Exception: pass

    return metrics

def main():

    while True:
        results = run_training()

        # Quality Check
        bad_models = [name for name, acc in results.items() if acc < 51.0]

        if not bad_models:
            print("\n[SUCCESS] All models passed quality threshold (>51% Acc). System Sleeping.")
            break
        else:
            print(f"\n[WARNING] {len(bad_models)} models failed (Acc < 51%): {bad_models}")
            print("Restarting training loop for improvements...")
            time.sleep(5)

if __name__ == "__main__":
    main()
