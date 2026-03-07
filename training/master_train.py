import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n>>> Starting: {description}")
    print(f"Executing: {command}")
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1
        )
        
        # Stream the output real-time
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        if process.returncode == 0:
            print(f">>> [SUCCESS] {description}")
        else:
            print(f">>> [FAILURE] {description} exited with code {process.returncode}")
            sys.exit(process.returncode)
            
    except Exception as e:
        print(f">>> [ERROR] {description} failed: {e}")
        sys.exit(1)

def main():
    print("\n" + "!"*50)
    print("WARNING: Running VAE and LDM training in parallel.")
    print("This may cause Out-Of-Memory (OOM) errors on a GTX 1650.")
    print("!"*50 + "\n")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    # Start Image VAE Training
    vae_process = subprocess.Popen(
        "python training/train_image_vae.py",
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    print(">>> Started Image VAE training in background.", flush=True)

    # Start Latent Diffusion Training
    ldm_process = subprocess.Popen(
        "python training/train_latent_diffusion.py",
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    print(">>> Started Latent Diffusion training in background.", flush=True)

    # Function to stream output from a process
    def stream_output(process, name):
        import sys
        for line in process.stdout:
            print(f"[{name}] {line}", end="", flush=True)

    import threading
    t1 = threading.Thread(target=stream_output, args=(vae_process, "VAE"))
    t2 = threading.Thread(target=stream_output, args=(ldm_process, "LDM"))
    
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()

    try:
        vae_process.wait()
        ldm_process.wait()
    except KeyboardInterrupt:
        print("\n>>> Interrupting parallel training...", flush=True)
        vae_process.terminate()
        ldm_process.terminate()

    print("\n" + "="*50)
    print("✅ MASTER TRAINING (PARALLEL) COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
