import subprocess
import os
import sys
import Bar_Tracking.barspeed_to_excel as BST
import OpenFace.test_openface as OF
import MasterResultsMaker as M
import LGBM_Regressor.LGBMTrain011123 as LGBM
import MMPose.dmetrics_to_excel as D


def fix_line_endings(file_path):
    """
    Reads the file and converts Windows CRLF (\r\n) line endings to Unix LF (\n).
    This ensures the script runs on Linux without 'command not found' errors.
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        # Only rewrite if we find Windows line endings (\r\n)
        if b'\r\n' in content:
            print(f"   [Auto-Fix] Converting Windows line endings for {file_path}...")
            new_content = content.replace(b'\r\n', b'\n')
            with open(file_path, 'wb') as f:
                f.write(new_content)
    except Exception as e:
        print(f"   [Warning] Could not check line endings for {file_path}: {e}")


if __name__ == "__main__":
    print("PIPELINE: RUNNING BAR SPEED TRACKING")
    #BST.run()
    print("PIPELINE: BAR SPEED TRACKING COMPLETE")

    print("PIPELINE: RUNNING OPEN FACE EXTRACTION")
    #OF.run()
    print("PIPELINE: OPEN FACE EXTRACTION COMPLETE")

    print("PIPELINE: RUNNING MMPOSE EXTRACTION")

    # Get the directory where train.py is located (src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        os.path.join(script_dir, "MMPose", "bench_extraction.sh"),
        os.path.join(script_dir, "MMPose", "deadlift_extraction.sh"),
        os.path.join(script_dir, "MMPose", "squat_extraction.sh")
    ]

    print("PIPELINE: Starting MMPose shell script execution...")

    # Change to MMPose directory so relative paths in scripts work correctly
    mmpose_dir = os.path.join(script_dir, "MMPose")
    
    for script in scripts:
        script_name = os.path.basename(script)
        print(f"--- Processing {script_name} ---")

        # 1. Auto-clean the file so they don't have to run 'sed' manually
        fix_line_endings(script)

        # 2. Run the script from MMPose directory (so relative paths work)
        try:
            subprocess.run(["bash", script_name], cwd=mmpose_dir, check=True)
            print(f"Finished {script_name}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {script_name} failed with exit code {e.returncode}")
            # Optional: sys.exit(1) if you want the whole pipeline to stop

    D.main()
    print("PIPELINE: MMPOSE EXTRACTION COMPLETE")

    print("PIPELINE: CONVERTING RESULTS TO EXCEL")
    # M.createMasterResults()
    print("PIPELINE: CONVERTING RESULTS TO EXCEL COMPLETE")

    print("PIPELINE: TRAINING LGBM REGRESSION MODEL")
    # LGBM.run()
    print("PIPELINE: LGBM REGRESSION TRAINING COMPLETE")