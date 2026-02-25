import subprocess
import os
import sys
import Bar_Tracking.barspeed_demo as BST_demo
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
    # Setup demo output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_output_dir = os.path.join(script_dir, "..", "Train_Outputs", "Demo")
    demo_output_dir = os.path.abspath(demo_output_dir)
    demo_videos_dir = os.path.join(script_dir, "..", "lifting_videos", "Demo_Videos")
    demo_videos_dir = os.path.abspath(demo_videos_dir)
    os.makedirs(demo_output_dir, exist_ok=True)
    print(f"Demo outputs will be saved to: {demo_output_dir}\n")
    
    print("DEMO PIPELINE: RUNNING MMPOSE EXTRACTION")

    # Get the directory where demo.py is located (src/)
    mmpose_dir = os.path.join(script_dir, "MMPose")
    
    # Run demo_extraction.sh
    demo_script = os.path.join(mmpose_dir, "demo_extraction.sh")
    script_name = "demo_extraction.sh"
    
    print(f"--- Processing {script_name} ---")
    
    # Auto-clean the file line endings
    fix_line_endings(demo_script)
    
    # Run the script from MMPose directory (so relative paths work)
    try:
        subprocess.run(["bash", script_name], cwd=mmpose_dir, check=True)
        print(f"Finished {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with exit code {e.returncode}")
        sys.exit(1)
    
    D.main()
    print("DEMO PIPELINE: MMPOSE EXTRACTION COMPLETE")

    print("DEMO PIPELINE: RUNNING OPEN FACE EXTRACTION")
    openface_csv = os.path.join(demo_output_dir, "openface_features_all.csv")
    OF.run(create_visualizations=True, output_dir=demo_output_dir, output_csv=openface_csv)
    print("DEMO PIPELINE: OPEN FACE EXTRACTION COMPLETE")

    print("DEMO PIPELINE: RUNNING BAR SPEED TRACKING")
    BST_demo.run_demo(demo_folder=demo_videos_dir, output_dir=demo_output_dir)
    print("DEMO PIPELINE: BAR SPEED TRACKING COMPLETE")

    print("DEMO PIPELINE COMPLETE")