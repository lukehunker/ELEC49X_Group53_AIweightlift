#!/usr/bin/env python3
"""
FastAPI Server for RPE Prediction

Handles video uploads from Flutter app and returns RPE predictions.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /predict - Upload video, get RPE prediction
    GET /health - Health check
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import json
import numpy as np
from datetime import datetime, timezone
from threading import Lock

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import unified pipeline
from rpe_pipeline import RPEPredictor


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


app = FastAPI(
    title="AI Weightlift Coach API",
    description="RPE prediction from workout videos",
    version="1.0.0"
)

# CORS - Allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
PREDICTOR = None
HISTORY_LOCK = Lock()
HISTORY_FILE = Path(__file__).parent.parent / "data" / "rpe_history.json"
OUTPUT_ROOT = Path(__file__).parent.parent / "output"


def ensure_history_file_exists() -> None:
    """Create history file if missing."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_history() -> list:
    """Load RPE history records from disk."""
    ensure_history_file_exists()
    with HISTORY_LOCK:
        try:
            content = HISTORY_FILE.read_text(encoding="utf-8")
            records = json.loads(content)
            return records if isinstance(records, list) else []
        except json.JSONDecodeError:
            return []


def append_history_record(record: dict) -> None:
    """Append a single RPE record to history file."""
    ensure_history_file_exists()
    with HISTORY_LOCK:
        try:
            current = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if not isinstance(current, list):
                current = []
        except json.JSONDecodeError:
            current = []

        current.append(record)
        HISTORY_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")


def clear_history_records() -> int:
    """Clear all history records and return number removed."""
    ensure_history_file_exists()
    with HISTORY_LOCK:
        try:
            current = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            removed_count = len(current) if isinstance(current, list) else 0
        except json.JSONDecodeError:
            removed_count = 0

        HISTORY_FILE.write_text("[]", encoding="utf-8")
        return removed_count


def resolve_openface_overlay_relative_path(video_name: str) -> Optional[str]:
    """Resolve generated OpenFace overlay video path relative to output root."""
    base_name = Path(video_name).stem
    candidate = OUTPUT_ROOT / "openface" / "visualized" / f"{base_name}_pose_guided.mp4"
    if candidate.exists():
        return str(candidate.relative_to(OUTPUT_ROOT)).replace("\\", "/")
    return None

@app.on_event("startup")
async def load_models():
    """Load ensemble models (LGBM + XGBoost) and predictor on server start"""
    global PREDICTOR
    
    model_dir = Path(__file__).parent / "Train_Outputs"
    
    try:
        ensure_history_file_exists()
        PREDICTOR = RPEPredictor(model_path=model_dir, verbose=True)
        
        # Check if ensemble models are loaded
        if PREDICTOR.lgbm_model and PREDICTOR.xgb_model:
            print("RPE Predictor with Ensemble models initialized successfully!")
            print(f"   LGBM: {PREDICTOR.ensemble_weights['lgbm']:.0%} weight")
            print(f"   XGBoost: {PREDICTOR.ensemble_weights['xgb']:.0%} weight")
        else:
            print("Warning: Ensemble models not fully loaded")
            print("   Run: python src/LGBM_Regressor/LGBMMod2_Ensemble.py")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("   Server will start but predictions will fail until models are trained")


    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="output")


@app.get("/history")
async def get_history(limit: Optional[int] = 50):
    """Get recent prediction history (newest first)."""
    max_limit = 500
    safe_limit = min(max(limit or 50, 1), max_limit)

    records = load_history()
    records_sorted = sorted(
        records,
        key=lambda item: item.get("timestamp", ""),
        reverse=True
    )

    return JSONResponse(content=convert_to_json_serializable({
        "success": True,
        "count": len(records_sorted[:safe_limit]),
        "records": records_sorted[:safe_limit]
    }))


@app.delete("/history")
async def clear_history():
    """Clear all prediction history records."""
    removed_count = clear_history_records()
    return JSONResponse(content={
        "success": True,
        "cleared": True,
        "removed_count": removed_count
    })


@app.get("/")
async def root():
    """Welcome message"""
    return {
        "message": "AI Weightlift Coach API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST with video)",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = (PREDICTOR is not None and 
                     PREDICTOR.lgbm_model is not None and 
                     PREDICTOR.xgb_model is not None)
    
    response = {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "version": "1.0.0",
    }
    
    if PREDICTOR and models_loaded:
        response["ensemble_config"] = {
            "lgbm_weight": PREDICTOR.ensemble_weights.get('lgbm', 0),
            "xgb_weight": PREDICTOR.ensemble_weights.get('xgb', 0),
            "n_features": len(PREDICTOR.feature_names) if PREDICTOR.feature_names else 0
        }
        if PREDICTOR.metadata:
            cv_perf = PREDICTOR.metadata.get('cv_performance', {})
            if cv_perf:
                response["model_performance"] = {
                    "cv_mae": cv_perf.get('ensemble_mae', 0),
                    "cv_r2": cv_perf.get('ensemble_r2', 0)
                }
    
    return convert_to_json_serializable(response)


@app.post("/predict")
async def predict_rpe(
    request: Request,
    video: UploadFile = File(...),
    lift_type: str = Form(...)
):
    """
    Predict RPE from uploaded workout video
    
    Args:
        video: Video file (mp4, mov, avi)
        lift_type: Type of lift (Bench Press, Squat, Deadlift)
    
    Returns:
        {
            "predicted_rpe": 8.5,
            "confidence": 0.92,
            "lift_type": "Squat",
            "video_name": "workout.mp4",
            "features": {
                "bar_speed": {...},
                "facial": {...},
                "posture": {...}
            },
            "warnings": []
        }
    """
    
    # Check if predictor is loaded
    if PREDICTOR is None or PREDICTOR.lgbm_model is None or PREDICTOR.xgb_model is None:
        raise HTTPException(
            status_code=503,
            detail="Ensemble models not loaded. Please train the models first."
        )
    
    # Validate lift type
    valid_lifts = ['Bench Press', 'Squat', 'Deadlift']
    if lift_type not in valid_lifts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lift_type. Must be one of: {valid_lifts}"
        )
    
    # Validate video format
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Please use MP4, MOV, AVI, or MKV"
        )
    
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp(prefix="rpe_api_")
    video_path = None
    
    try:
        # Save uploaded video to temp file
        video_path = os.path.join(temp_dir, video.filename)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        print(f"\n{'='*70}")
        print(f"Processing upload: {video.filename} ({lift_type})")
        print(f"{'='*70}")
        
        # Run complete prediction pipeline with visualizations enabled
        result = PREDICTOR.predict(
            video_path,
            movement=lift_type,
            output_dir=temp_dir,
            save_visualizations=True
        )

        import urllib.parse
        openface_overlay_relative = resolve_openface_overlay_relative_path(video.filename)
        openface_overlay_url = None
        if openface_overlay_relative:
            # URL-encode the path portion (not the whole URL)
            encoded_path = urllib.parse.quote(openface_overlay_relative)
            openface_overlay_url = str(request.base_url).rstrip("/") + f"/output/{encoded_path}"
        print(f"[DEBUG] OpenFace overlay URL: {openface_overlay_url}")

        # Format response for API
        response = {
            "success": result['success'],
            "predicted_rpe": result['predicted_rpe'],
            "confidence": result['confidence'],
            "lift_type": result['movement'],
            "video_name": result['video_name'],
            "openface_overlay_url": openface_overlay_url,
            "warnings": result['warnings'],
            "features": {
                "bar_speed": result['features'].get('bar_speed'),
                "facial": {
                    k: v for k, v in (result['features'].get('facial') or {}).items()
                    if not k.startswith('metadata_')
                },
                "posture": result['features'].get('posture'),
                "detection_rate": result['features']['combined'].get('detection_rate', 0)
            }
        }

        print(f"\nPrediction complete: RPE {response['predicted_rpe']}")

        history_record = {
            "id": f"{datetime.now(timezone.utc).timestamp()}_{video.filename}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lift_type": response["lift_type"],
            "predicted_rpe": response["predicted_rpe"],
            "confidence": response["confidence"],
            "video_name": response["video_name"]
        }
        append_history_record(convert_to_json_serializable(history_record))
        
        # Convert numpy types to JSON-serializable types
        response = convert_to_json_serializable(response)
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )
    
    finally:
        # Cleanup temp files
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not cleanup temp dir: {e}")


@app.post("/batch_predict")
async def batch_predict():
    """
    Batch processing endpoint (for future implementation)
    """
    raise HTTPException(
        status_code=501,
        detail="Batch prediction not yet implemented"
    )


if __name__ == "__main__":
    import uvicorn
    import os
    print("\n" + "="*70)
    print("  AI WEIGHTLIFT COACH API SERVER")
    print("="*70)
    print("\nStarting server...")
    print("API docs available at /docs")
    print("\nPress CTRL+C to stop\n")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
