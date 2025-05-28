# main.py - 완전 기능 버전 (MLflow + 모든 AI 기능)
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import clip
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import shutil
from typing import List, Dict, Any
import uvicorn
import base64
import io
import time
import uuid
from datetime import datetime
import logging

# MLflow 관련 imports (선택사항)
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
    print("✅ MLflow 사용 가능")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow 없음 - 기본 로깅 사용")

# 환경 변수
PORT = int(os.getenv("PORT", 8001))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

app = FastAPI(
    title="CCTV AI Analysis Server - Full Feature",
    version="3.0.0",
    description="Complete AI server with YOLO + CLIP + MLflow"
)

# CORS 설정
if ENVIRONMENT == "production":
    allowed_origins = [
        "https://your-frontend-app.onrender.com",
        "https://your-app.vercel.app",
        "https://localhost:3000"
    ]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 저장 폴더
ROOT_DIR = Path('./cctv_data')
VIDEO_DIR = ROOT_DIR / 'videos'
CROP_DIR = ROOT_DIR / 'crops'
DATA_DIR = ROOT_DIR / 'data'

# 폴더 생성
for folder in [ROOT_DIR, VIDEO_DIR, CROP_DIR, DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 전역 변수
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = None
clip_model = None
clip_preprocess = None

# MLflow 설정
if MLFLOW_AVAILABLE:
    mlflow.set_experiment("CCTV_Person_Detection")

# 현재 실험 설정
current_config = {
    "yolo_confidence_threshold": 0.5,
    "yolo_model_size": "yolov8n",
    "clip_model_type": "ViT-B/32",
    "max_frames_per_video": 30,
    "min_person_size": 50,
    "search_similarity_threshold": 0.1
}

# 요청/응답 모델
class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResult(BaseModel):
    image_path: str
    caption: str
    score: float
    image_base64: str = ""

class ExperimentConfig(BaseModel):
    yolo_confidence_threshold: float = 0.5
    yolo_model_size: str = "yolov8n"
    max_frames_per_video: int = 30
    min_person_size: int = 50

# 분석된 데이터 저장
analyzed_data = {
    "embeddings": None,
    "image_paths": [],
    "captions": [],
    "images": [],
    "experiment_id": None,
    "run_id": None
}

# 성능 메트릭 저장
performance_metrics = {
    "total_videos_processed": 0,
    "total_persons_detected": 0,
    "average_detection_confidence": 0.0,
    "processing_times": [],
    "search_accuracies": []
}

@app.on_event("startup")
async def startup_event():
    """서버 시작시 AI 모델 로드"""
    global yolo_model, clip_model, clip_preprocess
    
    try:
        print("🚀 Complete AI 서버 시작!")
        print(f"🌍 환경: {ENVIRONMENT}")
        print(f"🔧 디바이스: {device}")
        print(f"📊 MLflow: {'사용' if MLFLOW_AVAILABLE else '미사용'}")
        
        # MLflow 실험 시작
        run_info = None
        if MLFLOW_AVAILABLE:
            run = mlflow.start_run(run_name=f"Server_Startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            run_info = run.info
            
            # 시스템 정보 로깅
            mlflow.log_param("device", device)
            mlflow.log_param("environment", ENVIRONMENT)
            for key, value in current_config.items():
                mlflow.log_param(key, value)
        
        print("🤖 AI 모델 로딩 중...")
        start_time = time.time()
        
        # YOLO 모델 로드
        print(f"📦 YOLO 모델 로딩: {current_config['yolo_model_size']}")
        yolo_model = YOLO(f"{current_config['yolo_model_size']}.pt")
        yolo_load_time = time.time() - start_time
        
        # CLIP 모델 로드
        print(f"🧠 CLIP 모델 로딩: {current_config['clip_model_type']}")
        clip_start = time.time()
        clip_model, clip_preprocess = clip.load(current_config['clip_model_type'], device=device)
        clip_load_time = time.time() - clip_start
        
        total_load_time = time.time() - start_time
        
        # 로딩 시간 로깅
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("yolo_load_time_seconds", yolo_load_time)
            mlflow.log_metric("clip_load_time_seconds", clip_load_time)
            mlflow.log_metric("total_load_time_seconds", total_load_time)
            mlflow.log_param("yolo_model_loaded", True)
            mlflow.log_param("clip_model_loaded", True)
        
        print("✅ 모든 AI 모델 로딩 완료!")
        print(f"   - YOLO 로딩 시간: {yolo_load_time:.2f}초")
        print(f"   - CLIP 로딩 시간: {clip_load_time:.2f}초")
        print(f"   - 총 로딩 시간: {total_load_time:.2f}초")
        
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        if MLFLOW_AVAILABLE:
            mlflow.log_param("initialization_error", str(e))
            mlflow.end_run()

@app.get("/")
async def root():
    """메인 페이지"""
    return {
        "service": "CCTV AI Analysis Server - Full Feature",
        "version": "3.0.0",
        "environment": ENVIRONMENT,
        "mlflow_enabled": MLFLOW_AVAILABLE,
        "models": {
            "yolo": yolo_model is not None,
            "clip": clip_model is not None
        },
        "config": current_config,
        "stats": performance_metrics,
        "message": "✅ 완전 기능 서버 실행중!"
    }

@app.get("/health")
async def health_check():
    """상세 헬스체크"""
    return {
        "status": "healthy",
        "models": {
            "yolo": yolo_model is not None,
            "clip": clip_model is not None
        },
        "mlflow": MLFLOW_AVAILABLE,
        "device": device,
        "analyzed_data": len(analyzed_data["image_paths"]),
        "performance": performance_metrics
    }

def detect_persons_in_video(video_path: Path, max_frames: int = 30):
    """비디오에서 사람 탐지"""
    
    if not yolo_model:
        raise Exception("YOLO 모델이 로드되지 않았습니다")
    
    print(f"🎬 비디오 분석 시작: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    detected_persons = []
    frame_count = 0
    
    # 프레임 간격 계산
    frame_interval = max(1, total_frames // max_frames)
    
    print(f"📊 총 프레임: {total_frames}, 분석할 프레임: {min(max_frames, total_frames)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % frame_interval != 0:
            continue
            
        try:
            # YOLO 추론
            results = yolo_model(frame, classes=[0], verbose=False)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence > current_config["yolo_confidence_threshold"]:
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            if (x2 - x1) > current_config["min_person_size"] and (y2 - y1) > current_config["min_person_size"]:
                                person_img = frame[y1:y2, x1:x2]
                                person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                                
                                filename = f"{video_path.stem}_frame{frame_count}_person{len(detected_persons)}.jpg"
                                crop_path = CROP_DIR / filename
                                person_pil.save(crop_path)
                                
                                detected_persons.append({
                                    "파일경로": str(crop_path),
                                    "프레임번호": frame_count,
                                    "신뢰도": float(confidence),
                                    "박스좌표": [x1, y1, x2, y2],
                                    "이미지": person_pil
                                })
                                
                                print(f"✅ 사람 탐지: 프레임 {frame_count}, 신뢰도 {confidence:.2f}")
        
        except Exception as e:
            print(f"⚠️ 프레임 {frame_count} 분석 실패: {e}")
            continue
    
    cap.release()
    print(f"🎉 분석 완료: {len(detected_persons)}명의 사람 탐지")
    
    return detected_persons

def generate_clip_embeddings(detected_persons):
    """CLIP 임베딩 생성"""
    
    if not clip_model:
        raise Exception("CLIP 모델이 로드되지 않았습니다")
    
    print("🧠 CLIP 임베딩 생성 중...")
    
    embeddings = []
    image_paths = []
    captions = []
    images = []
    
    for i, person in enumerate(detected_persons):
        try:
            image = person["이미지"]
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features = image_features.cpu().numpy()[0]
            
            embeddings.append(image_features)
            image_paths.append(person["파일경로"])
            
            caption = f"프레임 {person['프레임번호']}에서 탐지된 인물 (신뢰도: {person['신뢰도']:.1%})"
            captions.append(caption)
            
            # Base64 인코딩
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_base64)
            
        except Exception as e:
            print(f"⚠️ 임베딩 생성 실패 {i}: {e}")
            continue
    
    print(f"✅ {len(embeddings)}개의 임베딩 생성 완료")
    
    return {
        "embeddings": np.vstack(embeddings) if embeddings else None,
        "image_paths": image_paths,
        "captions": captions,
        "images": images
    }

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """영상 업로드 및 완전 AI 분석"""
    
    print(f"📹 완전 AI 분석 시작: {file.filename}")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식")
    
    # MLflow 실험 시작
    mlflow_run_id = None
    mlflow_experiment_id = None
    
    if MLFLOW_AVAILABLE:
        run = mlflow.start_run(run_name=f"Video_Analysis_{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow_run_id = run.info.run_id
        mlflow_experiment_id = run.info.experiment_id
        
        mlflow.log_param("filename", file.filename)
        mlflow.log_param("upload_timestamp", datetime.now().isoformat())
    
    try:
        # 파일 저장
        video_path = VIDEO_DIR / file.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if MLFLOW_AVAILABLE:
            mlflow.log_param("file_size_mb", video_path.stat().st_size / 1024 / 1024)
        
        # 사람 탐지
        detected_persons = detect_persons_in_video(video_path, current_config["max_frames_per_video"])
        
        if not detected_persons:
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("persons_detected", 0)
                mlflow.log_metric("analysis_success", 1)
                mlflow.end_run()
            
            return {
                "status": "success",
                "message": "분석 완료, 탐지된 사람 없음",
                "total_crops": 0,
                "mlflow_run_id": mlflow_run_id,
                "mlflow_experiment_id": mlflow_experiment_id
            }
        
        # CLIP 임베딩 생성
        embedding_data = generate_clip_embeddings(detected_persons)
        
        # 성능 메트릭 업데이트
        performance_metrics["total_videos_processed"] += 1
        performance_metrics["total_persons_detected"] += len(detected_persons)
        avg_confidence = sum(p["신뢰도"] for p in detected_persons) / len(detected_persons)
        performance_metrics["average_detection_confidence"] = avg_confidence
        
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("persons_detected", len(detected_persons))
            mlflow.log_metric("average_confidence", avg_confidence)
            mlflow.log_metric("analysis_success", 1)
            mlflow.end_run()
        
        # 글로벌 데이터 업데이트
        global analyzed_data
        analyzed_data = embedding_data
        analyzed_data["experiment_id"] = mlflow_experiment_id
        analyzed_data["run_id"] = mlflow_run_id
        
        return {
            "status": "success",
            "message": f"'{file.filename}' 완전 AI 분석 완료!",
            "total_crops": len(detected_persons),
            "mlflow_run_id": mlflow_run_id,
            "mlflow_experiment_id": mlflow_experiment_id,
            "분석결과": f"{len(detected_persons)}명의 실제 인물이 AI로 탐지되었습니다"
        }
        
    except Exception as e:
        if MLFLOW_AVAILABLE:
            mlflow.log_param("error", str(e))
            mlflow.log_metric("analysis_success", 0)
            mlflow.end_run()
        
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_persons(request: SearchRequest):
    """완전 CLIP 기반 검색"""
    
    if analyzed_data["embeddings"] is None:
        raise HTTPException(status_code=404, detail="먼저 영상을 업로드하고 분석해주세요")
    
    # MLflow 검색 실험 시작
    if MLFLOW_AVAILABLE:
        search_run = mlflow.start_run(run_name=f"Search_{request.query[:20]}", nested=True)
        mlflow.log_param("search_query", request.query)
        mlflow.log_param("requested_results", request.k)
    
    try:
        # CLIP 텍스트 인코딩
        text_input = clip.tokenize([request.query]).to(device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(text_input)
            text_features = text_features.cpu().numpy()[0]
        
        # 유사도 계산
        image_embeddings = analyzed_data["embeddings"]
        similarities = np.dot(image_embeddings, text_features) / (
            np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(text_features)
        )
        
        # 상위 결과 선택
        top_indices = np.argsort(-similarities)[:request.k]
        
        results = []
        valid_results = 0
        
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity > current_config["search_similarity_threshold"]:
                results.append(SearchResult(
                    image_path=analyzed_data["image_paths"][idx],
                    caption=analyzed_data["captions"][idx],
                    score=float(similarity),
                    image_base64=analyzed_data["images"][idx]
                ))
                valid_results += 1
        
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("results_returned", valid_results)
            mlflow.log_metric("max_similarity", max(similarities) if len(similarities) > 0 else 0)
            mlflow.end_run()
        
        print(f"🔍 완전 AI 검색 완료: {valid_results}개 결과")
        return results
        
    except Exception as e:
        if MLFLOW_AVAILABLE:
            mlflow.log_param("search_error", str(e))
            mlflow.end_run()
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.get("/stats/performance")
async def get_performance_stats():
    """상세 성능 통계"""
    return {
        "전체_성능": performance_metrics,
        "현재_설정": current_config,
        "MLflow_정보": {
            "사용가능": MLFLOW_AVAILABLE,
            "실험_ID": analyzed_data.get("experiment_id"),
            "런_ID": analyzed_data.get("run_id"),
            "UI_링크": "http://localhost:5000" if MLFLOW_AVAILABLE else "MLflow 미사용"
        },
        "시스템_상태": {
            "YOLO_로딩됨": yolo_model is not None,
            "CLIP_로딩됨": clip_model is not None,
            "디바이스": device,
            "분석된_데이터": len(analyzed_data["image_paths"])
        }
    }

@app.get("/mlflow/experiments")
async def get_mlflow_experiments():
    """MLflow 실험 목록"""
    if not MLFLOW_AVAILABLE:
        return {"error": "MLflow가 설치되지 않았습니다"}
    
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        
        experiment_info = []
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id, max_results=10)
            experiment_info.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "total_runs": len(runs),
                "recent_runs": [
                    {
                        "run_id": run.info.run_id,
                        "run_name": run.data.tags.get("mlflow.runName", "Unnamed"),
                        "start_time": run.info.start_time,
                        "status": run.info.status
                    } for run in runs[:5]
                ]
            })
        
        return {"experiments": experiment_info}
        
    except Exception as e:
        return {"error": f"MLflow 연결 실패: {str(e)}"}

@app.get("/image/{filename}")
async def get_image(filename: str):
    """이미지 파일 반환"""
    image_path = CROP_DIR / filename
    if image_path.exists():
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

# 서버 실행
if __name__ == "__main__":
    print("🚀 완전 기능 AI 분석 서버 시작!")
    print("🤖 YOLO + CLIP + MLflow 모든 기능 포함")
    print("=" * 60)
    print(f"📍 서버: http://localhost:{PORT}")
    print(f"📊 MLflow: {'사용' if MLFLOW_AVAILABLE else '미사용'}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)