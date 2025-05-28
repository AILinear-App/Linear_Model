# main.py - MLflow 통합된 실제 AI 서버
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import clip
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from ultralytics import YOLO
import shutil
from typing import List
import uvicorn
import json
import base64
import io
import time
from datetime import datetime

# MLflow 임포트
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# FastAPI 앱 생성
app = FastAPI(title="CCTV AI 분석 서버 + MLflow", version="3.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow 설정
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CCTV_AI_Analysis")

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

# 성능 통계 저장
performance_stats = {
    "total_videos_processed": 0,
    "total_persons_detected": 0,
    "total_searches_performed": 0,
    "average_detection_confidence": 0.0,
    "processing_times": []
}

# 현재 MLflow 실행 정보
current_mlflow_info = {
    "run_id": None,
    "experiment_id": None
}

print(f"🔧 사용 중인 디바이스: {device}")

# 요청/응답 모델
class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResult(BaseModel):
    image_path: str
    caption: str
    score: float
    image_base64: str = ""

# 분석된 데이터 저장
analyzed_data = {
    "embeddings": None,
    "image_paths": [],
    "captions": [],
    "images": []
}

@app.on_event("startup")
async def startup_event():
    """서버 시작시 AI 모델 로드 및 MLflow 초기화"""
    global yolo_model, clip_model, clip_preprocess
    
    try:
        print("🤖 AI 모델 로딩 중...")
        
        # MLflow run 시작 (모델 로딩 추적)
        with mlflow.start_run(run_name="Model_Loading") as run:
            mlflow.log_param("device", device)
            mlflow.log_param("startup_time", datetime.now().isoformat())
            
            start_time = time.time()
            
            # YOLO 모델 로드
            print("📦 YOLO 모델 다운로드 중...")
            yolo_model = YOLO('yolov8n.pt')
            yolo_load_time = time.time() - start_time
            mlflow.log_metric("yolo_load_time_seconds", yolo_load_time)
            
            # CLIP 모델 로드
            print("🧠 CLIP 모델 로딩 중...")
            clip_start = time.time()
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            clip_load_time = time.time() - clip_start
            mlflow.log_metric("clip_load_time_seconds", clip_load_time)
            
            total_load_time = time.time() - start_time
            mlflow.log_metric("total_model_load_time_seconds", total_load_time)
            
            print("✅ 모든 AI 모델 로딩 완료!")
            print(f"   - YOLO: 사람 탐지 준비완료 ({yolo_load_time:.2f}초)")
            print(f"   - CLIP: 자연어 검색 준비완료 ({clip_load_time:.2f}초)")
            print(f"   - 총 로딩 시간: {total_load_time:.2f}초")
            
    except Exception as e:
        print(f"❌ AI 모델 로딩 실패: {e}")
        if mlflow.active_run():
            mlflow.log_param("loading_error", str(e))

@app.get("/")
async def 메인페이지():
    """서버 메인 페이지"""
    return {
        "메시지": "🤖 MLflow 통합 AI 분석 서버 실행중!",
        "YOLO_모델": "로딩됨" if yolo_model else "로딩실패",
        "CLIP_모델": "로딩됨" if clip_model else "로딩실패",
        "디바이스": device,
        "분석된_이미지": len(analyzed_data["image_paths"]),
        "MLflow_추적": "활성화됨"
    }

@app.get("/health")
async def 상태확인():
    """서버 상태 확인"""
    return {
        "상태": "정상",
        "AI_준비": all([yolo_model, clip_model]),
        "디바이스": device,
        "분석된_데이터": len(analyzed_data["image_paths"]),
        "MLflow_서버": "http://localhost:5000"
    }

@app.get("/stats/performance")
async def 성능통계():
    """MLflow 기반 성능 통계"""
    return {
        "전체_성능": {
            "total_videos_processed": performance_stats["total_videos_processed"],
            "total_persons_detected": performance_stats["total_persons_detected"],
            "total_searches_performed": performance_stats["total_searches_performed"],
            "average_detection_confidence": performance_stats["average_detection_confidence"],
            "average_processing_time": np.mean(performance_stats["processing_times"]) if performance_stats["processing_times"] else 0
        },
        "시스템_상태": {
            "YOLO_로딩됨": yolo_model is not None,
            "CLIP_로딩됨": clip_model is not None,
            "디바이스": device,
            "GPU_사용가능": torch.cuda.is_available()
        },
        "현재_MLflow_실행": current_mlflow_info
    }

def detect_persons_in_video(video_path: Path, max_frames: int = 30, run_id: str = None):
    """비디오에서 사람 탐지 및 crop 이미지 생성 (MLflow 추적 포함)"""
    
    if not yolo_model:
        raise Exception("YOLO 모델이 로드되지 않았습니다")
    
    print(f"🎬 비디오 분석 시작: {video_path.name}")
    
    # MLflow 메트릭 로깅을 위한 설정
    if run_id:
        mlflow.log_param("video_filename", video_path.name)
        mlflow.log_param("max_frames_to_analyze", max_frames)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    detected_persons = []
    frame_count = 0
    confidences = []
    
    # 프레임 간격 계산
    frame_interval = max(1, total_frames // max_frames)
    
    print(f"📊 총 프레임: {total_frames}, 분석할 프레임: {min(max_frames, total_frames)}")
    
    analysis_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % frame_interval != 0:
            continue
            
        try:
            # YOLO로 사람 탐지
            results = yolo_model(frame, classes=[0], verbose=False)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence > 0.5:
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            if (x2 - x1) > 50 and (y2 - y1) > 50:
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
                                
                                confidences.append(float(confidence))
                                print(f"✅ 사람 탐지: 프레임 {frame_count}, 신뢰도 {confidence:.2f}")
        
        except Exception as e:
            print(f"⚠️ 프레임 {frame_count} 분석 실패: {e}")
            continue
    
    cap.release()
    analysis_time = time.time() - analysis_start_time
    
    # MLflow에 분석 결과 기록
    if run_id:
        mlflow.log_metric("total_frames", total_frames)
        mlflow.log_metric("analyzed_frames", frame_count)
        mlflow.log_metric("persons_detected", len(detected_persons))
        mlflow.log_metric("analysis_time_seconds", analysis_time)
        mlflow.log_metric("fps", fps)
        
        if confidences:
            mlflow.log_metric("average_detection_confidence", np.mean(confidences))
            mlflow.log_metric("max_detection_confidence", np.max(confidences))
            mlflow.log_metric("min_detection_confidence", np.min(confidences))
        
        # 원본 비디오를 artifact로 저장
        mlflow.log_artifact(str(video_path), "input_videos")
    
    print(f"🎉 분석 완료: {len(detected_persons)}명의 사람 탐지 ({analysis_time:.2f}초)")
    
    return detected_persons

def generate_clip_embeddings(detected_persons, run_id: str = None):
    """탐지된 사람들의 CLIP 임베딩 생성 (MLflow 추적 포함)"""
    
    if not clip_model:
        raise Exception("CLIP 모델이 로드되지 않았습니다")
    
    print("🧠 CLIP 임베딩 생성 중...")
    
    embedding_start_time = time.time()
    
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
    
    embedding_time = time.time() - embedding_start_time
    
    # MLflow에 임베딩 생성 결과 기록
    if run_id:
        mlflow.log_metric("embeddings_generated", len(embeddings))
        mlflow.log_metric("embedding_generation_time_seconds", embedding_time)
        mlflow.log_param("clip_model", "ViT-B/32")
        mlflow.log_param("embedding_dimension", len(embeddings[0]) if embeddings else 0)
    
    print(f"✅ {len(embeddings)}개의 임베딩 생성 완료 ({embedding_time:.2f}초)")
    
    return {
        "embeddings": np.vstack(embeddings) if embeddings else None,
        "image_paths": image_paths,
        "captions": captions,
        "images": images
    }

@app.post("/upload-video")
async def 영상업로드(file: UploadFile = File(...)):
    """CCTV 영상 업로드 및 실제 AI 분석 (MLflow 추적 포함)"""
    
    print(f"📹 업로드 시작: {file.filename}")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다")
    
    # MLflow run 시작
    with mlflow.start_run(run_name=f"Video_Analysis_{file.filename}") as run:
        global current_mlflow_info, performance_stats
        
        current_mlflow_info["run_id"] = run.info.run_id
        current_mlflow_info["experiment_id"] = run.info.experiment_id
        
        try:
            # 파일 저장
            video_path = VIDEO_DIR / file.filename
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # MLflow에 기본 정보 기록
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("file_size_mb", video_path.stat().st_size / (1024*1024))
            mlflow.log_param("upload_timestamp", datetime.now().isoformat())
            
            print(f"💾 파일 저장 완료: {video_path}")
            
            if not yolo_model or not clip_model:
                mlflow.log_param("status", "models_not_loaded")
                return {
                    "status": "warning",
                    "message": "AI 모델이 로드되지 않아 분석할 수 없습니다",
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_experiment_id": run.info.experiment_id
                }
            
            # 1. YOLO로 사람 탐지
            detected_persons = detect_persons_in_video(video_path, max_frames=20, run_id=run.info.run_id)
            
            if not detected_persons:
                mlflow.log_param("status", "no_persons_detected")
                return {
                    "status": "success",
                    "message": "영상 분석 완료, 하지만 탐지된 사람이 없습니다",
                    "total_crops": 0,
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_experiment_id": run.info.experiment_id
                }
            
            # 2. CLIP 임베딩 생성
            embedding_data = generate_clip_embeddings(detected_persons, run_id=run.info.run_id)
            
            # 3. 글로벌 데이터에 저장
            global analyzed_data
            analyzed_data = embedding_data
            
            # 4. 성능 통계 업데이트
            performance_stats["total_videos_processed"] += 1
            performance_stats["total_persons_detected"] += len(detected_persons)
            avg_confidence = np.mean([p["신뢰도"] for p in detected_persons])
            performance_stats["average_detection_confidence"] = avg_confidence
            
            # MLflow에 최종 상태 기록
            mlflow.log_param("status", "success")
            mlflow.log_metric("final_persons_count", len(detected_persons))
            mlflow.log_metric("final_embeddings_count", len(embedding_data["embeddings"]) if embedding_data["embeddings"] is not None else 0)
            
            return {
                "status": "success", 
                "message": f"'{file.filename}' 분석 완료!",
                "total_crops": len(detected_persons),
                "분석결과": f"{len(detected_persons)}명의 실제 인물이 AI로 탐지되었습니다",
                "mlflow_run_id": run.info.run_id,
                "mlflow_experiment_id": run.info.experiment_id
            }
            
        except Exception as e:
            mlflow.log_param("status", "error")
            mlflow.log_param("error_message", str(e))
            print(f"❌ 분석 오류: {e}")
            raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def 인물검색(request: SearchRequest):
    """실제 CLIP 기반 자연어 검색 (MLflow 추적 포함)"""
    
    print(f"🔍 실제 AI 검색: '{request.query}'")
    
    if analyzed_data["embeddings"] is None:
        raise HTTPException(status_code=404, detail="먼저 영상을 업로드하고 분석해주세요")
    
    if not clip_model:
        raise HTTPException(status_code=500, detail="CLIP 모델이 로드되지 않았습니다")
    
    # MLflow run 시작 (검색 추적)
    with mlflow.start_run(run_name=f"Search_{request.query[:20]}", nested=True) as search_run:
        global performance_stats
        
        try:
            search_start_time = time.time()
            
            # MLflow에 검색 파라미터 기록
            mlflow.log_param("search_query", request.query)
            mlflow.log_param("k", request.k)
            mlflow.log_param("search_timestamp", datetime.now().isoformat())
            
            # 검색어를 CLIP으로 인코딩
            text_input = clip.tokenize([request.query]).to(device)
            
            with torch.no_grad():
                text_features = clip_model.encode_text(text_input)
                text_features = text_features.cpu().numpy()[0]
            
            # 코사인 유사도 계산
            image_embeddings = analyzed_data["embeddings"]
            similarities = np.dot(image_embeddings, text_features) / (
                np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(text_features)
            )
            
            # 상위 k개 결과 선택
            top_indices = np.argsort(-similarities)[:request.k]
            
            results = []
            similarity_scores = []
            
            for idx in top_indices:
                if similarities[idx] > 0.1:  # 최소 유사도 임계값
                    results.append(SearchResult(
                        image_path=analyzed_data["image_paths"][idx],
                        caption=analyzed_data["captions"][idx],
                        score=float(similarities[idx]),
                        image_base64=analyzed_data["images"][idx]
                    ))
                    similarity_scores.append(float(similarities[idx]))
            
            search_time = time.time() - search_start_time
            
            # MLflow에 검색 결과 기록
            mlflow.log_metric("search_results_count", len(results))
            mlflow.log_metric("search_time_seconds", search_time)
            mlflow.log_metric("max_similarity_score", np.max(similarity_scores) if similarity_scores else 0)
            mlflow.log_metric("avg_similarity_score", np.mean(similarity_scores) if similarity_scores else 0)
            mlflow.log_metric("min_similarity_threshold", 0.1)
            
            # 성능 통계 업데이트
            performance_stats["total_searches_performed"] += 1
            performance_stats["processing_times"].append(search_time)
            
            print(f"✅ 실제 AI 검색 완료: {len(results)}개 결과 ({search_time:.2f}초)")
            return results
            
        except Exception as e:
            mlflow.log_param("search_error", str(e))
            print(f"❌ 검색 오류: {e}")
            raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.get("/image/{filename}")
async def get_image(filename: str):
    """crop 이미지 파일 반환"""
    image_path = CROP_DIR / filename
    if image_path.exists():
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

@app.get("/stats")
async def 통계():
    """서버 통계"""
    return {
        "AI_모델_상태": {
            "YOLO": "로딩됨" if yolo_model else "로딩실패",
            "CLIP": "로딩됨" if clip_model else "로딩실패"
        },
        "분석_통계": {
            "탐지된_인물": len(analyzed_data["image_paths"]),
            "임베딩_생성됨": analyzed_data["embeddings"] is not None
        },
        "시스템_정보": {
            "디바이스": device,
            "GPU_사용가능": torch.cuda.is_available()
        },
        "MLflow_정보": current_mlflow_info
    }

# 서버 실행
if __name__ == "__main__":
    print("🚀 MLflow 통합 AI 분석 서버 시작!")
    print("🤖 YOLO + CLIP + MLflow 기반 인물 탐지 및 검색")
    print("📍 AI 서버: http://localhost:8001")
    print("📍 MLflow UI: http://localhost:5000")
    print("📚 API 문서: http://localhost:8001/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)