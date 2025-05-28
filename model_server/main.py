# main.py - MLflow가 통합된 AI 서버
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import clip
import cv2
import numpy as np
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

# MLflow 관련 imports
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# FastAPI 앱 생성
app = FastAPI(title="CCTV AI 분석 서버 (MLflow 통합)", version="3.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow 설정
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow 서버 주소
mlflow.set_experiment("CCTV_Person_Detection")

# 데이터 저장 폴더
ROOT_DIR = Path('./cctv_data')
VIDEO_DIR = ROOT_DIR / 'videos'
CROP_DIR = ROOT_DIR / 'crops'
DATA_DIR = ROOT_DIR / 'data'
MLFLOW_DIR = ROOT_DIR / 'mlruns'

# 폴더 생성
for folder in [ROOT_DIR, VIDEO_DIR, CROP_DIR, DATA_DIR, MLFLOW_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 전역 변수
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = None
clip_model = None
clip_preprocess = None

# 현재 실험 설정 (하이퍼파라미터)
current_config = {
    "yolo_confidence_threshold": 0.5,
    "yolo_model_size": "yolov8n",  # nano, small, medium, large
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
    """서버 시작시 AI 모델 로드 및 MLflow 초기화"""
    global yolo_model, clip_model, clip_preprocess
    
    try:
        print("🚀 MLflow 통합 AI 서버 시작")
        print("=" * 60)
        
        # MLflow 실험 시작
        with mlflow.start_run(run_name=f"Server_Startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print("📊 MLflow 실험 시작...")
            
            # 시스템 정보 로깅
            mlflow.log_param("device", device)
            mlflow.log_param("pytorch_version", torch.__version__)
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            
            # 하이퍼파라미터 로깅
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
            
            # 로딩 시간 로깅
            mlflow.log_metric("yolo_load_time_seconds", yolo_load_time)
            mlflow.log_metric("clip_load_time_seconds", clip_load_time)
            mlflow.log_metric("total_load_time_seconds", time.time() - start_time)
            
            # 모델 정보 로깅
            mlflow.log_param("yolo_model_loaded", True)
            mlflow.log_param("clip_model_loaded", True)
            mlflow.log_param("total_parameters", "estimated_millions")  # 실제 계산 가능
            
            print("✅ 모든 AI 모델 로딩 완료!")
            print(f"   - YOLO 로딩 시간: {yolo_load_time:.2f}초")
            print(f"   - CLIP 로딩 시간: {clip_load_time:.2f}초")
            print(f"   - 총 로딩 시간: {time.time() - start_time:.2f}초")
            
            # 모델 아티팩트 저장 준비 (선택사항)
            # mlflow.pytorch.log_model(clip_model, "clip_model")
            
        print("📊 MLflow 초기화 완료")
        print("🌐 MLflow UI: http://localhost:5000")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        with mlflow.start_run():
            mlflow.log_param("initialization_error", str(e))

@app.get("/")
async def 메인페이지():
    """서버 메인 페이지"""
    return {
        "메시지": "🚀 MLflow 통합 AI 분석 서버",
        "버전": "3.0.0",
        "MLflow_UI": "http://localhost:5000",
        "모델_상태": {
            "YOLO": "로딩됨" if yolo_model else "로딩실패",
            "CLIP": "로딩됨" if clip_model else "로딩실패"
        },
        "현재_설정": current_config,
        "성능_통계": performance_metrics
    }

@app.get("/health")
async def 상태확인():
    """서버 상태 확인"""
    return {
        "상태": "정상",
        "AI_준비": all([yolo_model, clip_model]),
        "MLflow_연결": True,
        "실험_정보": {
            "현재_실험": mlflow.get_experiment_by_name("CCTV_Person_Detection"),
            "활성_런": analyzed_data.get("run_id")
        }
    }

def detect_persons_in_video_with_mlflow(video_path: Path, run_id: str = None):
    """MLflow 추적이 포함된 비디오 사람 탐지"""
    
    if not yolo_model:
        raise Exception("YOLO 모델이 로드되지 않았습니다")
    
    # MLflow 실험 컨텍스트에서 실행
    with mlflow.start_run(run_id=run_id, nested=True) as run:
        
        # 비디오 정보 로깅
        mlflow.log_param("video_filename", video_path.name)
        mlflow.log_param("video_size_mb", video_path.stat().st_size / 1024 / 1024)
        
        print(f"🎬 비디오 분석 시작: {video_path.name}")
        processing_start_time = time.time()
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        
        # 비디오 메타데이터 로깅
        mlflow.log_param("total_frames", total_frames)
        mlflow.log_param("fps", fps)
        mlflow.log_param("duration_seconds", duration)
        
        detected_persons = []
        frame_count = 0
        confidences = []
        
        # 프레임 간격 계산
        max_frames = current_config["max_frames_per_video"]
        frame_interval = max(1, total_frames // max_frames)
        mlflow.log_param("frame_interval", frame_interval)
        mlflow.log_param("frames_to_analyze", min(max_frames, total_frames))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % frame_interval != 0:
                continue
                
            try:
                # YOLO 추론 시간 측정
                inference_start = time.time()
                results = yolo_model(frame, classes=[0], verbose=False)
                inference_time = time.time() - inference_start
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            # 신뢰도 임계값 적용
                            if confidence > current_config["yolo_confidence_threshold"]:
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # 크기 임계값 적용
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
                                        "이미지": person_pil,
                                        "추론시간": inference_time
                                    })
                                    
                                    confidences.append(float(confidence))
            
            except Exception as e:
                mlflow.log_param(f"frame_{frame_count}_error", str(e))
                continue
        
        cap.release()
        processing_time = time.time() - processing_start_time
        
        # 성능 메트릭 계산 및 로깅
        avg_confidence = np.mean(confidences) if confidences else 0.0
        detection_rate = len(detected_persons) / (total_frames / frame_interval) if total_frames > 0 else 0
        
        mlflow.log_metric("persons_detected", len(detected_persons))
        mlflow.log_metric("average_confidence", avg_confidence)
        mlflow.log_metric("detection_rate", detection_rate)
        mlflow.log_metric("processing_time_seconds", processing_time)
        mlflow.log_metric("fps_processing", (total_frames / frame_interval) / processing_time if processing_time > 0 else 0)
        
        # 신뢰도 분포 히스토그램 저장 (선택사항)
        if confidences:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(confidences, bins=20, alpha=0.7)
            plt.title('Detection Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            
            hist_path = DATA_DIR / f"confidence_hist_{run.info.run_id}.png"
            plt.savefig(hist_path)
            mlflow.log_artifact(str(hist_path))
            plt.close()
        
        print(f"🎉 분석 완료: {len(detected_persons)}명 탐지")
        print(f"   - 평균 신뢰도: {avg_confidence:.3f}")
        print(f"   - 처리 시간: {processing_time:.2f}초")
        
        # 글로벌 성능 메트릭 업데이트
        performance_metrics["total_videos_processed"] += 1
        performance_metrics["total_persons_detected"] += len(detected_persons)
        performance_metrics["processing_times"].append(processing_time)
        performance_metrics["average_detection_confidence"] = (
            performance_metrics["average_detection_confidence"] + avg_confidence
        ) / 2 if performance_metrics["average_detection_confidence"] > 0 else avg_confidence
        
        return detected_persons, run.info.run_id

def generate_clip_embeddings_with_mlflow(detected_persons, run_id: str = None):
    """MLflow 추적이 포함된 CLIP 임베딩 생성"""
    
    if not clip_model:
        raise Exception("CLIP 모델이 로드되지 않았습니다")
    
    with mlflow.start_run(run_id=run_id, nested=True):
        print("🧠 CLIP 임베딩 생성 중...")
        embedding_start_time = time.time()
        
        embeddings = []
        image_paths = []
        captions = []
        images = []
        embedding_times = []
        
        mlflow.log_param("total_images_to_embed", len(detected_persons))
        
        for i, person in enumerate(detected_persons):
            try:
                embed_start = time.time()
                
                image = person["이미지"]
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    image_features = image_features.cpu().numpy()[0]
                
                embed_time = time.time() - embed_start
                embedding_times.append(embed_time)
                
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
                mlflow.log_param(f"embedding_error_{i}", str(e))
                continue
        
        total_embedding_time = time.time() - embedding_start_time
        avg_embedding_time = np.mean(embedding_times) if embedding_times else 0
        
        # 임베딩 성능 메트릭 로깅
        mlflow.log_metric("embeddings_generated", len(embeddings))
        mlflow.log_metric("total_embedding_time_seconds", total_embedding_time)
        mlflow.log_metric("average_embedding_time_seconds", avg_embedding_time)
        mlflow.log_metric("embeddings_per_second", len(embeddings) / total_embedding_time if total_embedding_time > 0 else 0)
        
        print(f"✅ {len(embeddings)}개의 임베딩 생성 완료")
        print(f"   - 총 임베딩 시간: {total_embedding_time:.2f}초")
        print(f"   - 평균 임베딩 시간: {avg_embedding_time:.4f}초")
        
        return {
            "embeddings": np.vstack(embeddings) if embeddings else None,
            "image_paths": image_paths,
            "captions": captions,
            "images": images
        }

@app.post("/upload-video")
async def 영상업로드(file: UploadFile = File(...)):
    """MLflow 추적이 포함된 영상 업로드 및 분석"""
    
    print(f"📹 MLflow 추적 영상 업로드: {file.filename}")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식")
    
    try:
        # 파일 저장
        video_path = VIDEO_DIR / file.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # MLflow 실험 실행
        with mlflow.start_run(run_name=f"Video_Analysis_{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # 파일 정보 로깅
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("file_size_mb", video_path.stat().st_size / 1024 / 1024)
            mlflow.log_param("upload_timestamp", datetime.now().isoformat())
            
            # 비디오 분석
            detected_persons, detection_run_id = detect_persons_in_video_with_mlflow(video_path, run.info.run_id)
            
            if not detected_persons:
                mlflow.log_metric("analysis_success", 0)
                return {
                    "status": "success",
                    "message": "분석 완료, 탐지된 사람 없음",
                    "total_crops": 0,
                    "mlflow_run_id": run.info.run_id
                }
            
            # CLIP 임베딩 생성
            embedding_data = generate_clip_embeddings_with_mlflow(detected_persons, run.info.run_id)
            
            # 전체 분석 성공 로깅
            mlflow.log_metric("analysis_success", 1)
            mlflow.log_metric("total_pipeline_persons", len(detected_persons))
            
            # 글로벌 데이터 업데이트
            global analyzed_data
            analyzed_data = embedding_data
            analyzed_data["experiment_id"] = run.info.experiment_id
            analyzed_data["run_id"] = run.info.run_id
            
            return {
                "status": "success",
                "message": f"'{file.filename}' MLflow 추적 완료!",
                "total_crops": len(detected_persons),
                "mlflow_run_id": run.info.run_id,
                "mlflow_experiment_id": run.info.experiment_id,
                "분석결과": f"{len(detected_persons)}명의 실제 인물이 AI로 탐지되었습니다"
            }
            
    except Exception as e:
        # 오류도 MLflow에 로깅
        with mlflow.start_run():
            mlflow.log_param("error_type", "upload_analysis_error")
            mlflow.log_param("error_message", str(e))
            mlflow.log_metric("analysis_success", 0)
        
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def 인물검색(request: SearchRequest):
    """MLflow 추적이 포함된 CLIP 기반 검색"""
    
    if analyzed_data["embeddings"] is None:
        raise HTTPException(status_code=404, detail="먼저 영상을 업로드하고 분석해주세요")
    
    # 검색 실험 시작
    with mlflow.start_run(run_name=f"Search_{request.query[:20]}_{datetime.now().strftime('%H%M%S')}", nested=True):
        
        search_start_time = time.time()
        
        # 검색 파라미터 로깅
        mlflow.log_param("search_query", request.query)
        mlflow.log_param("requested_results", request.k)
        mlflow.log_param("available_embeddings", len(analyzed_data["image_paths"]))
        
        try:
            # 텍스트 인코딩
            text_input = clip.tokenize([request.query]).to(device)
            
            with torch.no_grad():
                text_features = clip_model.encode_text(text_input)
                text_features = text_features.cpu().numpy()[0]
            
            # 유사도 계산
            image_embeddings = analyzed_data["embeddings"]
            similarities = np.dot(image_embeddings, text_features) / (
                np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(text_features)
            )
            
            # 결과 선택
            top_indices = np.argsort(-similarities)[:request.k]
            
            results = []
            valid_results = 0
            similarity_scores = []
            
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > current_config["search_similarity_threshold"]:
                    results.append(SearchResult(
                        image_path=analyzed_data["image_paths"][idx],
                        caption=analyzed_data["captions"][idx],
                        score=float(similarity),
                        image_base64=analyzed_data["images"][idx]
                    ))
                    similarity_scores.append(float(similarity))
                    valid_results += 1
            
            search_time = time.time() - search_start_time
            
            # 검색 성능 메트릭 로깅
            mlflow.log_metric("search_time_seconds", search_time)
            mlflow.log_metric("results_returned", valid_results)
            mlflow.log_metric("max_similarity", max(similarity_scores) if similarity_scores else 0)
            mlflow.log_metric("avg_similarity", np.mean(similarity_scores) if similarity_scores else 0)
            mlflow.log_metric("min_similarity", min(similarity_scores) if similarity_scores else 0)
            
            # 검색 성공률 업데이트
            performance_metrics["search_accuracies"].append(max(similarity_scores) if similarity_scores else 0)
            
            print(f"🔍 MLflow 추적 검색 완료: {valid_results}개 결과")
            return results
            
        except Exception as e:
            mlflow.log_param("search_error", str(e))
            mlflow.log_metric("search_success", 0)
            raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.get("/mlflow/experiments")
async def mlflow_실험목록():
    """MLflow 실험 목록 조회"""
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

@app.post("/config/update")
async def 설정업데이트(config: ExperimentConfig):
    """실험 설정 업데이트"""
    global current_config
    
    with mlflow.start_run(run_name=f"Config_Update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # 기존 설정 로깅
        for key, value in current_config.items():
            mlflow.log_param(f"old_{key}", value)
        
        # 새 설정 적용 및 로깅
        current_config.update(config.dict())
        for key, value in current_config.items():
            mlflow.log_param(f"new_{key}", value)
        
        mlflow.log_param("config_update_timestamp", datetime.now().isoformat())
    
    return {
        "status": "success",
        "message": "설정이 업데이트되었습니다",
        "new_config": current_config
    }

@app.get("/stats/performance")
async def 성능통계():
    """상세 성능 통계"""
    return {
        "전체_성능": performance_metrics,
        "현재_설정": current_config,
        "MLflow_정보": {
            "실험_ID": analyzed_data.get("experiment_id"),
            "런_ID": analyzed_data.get("run_id"),
            "UI_링크": "http://localhost:5000"
        },
        "시스템_상태": {
            "YOLO_로딩됨": yolo_model is not None,
            "CLIP_로딩됨": clip_model is not None,
            "디바이스": device,
            "분석된_데이터": len(analyzed_data["image_paths"])
        }
    }

# 서버 실행
if __name__ == "__main__":
    print("🚀 MLflow 통합 AI 분석 서버 시작!")
    print("🔬 실험 추적 및 모델 관리 포함")
    print("=" * 60)
    print("📍 서버: http://localhost:8001")
    print("📊 MLflow UI: http://localhost:5000")
    print("📚 API 문서: http://localhost:8001/docs")
    print("🌐 프론트엔드: http://localhost:5173")
    print("=" * 60)
    print("💡 MLflow UI 실행: 별도 터미널에서 'mlflow ui' 실행")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)