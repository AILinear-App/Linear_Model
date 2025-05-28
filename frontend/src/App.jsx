import React, { useState, useEffect } from 'react';
import './App.css';

const API_URL = 'http://localhost:8001';

function App() {
  // 기본 상태 관리
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [serverStatus, setServerStatus] = useState('확인 중...');
  
  // MLflow 관련 상태
  const [mlflowInfo, setMlflowInfo] = useState({});
  const [performanceStats, setPerformanceStats] = useState({});

  // 서버 상태 및 성능 통계 확인
  useEffect(() => {
    checkServer();
    fetchPerformanceStats();
  }, []);

  const checkServer = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        setServerStatus('✅ AI + MLflow 서버 연결됨');
      } else {
        setServerStatus('❌ 서버 응답 없음');
      }
    } catch (error) {
      setServerStatus('❌ 서버 연결 실패');
    }
  };

  // 성능 통계 가져오기
  const fetchPerformanceStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats/performance`);
      if (response.ok) {
        const data = await response.json();
        setPerformanceStats(data);
      }
    } catch (error) {
      console.log('성능 통계 가져오기 실패:', error);
    }
  };

  // 파일 업로드 (MLflow 추적 포함)
  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setUploadStatus('📤 MLflow 추적 업로드 중...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/upload-video`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setUploadStatus(`✅ ${result.message}`);
        
        // MLflow 정보 업데이트
        if (result.mlflow_run_id) {
          setMlflowInfo({
            run_id: result.mlflow_run_id,
            experiment_id: result.mlflow_experiment_id
          });
        }
        
        // 성능 통계 새로고침
        fetchPerformanceStats();
      } else {
        setUploadStatus('❌ 업로드 실패');
      }
    } catch (error) {
      setUploadStatus(`❌ 오류: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 검색 실행 (MLflow 추적 포함)
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      alert('검색어를 입력하세요!');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          k: 5
        }),
      });

      if (response.ok) {
        const results = await response.json();
        setSearchResults(results);
        
        // 검색 후 성능 통계 업데이트
        fetchPerformanceStats();
      } else {
        alert('검색 실패');
        setSearchResults([]);
      }
    } catch (error) {
      alert(`검색 오류: ${error.message}`);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '1200px', margin: '0 auto' }}>
      
      {/* 헤더 */}
      <div style={{ 
        textAlign: 'center', 
        marginBottom: '40px', 
        padding: '30px', 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '15px',
        color: 'white',
        boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ 
          margin: '0 0 15px 0', 
          fontSize: '2.5em',
          textShadow: '2px 2px 4px rgba(0,0,0,0.3)'
        }}>
          🤖 AI 수사 시스템 + MLflow
        </h1>
        <p style={{ 
          margin: 0, 
          fontSize: '1.2em',
          opacity: 0.9
        }}>
          서버 상태: {serverStatus}
        </p>
      </div>

      {/* MLflow 대시보드 섹션 */}
      <div style={{ 
        marginBottom: '40px', 
        padding: '40px', 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '15px',
        color: 'white'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '25px' }}>
          <div style={{ 
            width: '50px', 
            height: '50px', 
            background: 'rgba(255,255,255,0.2)',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginRight: '15px',
            fontSize: '24px'
          }}>
            📊
          </div>
          <h2 style={{ margin: 0, color: 'white', fontSize: '1.8em' }}>
            MLflow 실험 관리 대시보드
          </h2>
        </div>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '20px',
          marginBottom: '25px'
        }}>
          {/* 현재 실험 정보 */}
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '20px',
            borderRadius: '10px',
            backdropFilter: 'blur(10px)'
          }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'white' }}>🔬 현재 실험</h4>
            {mlflowInfo.run_id ? (
              <div>
                <p style={{ margin: '5px 0', fontSize: '14px' }}>
                  Run ID: {mlflowInfo.run_id.substring(0, 8)}...
                </p>
                <p style={{ margin: '5px 0', fontSize: '14px' }}>
                  Experiment: {mlflowInfo.experiment_id}
                </p>
              </div>
            ) : (
              <p style={{ margin: 0, fontSize: '14px', opacity: 0.8 }}>
                영상을 업로드하면 MLflow 실험이 시작됩니다
              </p>
            )}
          </div>
          
          {/* 처리 통계 */}
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '20px',
            borderRadius: '10px',
            backdropFilter: 'blur(10px)'
          }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'white' }}>📈 처리 통계</h4>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              처리된 영상: {performanceStats.전체_성능?.total_videos_processed || 0}개
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              탐지된 인물: {performanceStats.전체_성능?.total_persons_detected || 0}명
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              평균 신뢰도: {(performanceStats.전체_성능?.average_detection_confidence * 100 || 0).toFixed(1)}%
            </p>
          </div>
          
          {/* AI 모델 상태 */}
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '20px',
            borderRadius: '10px',
            backdropFilter: 'blur(10px)'
          }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'white' }}>🤖 AI 모델 상태</h4>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              YOLO: {performanceStats.시스템_상태?.YOLO_로딩됨 ? '✅ 로딩됨' : '❌ 실패'}
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              CLIP: {performanceStats.시스템_상태?.CLIP_로딩됨 ? '✅ 로딩됨' : '❌ 실패'}
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              디바이스: {performanceStats.시스템_상태?.디바이스 || 'CPU'}
            </p>
          </div>
        </div>
        
        {/* MLflow UI 링크 */}
        <div style={{ textAlign: 'center' }}>
          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '10px',
              padding: '12px 24px',
              background: 'rgba(255,255,255,0.2)',
              color: 'white',
              textDecoration: 'none',
              borderRadius: '8px',
              fontSize: '16px',
              fontWeight: '600',
              transition: 'all 0.3s ease'
            }}
          >
            🔗 MLflow UI에서 상세 실험 결과 보기
          </a>
          <p style={{ margin: '10px 0 0 0', fontSize: '14px', opacity: 0.8 }}>
            모든 실험이 자동으로 추적되고 있습니다
          </p>
        </div>
      </div>

      {/* 업로드 섹션 */}
      <div style={{ 
        marginBottom: '40px', 
        padding: '40px', 
        backgroundColor: 'white', 
        borderRadius: '15px', 
        boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
        border: '1px solid #f0f0f0'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '25px' }}>
          <div style={{ 
            width: '50px', 
            height: '50px', 
            background: 'linear-gradient(45deg, #FF6B6B, #4ECDC4)',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginRight: '15px',
            fontSize: '24px'
          }}>
            📹
          </div>
          <h2 style={{ margin: 0, color: '#333', fontSize: '1.8em' }}>
            CCTV 영상 업로드 (MLflow 추적)
          </h2>
        </div>
        
        <label style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '12px',
          padding: '15px 30px',
          background: 'linear-gradient(45deg, #007bff, #0056b3)',
          color: 'white',
          borderRadius: '10px',
          cursor: 'pointer',
          fontSize: '18px',
          fontWeight: '600',
          transition: 'all 0.3s ease',
          boxShadow: '0 4px 15px rgba(0,123,255,0.3)'
        }}>
          📤 영상 파일 선택 (AI 분석 + MLflow 기록)
          <input
            type="file"
            accept="video/*"
            onChange={handleUpload}
            style={{ display: 'none' }}
          />
        </label>

        {uploadStatus && (
          <div style={{ 
            marginTop: '20px', 
            padding: '20px', 
            backgroundColor: '#f8f9fa', 
            borderRadius: '10px',
            fontSize: '16px',
            border: '1px solid #e9ecef'
          }}>
            {uploadStatus}
            {mlflowInfo.run_id && (
              <p style={{ margin: '10px 0 0 0', fontSize: '14px', color: '#666' }}>
                🔬 MLflow Run ID: {mlflowInfo.run_id.substring(0, 8)}...
              </p>
            )}
          </div>
        )}
      </div>

      {/* 검색 섹션 */}
      <div style={{ 
        marginBottom: '40px', 
        padding: '40px', 
        backgroundColor: 'white', 
        borderRadius: '15px', 
        boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
        border: '1px solid #f0f0f0'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '25px' }}>
          <div style={{ 
            width: '50px', 
            height: '50px', 
            background: 'linear-gradient(45deg, #28a745, #20c997)',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginRight: '15px',
            fontSize: '24px'
          }}>
            🔍
          </div>
          <h2 style={{ margin: 0, color: '#333', fontSize: '1.8em' }}>
            AI 인물 검색 (CLIP + MLflow)
          </h2>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', marginBottom: '25px' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="영어로 검색하세요: person wearing red clothes, man with glasses..."
            style={{
              flex: 1,
              padding: '15px 20px',
              border: '2px solid #e9ecef',
              borderRadius: '10px',
              fontSize: '16px',
              transition: 'border-color 0.3s ease',
              outline: 'none'
            }}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            style={{
              padding: '15px 30px',
              background: loading ? '#6c757d' : 'linear-gradient(45deg, #28a745, #20c997)',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '16px',
              fontWeight: '600',
              transition: 'all 0.3s ease',
              boxShadow: loading ? 'none' : '0 4px 15px rgba(40,167,69,0.3)',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            {loading ? '🔄 검색 중...' : '🔍 AI 검색'}
          </button>
        </div>

        {/* 검색 결과 */}
        {searchResults.length > 0 && (
          <div style={{ marginTop: '30px' }}>
            <h3 style={{ color: '#333', fontSize: '1.4em', marginBottom: '20px' }}>
              🎯 검색 결과 ({searchResults.length}개) - MLflow에 기록됨
            </h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', 
              gap: '25px'
            }}>
              {searchResults.map((result, index) => (
                <div key={index} style={{
                  border: '1px solid #e9ecef',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  backgroundColor: '#fafafa',
                  transition: 'transform 0.3s ease, box-shadow 0.3s ease',
                  cursor: 'pointer'
                }}>
                  {/* 실제 이미지 표시 */}
                  {result.image_base64 ? (
                    <img
                      src={`data:image/jpeg;base64,${result.image_base64}`}
                      alt={result.caption}
                      style={{
                        width: '100%',
                        height: '200px',
                        objectFit: 'cover'
                      }}
                    />
                  ) : (
                    <div style={{
                      width: '100%',
                      height: '200px',
                      background: '#f0f0f0',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#666'
                    }}>
                      📷 실제 AI 탐지 이미지
                    </div>
                  )}
                  
                  <div style={{ padding: '20px' }}>
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      marginBottom: '15px'
                    }}>
                      <div style={{
                        background: 'linear-gradient(45deg, #007bff, #6f42c1)',
                        color: 'white',
                        padding: '8px 16px',
                        borderRadius: '20px',
                        fontSize: '14px',
                        fontWeight: '600'
                      }}>
                        #{index + 1}
                      </div>
                      <div style={{ 
                        background: 'linear-gradient(45deg, #28a745, #20c997)',
                        color: 'white',
                        padding: '6px 12px',
                        borderRadius: '15px',
                        fontSize: '13px',
                        fontWeight: '600'
                      }}>
                        {(result.score * 100).toFixed(1)}% 유사
                      </div>
                    </div>
                    <p style={{ 
                      margin: 0, 
                      fontSize: '15px', 
                      color: '#495057',
                      lineHeight: '1.5'
                    }}>
                      {result.caption}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 사용법 안내 */}
      <div style={{ 
        padding: '30px', 
        background: 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)',
        borderRadius: '15px',
        color: 'white'
      }}>
        <h3 style={{ 
          color: 'white', 
          marginTop: 0, 
          fontSize: '1.5em',
          marginBottom: '20px'
        }}>
          📝 MLflow 통합 AI 시스템 사용법
        </h3>
        <div style={{ fontSize: '16px', lineHeight: '1.8' }}>
          <div style={{ marginBottom: '12px' }}>
            <strong>1.</strong> 영상을 업로드하면 MLflow가 자동으로 실험을 추적합니다
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>2.</strong> YOLO AI가 사람을 탐지하고 모든 성능 데이터를 기록합니다
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>3.</strong> 영어로 검색하면 CLIP AI가 유사한 사람을 찾습니다
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>4.</strong> MLflow UI에서 모든 실험 결과와 성능을 확인하세요
          </div>
          <div>
            <strong>5.</strong> 모든 과정이 전문적으로 관리되고 재현 가능합니다
          </div>
        </div>
      </div>

      {/* 로딩 오버레이 */}
      {loading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          backdropFilter: 'blur(5px)'
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '40px',
            borderRadius: '15px',
            textAlign: 'center',
            boxShadow: '0 20px 40px rgba(0,0,0,0.3)'
          }}>
            <div style={{
              width: '60px',
              height: '60px',
              border: '6px solid #f3f3f3',
              borderTop: '6px solid #007bff',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 25px'
            }}></div>
            <p style={{ 
              margin: 0, 
              fontSize: '20px',
              color: '#333',
              fontWeight: '600'
            }}>
              🤖 AI 분석 중... MLflow 기록 중...
            </p>
          </div>
        </div>
      )}

      {/* CSS 애니메이션 */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default App;