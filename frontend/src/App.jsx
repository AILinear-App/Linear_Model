import React, { useState, useEffect } from 'react';
import './App.css';

// 환경변수에서 API URL 가져오기
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

function App() {
  // 기본 상태 관리
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [serverStatus, setServerStatus] = useState('확인 중...');
  
  // Render 최적화 상태
  const [renderStats, setRenderStats] = useState({});

  // 서버 상태 확인
  useEffect(() => {
    checkServer();
    fetchRenderStats();
  }, []);

  const checkServer = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        const data = await response.json();
        setServerStatus(`✅ Render AI 서버 연결됨 (${data.platform})`);
      } else {
        setServerStatus('❌ 서버 응답 없음');
      }
    } catch (error) {
      setServerStatus('❌ 서버 연결 실패');
    }
  };

  // Render 통계 가져오기
  const fetchRenderStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats`);
      if (response.ok) {
        const data = await response.json();
        setRenderStats(data);
      }
    } catch (error) {
      console.log('통계 가져오기 실패:', error);
    }
  };

  // 파일 업로드 (Render 최적화)
  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // 파일 크기 체크 (50MB 제한)
    if (file.size > 50 * 1024 * 1024) {
      alert('파일 크기가 50MB를 초과합니다. Render 최적화를 위해 더 작은 파일을 선택해주세요.');
      return;
    }

    setLoading(true);
    setUploadStatus('📤 Render AI 서버에 업로드 중...');

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
        
        // 통계 새로고침
        fetchRenderStats();
      } else {
        const error = await response.json();
        setUploadStatus(`❌ 업로드 실패: ${error.detail}`);
      }
    } catch (error) {
      setUploadStatus(`❌ 오류: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 검색 실행 (Render 최적화)
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
          k: 3  // Render 최적화를 위해 3개로 제한
        }),
      });

      if (response.ok) {
        const results = await response.json();
        setSearchResults(results);
        
        // 검색 후 통계 업데이트
        fetchRenderStats();
      } else {
        const error = await response.json();
        alert(`검색 실패: ${error.detail}`);
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
          🚀 Render AI 수사 시스템
        </h1>
        <p style={{ 
          margin: 0, 
          fontSize: '1.2em',
          opacity: 0.9
        }}>
          서버 상태: {serverStatus}
        </p>
        <p style={{ 
          margin: '10px 0 0 0', 
          fontSize: '1em',
          opacity: 0.8
        }}>
          💪 7달러로 클라우드 AI 서버 구축 성공!
        </p>
      </div>

      {/* Render 대시보드 섹션 */}
      <div style={{ 
        marginBottom: '40px', 
        padding: '40px', 
        background: 'linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%)',
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
            🔥
          </div>
          <h2 style={{ margin: 0, color: 'white', fontSize: '1.8em' }}>
            Render 클라우드 AI 대시보드
          </h2>
        </div>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '20px',
          marginBottom: '25px'
        }}>
          {/* 서버 상태 */}
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '20px',
            borderRadius: '10px',
            backdropFilter: 'blur(10px)'
          }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'white' }}>🌐 서버 상태</h4>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              플랫폼: {renderStats.플랫폼 || 'Render'}
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              메모리 모드: {renderStats.메모리_모드 || '최적화됨'}
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              분석된 데이터: {renderStats.분석된_데이터 || 0}개
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
              YOLO: {renderStats.모델_로딩?.YOLO ? '✅ 로딩됨' : '⏳ 대기중'}
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              CLIP: {renderStats.모델_로딩?.CLIP ? '✅ 로딩됨' : '⏳ 대기중'}
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              디바이스: CPU (Render 최적화)
            </p>
          </div>
          
          {/* 최적화 설정 */}
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '20px',
            borderRadius: '10px',
            backdropFilter: 'blur(10px)'
          }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'white' }}>⚡ 최적화 설정</h4>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              최대 파일크기: 50MB
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              최대 프레임: 10개
            </p>
            <p style={{ margin: '5px 0', fontSize: '14px' }}>
              검색 결과: 3개
            </p>
          </div>
        </div>
        
        {/* 마지막 분석 정보 */}
        {renderStats.마지막_분석 && (
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <p style={{ margin: 0, fontSize: '16px', fontWeight: '600' }}>
              🕒 마지막 분석: {new Date(renderStats.마지막_분석).toLocaleString('ko-KR')}
            </p>
          </div>
        )}
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
            CCTV 영상 업로드 (Render AI 분석)
          </h2>
        </div>
        
        <div style={{ 
          background: '#fff3cd', 
          border: '1px solid #ffeaa7', 
          borderRadius: '8px', 
          padding: '15px', 
          marginBottom: '20px',
          fontSize: '14px',
          color: '#856404'
        }}>
          ⚠️ <strong>Render 최적화 안내:</strong> 파일 크기는 50MB 이하, 첫 분석 시 모델 로딩으로 1-2분 소요될 수 있습니다.
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
          📤 영상 파일 선택 (최대 50MB)
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
            AI 인물 검색 (Render CLIP)
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
              🎯 검색 결과 ({searchResults.length}개) - Render AI 분석
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
                      📷 Render AI 탐지 이미지
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
          🚀 Render 클라우드 AI 시스템 사용법
        </h3>
        <div style={{ fontSize: '16px', lineHeight: '1.8' }}>
          <div style={{ marginBottom: '12px' }}>
            <strong>1.</strong> 50MB 이하의 영상을 업로드하세요 (Render 최적화)
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>2.</strong> 첫 분석 시 AI 모델 로딩으로 1-2분 소요됩니다
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>3.</strong> YOLO AI가 사람을 탐지하고 CLIP AI가 분석합니다
          </div>
          <div style={{ marginBottom: '12px' }}>
            <strong>4.</strong> 영어로 검색하면 유사한 사람을 찾아줍니다
          </div>
          <div>
            <strong>5.</strong> 모든 처리가 클라우드에서 실행됩니다 (7달러의 가치!)
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
              🚀 Render AI 분석 중...
            </p>
            <p style={{ 
              margin: '10px 0 0 0', 
              fontSize: '14px',
              color: '#666'
            }}>
              클라우드에서 AI 모델이 열심히 작업 중입니다
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