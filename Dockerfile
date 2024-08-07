FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .
COPY app.py .

# 필요한 패키지 설치
RUN pip install —no-cache-dir -r requirements.txt

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "app.py"]
