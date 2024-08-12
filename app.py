import os
import re
import torch
import pymysql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch import nn
from transformers import BertTokenizer, BertModel
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

pymysql.install_as_MySQLdb()

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Set up database connection and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define database model
class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    diary_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    average_depression_percent = Column(Integer, nullable=False)

# Create the table
Base.metadata.create_all(bind=engine)

# BERT 기반 분류 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=11, dropout_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else None
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooler_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        if self.dropout:
            pooler_output = self.dropout(pooler_output)
        return self.classifier(pooler_output)


# 현재 경로에서 모델 파일 로드
model = BERTClassifier(BertModel.from_pretrained('monologg/kobert'))
model_path = './bert_emotion_model.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))

    model.eval()
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BERTClassifier(BertModel.from_pretrained('monologg/kobert'))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 감정 매핑
emotion_mapping = {
    '행복': 0,
    '기쁨': 0,  # '행복'과 '기쁨'을 같은 것으로 간주
    '슬픔': 1,
    '불안': 2,
    '상처': 3,
    '당황': 4,
    '분노': 5,
    '혐오': 6,
    '중립': 7,
    '놀람': 8,
    '공포': 9
}

def preprocess_text(text):
    # ......과 같은 여러 개의 점을 하나로 변환
    text = re.sub(r'\.{2,}', '', text)
    # . 으로 문장을 분리
    sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    return sentences

def detect_emotions(model, tokenizer, texts, max_len=128):
    emotion_labels = ['행복', '슬픔', '불안', '상처', '당황', '분노', '혐오', '중립', '놀람', '공포']
    negative_emotions = ['슬픔', '상처', '혐오', '불안', '분노', '공포']

    # 각 부정적 감정에 대한 기여도 설정 (부정적 감정의 기여도를 높임)
    emotion_contribution = {
        '슬픔': 0.6,  # 기여도 증가
        '상처': 0.5,  # 기여도 증가
        '혐오': 0.4,  # 기여도 증가
        '불안': 0.6,  # 기여도 증가
        '분노': 0.7,  # 기여도 증가
        '공포': 0.5,  # 기여도 증가
        '행복': -0.3  # 행복은 우울도를 감소시키는 방향으로 설정
    }

    results = []
    model.eval()
    with torch.no_grad():
        encodings = tokenizer(texts,
                              add_special_tokens=True,
                              max_length=max_len,
                              padding=True,
                              truncation=True,
                              return_attention_mask=True,
                              return_tensors='pt')

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        outputs = model(input_ids, attention_mask)
        emotion_scores = torch.softmax(outputs, dim=1).cpu().numpy()

        # 각 문장에 대한 결과 계산
        for j, text in enumerate(texts):
            scores = emotion_scores[j]
            predicted_emotions = [emotion_labels[k] for k in range(len(scores)) if scores[k] > 0.3]  # 임계값을 0.3으로 조정

            # 부정적 감정 점수와 기여도를 곱하여 우울도 계산
            depression_percent = 0
            for emo in negative_emotions + ['행복']:  # 행복도 고려
                score = scores[emotion_mapping[emo]]
                contribution = emotion_contribution.get(emo, 0)
                depression_percent += score * contribution

            # 우울도는 최대 100%로 제한
            depression_percent = min(100, max(0, depression_percent * 100))  # 0~1 범위의 점수를 0~100 범위로 변환, 최소값 0 적용
            depression_percent = int(depression_percent)

            results.append({
                'text': text,
                'predicted_emotions': predicted_emotions,
                'depression_percent': depression_percent
            })
    return results


def analyze_text(model, tokenizer, input_text):
    sentences = preprocess_text(input_text)
    num_sentences = len(sentences)
    results = detect_emotions(model, tokenizer, sentences)

    # 각 문장당 결과 출력 및 평균 계산
    total_depression_percent = 0
    for result in results:
        total_depression_percent += result['depression_percent']

    # 평균 우울도 계산
    average_depression_percent = total_depression_percent / num_sentences if num_sentences > 0 else 0
    average_depression_percent = int(average_depression_percent)  # Ensure it's a float

    return average_depression_percent

# FastAPI 설정
app = FastAPI()

class TextInput(BaseModel):
    diary_id: int
    user_id: int
    text: str

@app.post("/analyze/")
def analyze(input: TextInput):
    try:
        average_depression_percent = analyze_text(model, tokenizer, input.text)
        # 데이터베이스에 결과 저장
        db = SessionLocal()
        analysis_result = AnalysisResult(
            average_depression_percent=average_depression_percent,
            diary_id=input.diary_id,
            user_id=input.user_id
        )
        db.add(analysis_result)
        db.commit()
        db.refresh(analysis_result)
        db.close()
        return {"average_depression_percent": average_depression_percent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uvicorn 서버 실행을 위한 엔트리 포인트
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
