## LLM 기반 챗봇 서비스
- Langchain과 Streamlit, LLM을 이용해 문서 분석, 퀴즈 생성, 웹사이트 Q&A, 오디오·비디오 대본 분석등의 기능들을 구현했습니다.

## 기술 스택
- Python (3.11 권장)
- Streamlit
- LangChain
- OpenAI APIs 또는 로컬 LLM(Ollama 등)
- HuggingFace
- 기타 AI/ML 라이브러리(yt_dlp, pydub, ffmpeg 등)

## 주요 기능
### [DocumentGPT](https://velog.io/@kitkat/LangChain%EA%B3%BC-ChatGPT%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%AC%B8%EC%84%9C-QA-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0)
> 업로드한 PDF, 텍스트, Word 문서를 기반으로 사용자의 질문에 관련된 답변을 생성해주는 서비스
1. 업로드한 문서를 임베딩 & VectorStore 저장
2. MMR 검색 알고리즘을 사용해 연관성과 다양성 모두 고려
3. 실시간 답변 스트리밍(CallbackHandler)
4. [배포버전](https://gpt-challenge-cnojcfovivvohk6ed6o7yq.streamlit.app/)
### PrivateGPT
> 로컬 기반의 LLM과 자체 임베딩을 활용하여 로컬 환경에서도 Q&A가 가능하도록 구현
1. HuggingFaceEmbeddings를 통해 상황에 따른 적절한 임베딩 모델 사용 및 임베딩 캐싱
2. ChatOllama를 통해 원하는 로컬 모델을 지정하여 사용 가능
### QuizGPT
> 문서나 위키피디아 내용을 기반으로 5문항의 객관식 퀴즈를 생성해주는 서비스
1. Output parsers를 사용하여 일관된 형태의 문제 제공(아래의 배포 버전에서는 Function Calling을 사용해 미리 정의된 JSON 스키마에 맞춰 응답 생성)
2. Streamlit의 st.session_state를 사용해 캐싱 구현
3. [배포버전](https://gpt-challenge-izyrfdqdbcsba4dyqrztcc.streamlit.app/) ([챌린지 3기 우수작 선정](https://nomadcoders.co/community/thread/10769))

### SiteGPT
> 특정 사이트 맵(XML 형태의 사이트맵 URL 필요)을 로드하여 해당 사이트에 대한 Q&A를 제공하는 서비스
1. SitemapLoader를 통해 웹페이지 텍스트 크롤링
2. 검색 단계에서 LLM이 답변에 점수를 부여한 뒤 이를 바탕으로 최종 응답 생성
### [AudioGPT](https://velog.io/@kitkat/AudioGPT)
> 오디오(또는 영상)에서 텍스트 대본을 추출한 뒤, 이를 기반으로 내용 요약과 Q&A를 제공하는 서비스
1. 파일 업로드 및 YouTube 링크 지원
2. Whisper API를 이용한 STT
3. LangChain의 ConversationBufferMemory를 통해 이전 대화 맥락을 저장

## 설치 및 실행
1. 저장소 클론
```
  git clone https://github.com/Czerny40/PROJECT-GPT.git
  cd PROJECT-GPT
```
2. Python 가상환경 및 의존성 설치
```
  python -m venv ./env
  env\Scripts\activate
  pip install -r requirements.txt
```
3. .env 파일 작성
- PrivateGPT를 제외한 대부분의 서비스에서 OpenAI의 API 키가 필요할 수 있습니다.
- 예:
```
  OPENAI_API_KEY=YOUR_OPENAI_KEY
  HUGGINGFACEHUB_API_TOKEN=YOUR_HF_TOKEN
```
4. 앱 실행
```
 streamlit run Home.py
```

## Third-Party Libraries License
> - [LangChain](https://github.com/langchain-ai/langchain/blob/master/LICENSE)
> - [yt_dlp](https://github.com/yt-dlp/yt-dlp/blob/master/LICENSE)
> - [FFmpeg](https://ffmpeg.org/legal.html)
- 본 프로젝트는 위 라이브러리들을 포함/사용하며, 각 라이브러리의 라이선스를 준수합니다.
