FROM python:3.12-slim
WORKDIR /project
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
ENV PYTHONPATH=/project/src
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY run.py ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/grammar/ ./models/grammar/
COPY data/raw/ ./data/raw/
COPY data/evaluation/ ./data/evaluation/
COPY data/slide_content/ ./data/slide_content/
COPY input/ ./input/
COPY tests/ ./tests/
RUN mkdir -p /project/output /nlp/input /nlp/output
CMD ["python", "-m", "course_assistant.docker_entry"]
