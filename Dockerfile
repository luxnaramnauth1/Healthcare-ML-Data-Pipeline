FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline ./pipeline

# Writes the model, reports and figures to /app/output and /app/docs/images
CMD ["python", "-m", "pipeline.run_pipeline"]
