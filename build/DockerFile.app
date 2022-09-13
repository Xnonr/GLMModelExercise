FROM xnonr/sklearn:0.1.0

WORKDIR /app

COPY predicting /app/predicting
COPY models /app/models

CMD ["uvicorn", "predicting.main:app", "--host", "0.0.0.0", "--port", "1313"]