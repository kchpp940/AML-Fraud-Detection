FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

ENV PORT=8080
ENV APP_MODE=streamlit

EXPOSE ${PORT}

CMD if [ "$APP_MODE" = "flask" ]; then \
      python app.py; \
    else \
      streamlit run app_streamlit.py --server.port=${PORT} --server.address=0.0.0.0; \
    fi
