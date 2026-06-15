FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . /app

RUN apt update -y && apt install awscli -y && chmod +x /app/start.sh

RUN pip install -r requirements.txt

ENV PORT=8080
ENV APP_MODE=streamlit

EXPOSE ${PORT}

CMD ["/app/start.sh"]
