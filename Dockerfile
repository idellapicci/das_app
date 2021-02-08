FROM python:3.7
LABEL maintainer "Isabela Piccinini <isabela@picci.com.br>"

USER root

RUN mkdir /app
WORKDIR /app

COPY ./ ./
RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8050

CMD ["python", "app.py"]