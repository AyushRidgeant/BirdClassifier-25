FROM tensorflow/tensorflow:latest

RUN pip install --ignore-installed tensorflow Flask gunicorn tensorflow-hub tf_keras numpy pillow

COPY . /app
WORKDIR /app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
