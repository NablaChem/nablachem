FROM python:3.10-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
EXPOSE 80
COPY config.toml /root/.streamlit/config.toml
ENTRYPOINT ["streamlit", "run", "app/app.py", "--logger.level=debug"]
