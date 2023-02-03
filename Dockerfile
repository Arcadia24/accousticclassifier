FROM python:latest

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN export PATH="$HOME/.local/bin:$PATH" 

COPY . .

CMD ["streamlit", "--run","Home.py"]