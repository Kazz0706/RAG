FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# キャッシュ残さずインストール
RUN pip install --no-cache-dir -r requirements.txt

# 現在のディレクトリ内全てのファイルを、WORKDIR内にコピーする
COPY . .

# コンテナが使うポート(8501はstreamlitのデフォルトポート)
EXPOSE 8501

# コンテナ起動時にデフォルトで実行するコマンド
CMD ["streamlit", "run", "mistral.py", "--server.address=0.0.0.0"]