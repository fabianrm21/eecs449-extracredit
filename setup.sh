sudo apt update
sudo apt upgrade
sudo apt install python3.12-venv
python3 -m venv ./.env
source .env/bin/activate

pip install -r requirements.txt
ollama run llama3.2
ollama pull deepseek
ollama pull qwen-vl
ollama pull whisper
pip install --upgrade ollama langchain
pip install -U langchain-ollama


