sudo iptables -I INPUT -p tcp --dport 80 -j NFQUEUE --queue-num 1
sudo iptables -I OUTPUT -p tcp --sport 80 -j NFQUEUE --queue-num 1
python waf.py
(
After you finish the work remember to disable the request forwarding module
sudo iptables -D INPUT -p tcp --dport 80 -j NFQUEUE --queue-num 1
sudo iptables -D OUTPUT -p tcp --sport 80 -j NFQUEUE --queue-num 1
)

python dashboard.py

python dvwa_log_ingest.py

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

If you want to speed up the unban process manually use this urls in api endpoint
/unban/{ip}
/clear-rate-limit/{ip}

Add in telegram.py:
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
