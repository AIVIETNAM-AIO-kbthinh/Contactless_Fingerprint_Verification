#!/usr/bin/env bash
# Cài hoặc cập nhật server fpsys trên VPS Ubuntu 22.04/24.04 (HTTPS qua Caddy + sslip.io).
# Clone repo vào /opt/fpsys rồi chạy: sudo bash deploy/setup.sh
# Chạy lại được nhiều lần; cập nhật: git pull && sudo bash deploy/setup.sh
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "Cần quyền root: sudo bash deploy/setup.sh"; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE=/etc/fpsys.env
DATA=/var/lib/fpsys
PORT=8000
case "$REPO" in
  /root*|/home/*) echo "User fpsys không đọc được $REPO. Clone repo vào /opt/fpsys."; exit 1 ;;
esac

echo "== 1/8 Swap"
# 1 GB RAM không đủ để cài torch và chạy YOLO trên ảnh 12 MP
if [ -z "$(swapon --show --noheadings)" ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== 2/8 Gói hệ thống"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
# libgl1, libglib: opencv-python (ultralytics kéo theo) cần trên Ubuntu server
apt-get install -y -q python3-venv curl gnupg libgl1 debian-keyring debian-archive-keyring \
  apt-transport-https
apt-get install -y -q libglib2.0-0t64 2>/dev/null || apt-get install -y -q libglib2.0-0
if ! command -v caddy >/dev/null; then
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    > /etc/apt/sources.list.d/caddy-stable.list
  chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
    /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q
  apt-get install -y -q caddy
fi

echo "== 3/8 User fpsys"
id fpsys >/dev/null 2>&1 || useradd --system --create-home --home-dir "$DATA" \
  --shell /usr/sbin/nologin fpsys

echo "== 4/8 Python (torch CPU + fpsys[server])"
[ -x "$REPO/.venv/bin/python" ] || python3 -m venv "$REPO/.venv"
PIP="$REPO/.venv/bin/pip"
"$PIP" install -q --no-cache-dir --upgrade pip
"$PIP" install -q --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$PIP" install -q --no-cache-dir -e "$REPO[server]"
install -D -m 644 "$REPO/yolo11n_640/weights/best.pt" "$REPO/models/best.pt"

echo "== 5/8 Khóa bí mật ($ENV_FILE)"
# Chỉ tạo một lần: đổi FPSYS_DB_KEY thì không giải mã được template đã đăng ký
if [ ! -s "$ENV_FILE" ]; then
  tmp="$(mktemp)"
  "$REPO/.venv/bin/python" - > "$tmp" <<'EOF'
import secrets
from cryptography.fernet import Fernet
print("FPSYS_DB_KEY=" + Fernet.generate_key().decode())
print("FPSYS_API_KEY=" + secrets.token_urlsafe(24))
EOF
  install -m 600 "$tmp" "$ENV_FILE"
  rm -f "$tmp"
fi
API_KEY="$(grep '^FPSYS_API_KEY=' "$ENV_FILE" | cut -d= -f2-)"

echo "== 6/8 Dịch vụ systemd fpsys"
cat > /etc/systemd/system/fpsys.service <<EOF
[Unit]
Description=fpsys - xác thực vân tay (FastAPI)
After=network-online.target
Wants=network-online.target

[Service]
User=fpsys
WorkingDirectory=$REPO
EnvironmentFile=$ENV_FILE
ExecStart=$REPO/.venv/bin/python -m server --host 127.0.0.1 --port $PORT --db $DATA/server.db --set enroll.min_consistency=0.08
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q fpsys
systemctl restart fpsys

echo "== 7/8 HTTPS (Caddy + sslip.io)"
IP="$(curl -4 -fsS https://api.ipify.org)"
DOMAIN="${IP//./-}.sslip.io"
CADDYFILE=/etc/caddy/Caddyfile
if [ -f "$CADDYFILE" ] && ! grep -q '^# fpsys' "$CADDYFILE"; then
  cp "$CADDYFILE" "$CADDYFILE.bak"
  echo "   Đã sao lưu Caddyfile cũ: $CADDYFILE.bak"
fi
cat > "$CADDYFILE" <<EOF
# fpsys (tạo bởi deploy/setup.sh)
$DOMAIN {
	request_body {
		max_size 80MB
	}
	reverse_proxy 127.0.0.1:$PORT
}
EOF
systemctl enable -q caddy
systemctl restart caddy
if command -v ufw >/dev/null && ufw status | grep -q 'Status: active'; then
  ufw allow 80/tcp
  ufw allow 443/tcp
fi

echo "== 8/8 Kiểm tra https://$DOMAIN (chờ tối đa ~2 phút để xin chứng chỉ)"
code=000
for _ in $(seq 1 40); do
  code="$(curl -s -o /dev/null -w '%{http_code}' -H "X-API-Key: $API_KEY" "https://$DOMAIN/users" || true)"
  [ "$code" = 200 ] && break
  sleep 3
done

if [ "$code" = 200 ]; then
  echo "OK 200: server đang chạy tại https://$DOMAIN"
else
  echo "LỖI: https://$DOMAIN trả về $code. Kiểm tra:"
  echo "  - Tường lửa của nhà cung cấp VPS đã mở cổng 80 và 443 chưa"
  echo "  - journalctl -u fpsys -n 50     (server)"
  echo "  - journalctl -u caddy -n 50     (HTTPS)"
fi
cat <<EOF

Build APK trên máy có Flutter (thư mục mobile):
  flutter build apk --release --split-per-abi --dart-define=SERVER_URL=https://$DOMAIN --dart-define=API_KEY=$API_KEY

Sao lưu $ENV_FILE và $DATA/server.db: mất FPSYS_DB_KEY là mất toàn bộ dữ liệu đã đăng ký.
EOF
[ "$code" = 200 ]
