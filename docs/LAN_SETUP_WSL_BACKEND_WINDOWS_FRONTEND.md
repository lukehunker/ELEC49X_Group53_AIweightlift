# LAN Setup Guide (Backend in WSL, Frontend on Windows)

Use this guide when one teammate hosts the backend in WSL and other teammates run the frontend on Windows machines over the same LAN.

## Network Goal

- Backend API runs in WSL on host PC.
- Windows on host PC forwards port `8000` to WSL.
- Teammates point frontend to host PC LAN IP.

---

## 1) Start backend in WSL (host PC)

In WSL terminal on host machine:

```bash
cd /home/lukehunker/ELEC49X_Group53_AIweightlift
source /home/lukehunker/venv_weightlift/bin/activate
cd src
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Keep this terminal running.

---

## 2) Get WSL IP (host PC)

In WSL terminal:

```bash
hostname -I
```

Copy the first IPv4 value (example: `172.28.209.15`).

---

## 3) Forward Windows port 8000 to WSL (host PC)

Open **Windows PowerShell as Administrator** and run:

```powershell
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8000
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=<WSL_IP> connectport=8000
```

Replace `<WSL_IP>` with the value from Step 2.

Check rule:

```powershell
netsh interface portproxy show all
```

---

## 4) Allow Windows firewall inbound TCP/8000 (host PC)

In **Windows PowerShell as Administrator**:

```powershell
New-NetFirewallRule -DisplayName "AI Weightlift API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

---

## 5) Get host Windows LAN IP (host PC)

In Windows Command Prompt or PowerShell:

```powershell
ipconfig
```

Use the IPv4 address for your active network adapter (example: `192.168.1.42`).

This is the address teammates will use.

---

## 6) Teammates run frontend pointing to host API

On teammate machine, from `frontend/`:

```bash
flutter run --dart-define=API_BASE_URL=http://<HOST_WINDOWS_LAN_IP>:8000
```

Replace `<HOST_WINDOWS_LAN_IP>` with host address from Step 5.

---

## 7) Verify connectivity

From teammate machine:

```bash
curl http://<HOST_WINDOWS_LAN_IP>:8000/health
```

Expected: JSON response with status and model info.

---

## Troubleshooting

- If teammate cannot connect:
  - Confirm backend terminal in WSL is still running.
  - Confirm firewall rule exists and is enabled.
  - Confirm both devices are on same network/VPN.
  - Re-run portproxy setup (WSL IP changes after restart).

- If API works locally but not from teammates:
  - Check `netsh interface portproxy show all` output.
  - Ensure `listenaddress=0.0.0.0` and `listenport=8000` are set.

- If WSL was restarted:
  - Get new WSL IP with `hostname -I`.
  - Recreate portproxy rule with new `connectaddress`.

---

## One-time vs recurring setup

- Usually one-time:
  - Firewall rule
- Usually recurring (after reboot/WSL restart):
  - Portproxy update with current WSL IP
