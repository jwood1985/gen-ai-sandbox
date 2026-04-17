# Ntfy VS Code Listener — Setup Guide

## Architecture Overview

```
Phone (ntfy app)
      │
      │  HTTPS / TLS
      ▼
 ntfy.sh (or self-hosted)  ◄──── VS Code extension publishes questions here
      │
      │  SSE stream (HTTPS)
      ▼
 VS Code Extension  ──► sanitize ──► keyword check ──► rate-limit ──► terminal.sendText()
```

The phone and VS Code never communicate directly.  All traffic goes through
the ntfy relay, which means the PC's external attack surface is zero — ntfy
pull-only, no inbound ports opened.

---

## 1. Install the Extension

### From source (development)
```bash
cd ntfy-vscode-listener
npm install
npm run compile
# Press F5 in VS Code to launch Extension Development Host
# Or package it:
npm run package      # produces ntfy-vscode-listener-0.1.0.vsix
code --install-extension ntfy-vscode-listener-0.1.0.vsix
```

The extension activates on every VS Code start-up (`onStartupFinished`),
so it is available in all workspaces automatically.

---

## 2. Configure the Extension

Run **Ntfy: Configure** from the Command Palette (`Ctrl+Shift+P`).

| Setting | Recommendation |
|---|---|
| **Topic** | Use a random UUID: `uuidgen` (Linux/macOS) or `[System.Guid]::NewGuid()` (PowerShell) |
| **Keyword password** | A memorable phrase you won't accidentally type — stored in OS keychain |
| **Server URL** | `https://ntfy.sh` for quick start; self-hosted for maximum security |
| **Access token** | Optional — create one in ntfy.sh account settings for private topics |

Sensitive values (keyword, tokens) are stored via VS Code `SecretStorage`,
which delegates to Windows Credential Manager / macOS Keychain / libsecret.
They are **never** written to `settings.json`.

---

## 3. Phone Setup

1. Install the **ntfy** app ([Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy) · [iOS](https://apps.apple.com/app/ntfy/id1625396347)).
2. Subscribe to the same topic name you entered in the extension.
3. To send a **notification** (read-only): just send any message.
4. To send a **command** (terminal injection): prefix with `keyword: `
   ```
   opensesame: yes
   opensesame: n
   opensesame: my answer to the prompt
   ```

---

## 4. Claude Code Stop Hook

When Claude Code finishes a turn and waits for input, the hook posts a push
notification to your phone so you know to check VS Code.

### Installation
```bash
mkdir -p ~/.claude/hooks ~/.config/ntfy-vscode

# Copy the hook script
cp ntfy-vscode-listener/hooks/stop-notify.sh ~/.claude/hooks/

# Create configuration
cat > ~/.config/ntfy-vscode/config <<'EOF'
NTFY_SERVER=https://ntfy.sh
NTFY_TOPIC=your-topic-here
NTFY_TOKEN=tk_your_token_here   # remove line if using public topic
EOF

chmod 600 ~/.config/ntfy-vscode/config
```

### Register in Claude Code settings

Add to `~/.claude/settings.json` (create if absent):
```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/stop-notify.sh"
          }
        ]
      }
    ]
  }
}
```

---

## 5. WireGuard VPN — Securing the Host PC

The ntfy channel secures the *communication content* (keyword + sanitization).
WireGuard secures the *host PC itself* — preventing anyone from reaching VS Code,
SSH, RDP, or any other local service unless they are on your VPN.

### Why WireGuard?
- Minimal attack surface (single UDP port, no daemon listening on TCP)
- Cryptographically modern (Curve25519, ChaCha20-Poly1305, BLAKE2s)
- Cross-platform: Windows, Android, iOS, Linux, macOS
- Low latency — important for interactive terminal sessions

### Setup (self-hosted VPN server on a VPS)

#### VPS side (Linux)
```bash
# Install WireGuard
apt install wireguard

# Generate server keys
wg genkey | tee /etc/wireguard/server_private.key | wg pubkey > /etc/wireguard/server_public.key

# /etc/wireguard/wg0.conf
[Interface]
Address    = 10.8.0.1/24
ListenPort = 51820
PrivateKey = <server_private_key>
PostUp     = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown   = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# Windows PC peer
[Peer]
PublicKey  = <pc_public_key>
AllowedIPs = 10.8.0.2/32

# Phone peer
[Peer]
PublicKey  = <phone_public_key>
AllowedIPs = 10.8.0.3/32

# Start
systemctl enable --now wg-quick@wg0
```

#### Windows PC
1. Install [WireGuard for Windows](https://www.wireguard.com/install/)
2. Generate keys in the WireGuard UI → "Add empty tunnel"
3. Configure:
   ```ini
   [Interface]
   PrivateKey = <pc_private_key>
   Address    = 10.8.0.2/24
   DNS        = 1.1.1.1

   [Peer]
   PublicKey  = <server_public_key>
   Endpoint   = your-vps-ip:51820
   AllowedIPs = 10.8.0.0/24
   PersistentKeepalive = 25
   ```
4. Click "Activate".

#### Phone (Android/iOS)
- Import a generated config from the VPS into the WireGuard app.

#### Firewall hardening on the PC
```powershell
# Windows Firewall — allow inbound only from VPN subnet
New-NetFirewallRule -DisplayName "Allow VPN only" `
  -Direction Inbound -LocalPort Any -Protocol TCP `
  -RemoteAddress 10.8.0.0/24 -Action Allow

# Block all other inbound TCP (adjust as needed for your environment)
Set-NetFirewallProfile -Profile Public,Private -DefaultInboundAction Block
```

---

## 6. Self-Hosted ntfy (Maximum Security Option)

Running ntfy on your VPS behind WireGuard means even the relay is private.

```bash
# On VPS
docker run -p 10.8.0.1:80:80 \
  -v /var/cache/ntfy:/var/cache/ntfy \
  -v /etc/ntfy:/etc/ntfy \
  binwiederhier/ntfy serve \
    --cache-file /var/cache/ntfy/cache.db \
    --auth-file /var/cache/ntfy/auth.db \
    --auth-default-access deny-all
```

Then set the extension's server URL to `http://10.8.0.1` and use ntfy access
tokens for both subscribe and publish.

With this topology:
- ntfy is unreachable from the internet (VPN-only)
- Both the relay and the PC require VPN access
- Traffic never leaves your infrastructure

---

## 7. Encryption Summary

| Layer | Mechanism |
|---|---|
| **Transit** | TLS 1.3 (ntfy.sh HTTPS) or WireGuard ChaCha20 (self-hosted on VPN) |
| **At rest — secrets** | OS keychain via VS Code SecretStorage (Windows Credential Manager) |
| **Application auth** | Keyword prefix + rate limiter + input sanitization |
| **Topic obscurity** | Random UUID topic name — acts as a bearer token |
| **Network isolation** | WireGuard VPN — PC unreachable without VPN credentials |

---

## 8. Security Checklist

- [ ] Topic name is a random UUID (not a dictionary word)
- [ ] Keyword password is set and stored in OS keychain
- [ ] `requirePassword = true` in extension settings
- [ ] ntfy access token set for private topic (ntfy.sh account)
- [ ] WireGuard VPN active on PC and phone
- [ ] Windows Firewall blocks inbound connections not from VPN subnet
- [ ] Hook script config file has `chmod 600` permissions
- [ ] Reviewed output channel logs for unexpected blocked messages
