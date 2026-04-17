import * as https from 'https';
import * as http from 'http';
import { URL } from 'url';
import { EventEmitter } from 'events';

export interface NtfyMessage {
  id: string;
  time: number;
  event: string;
  topic: string;
  message: string;
  title?: string;
  priority?: number;
  tags?: string[];
}

export interface PublishOptions {
  message: string;
  title?: string;
  priority?: 1 | 2 | 3 | 4 | 5;
  tags?: string[];
  token?: string;
}

/**
 * Lightweight ntfy.sh SSE subscriber and HTTP publisher.
 *
 * Uses only Node.js built-ins (https/http) — no runtime dependencies.
 * Implements exponential backoff reconnection capped at 30 seconds.
 */
export class NtfyClient extends EventEmitter {
  private request: http.ClientRequest | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private reconnectDelay = 2000;
  private shouldReconnect = false;
  private lastEventId: string | null = null;
  private sseBuffer = '';
  private sseFields: Record<string, string> = {};

  constructor(
    private readonly serverUrl: string,
    private readonly topic: string,
    private readonly accessToken?: string,
  ) {
    super();
  }

  start(): void {
    this.shouldReconnect = true;
    this.reconnectDelay = 2000;
    this.connect();
  }

  stop(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null; }
    if (this.request)        { this.request.destroy(); this.request = null; }
    this.emit('disconnected');
  }

  // ── Subscriber ────────────────────────────────────────────────────────────

  private connect(): void {
    if (!this.topic) {
      this.emit('error', new Error('No topic configured'));
      return;
    }

    const sseUrl = new URL(`${this.serverUrl}/${encodeURIComponent(this.topic)}/sse`);
    // Resume from the last seen event so we don't re-process old messages
    if (this.lastEventId) {
      sseUrl.searchParams.set('since', this.lastEventId);
    } else {
      // Only new messages from the moment we connect
      sseUrl.searchParams.set('since', 'now');
    }

    const headers: Record<string, string> = {
      'Accept':        'text/event-stream',
      'Cache-Control': 'no-cache',
      'User-Agent':    'ntfy-vscode-listener/0.1',
    };
    if (this.accessToken) {
      headers['Authorization'] = `Bearer ${this.accessToken}`;
    }

    const lib = sseUrl.protocol === 'https:' ? https : http;

    this.request = lib.get(sseUrl.toString(), { headers }, (res) => {
      if (res.statusCode !== 200) {
        this.emit('error', new Error(`HTTP ${res.statusCode} ${res.statusMessage}`));
        res.resume();
        this.scheduleReconnect();
        return;
      }

      this.reconnectDelay = 2000;
      this.emit('connected');

      res.setEncoding('utf8');
      res.on('data', (chunk: string) => {
        this.sseBuffer += chunk;
        const lines = this.sseBuffer.split('\n');
        this.sseBuffer = lines.pop() ?? '';
        this.parseSSELines(lines);
      });
      res.on('end', () => {
        this.emit('disconnected');
        if (this.shouldReconnect) { this.scheduleReconnect(); }
      });
      res.on('error', (err) => {
        this.emit('error', err);
        if (this.shouldReconnect) { this.scheduleReconnect(); }
      });
    });

    this.request.on('error', (err) => {
      this.emit('error', err);
      if (this.shouldReconnect) { this.scheduleReconnect(); }
    });
  }

  private parseSSELines(lines: string[]): void {
    for (const line of lines) {
      if (line.startsWith(':')) { continue; } // SSE keep-alive comment

      if (line === '') {
        // Blank line = event boundary: dispatch accumulated fields
        const data = this.sseFields['data'];
        if (data) {
          try {
            const msg = JSON.parse(data) as NtfyMessage;
            if (msg.event === 'message') {
              this.lastEventId = msg.id;
              this.emit('message', msg);
            }
          } catch {
            // Silently ignore malformed JSON
          }
        }
        this.sseFields = {};
        continue;
      }

      const colon = line.indexOf(':');
      if (colon > 0) {
        const field = line.slice(0, colon).trim();
        const value = line.slice(colon + 1).trimStart();
        this.sseFields[field] = value;
      }
    }
  }

  private scheduleReconnect(): void {
    if (!this.shouldReconnect) { return; }
    const delay = this.reconnectDelay;
    this.reconnectDelay = Math.min(delay * 2, 30_000);
    this.emit('reconnecting', delay);
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  // ── Publisher ─────────────────────────────────────────────────────────────

  /**
   * POST a message back to the same topic (VS Code → phone direction).
   * Returns a promise that resolves to the HTTP status code.
   */
  publish(opts: PublishOptions): Promise<number> {
    return new Promise((resolve, reject) => {
      const pubUrl = new URL(`${this.serverUrl}/${encodeURIComponent(this.topic)}`);
      const body = JSON.stringify({
        message:  opts.message,
        title:    opts.title   ?? 'VS Code',
        priority: opts.priority ?? 3,
        tags:     opts.tags    ?? ['vscode'],
      });
      const headers: Record<string, string> = {
        'Content-Type':   'application/json',
        'Content-Length': Buffer.byteLength(body).toString(),
        'User-Agent':     'ntfy-vscode-listener/0.1',
      };
      if (opts.token) {
        headers['Authorization'] = `Bearer ${opts.token}`;
      }

      const lib = pubUrl.protocol === 'https:' ? https : http;
      const req = lib.request(
        { method: 'POST', hostname: pubUrl.hostname, port: pubUrl.port, path: pubUrl.pathname + pubUrl.search, headers },
        (res) => { res.resume(); resolve(res.statusCode ?? 0); }
      );
      req.on('error', reject);
      req.write(body);
      req.end();
    });
  }
}
