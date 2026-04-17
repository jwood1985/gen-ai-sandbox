import * as vscode from 'vscode';
import { NtfyClient, NtfyMessage } from './ntfyClient';
import { sanitize, extractCommand } from './sanitizer';
import { ConfigManager } from './config';
import { RateLimiter } from './rateLimiter';

let client: NtfyClient | null = null;
let statusBarItem: vscode.StatusBarItem;
let configManager: ConfigManager;
let outputChannel: vscode.LogOutputChannel;
let rateLimiter: RateLimiter;

export function activate(context: vscode.ExtensionContext): void {
  configManager = new ConfigManager(context.secrets);
  outputChannel = vscode.window.createOutputChannel('Ntfy Listener', { log: true }) as vscode.LogOutputChannel;
  rateLimiter   = new RateLimiter(configManager.get().maxMessagesPerMinute);

  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = 'ntfy-listener.status';
  setStatus('idle');
  statusBarItem.show();

  context.subscriptions.push(
    vscode.commands.registerCommand('ntfy-listener.start',       ()  => startListener(context)),
    vscode.commands.registerCommand('ntfy-listener.stop',        ()  => stopListener()),
    vscode.commands.registerCommand('ntfy-listener.configure',   ()  => configManager.configure(context).then(ok => { if (ok) { startListener(context); } })),
    vscode.commands.registerCommand('ntfy-listener.status',      ()  => showStatus()),
    vscode.commands.registerCommand('ntfy-listener.sendToPhone', ()  => sendToPhone()),
    statusBarItem,
    outputChannel,
  );

  // Restart automatically when settings change
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(e => {
      if (e.affectsConfiguration('ntfyListener')) {
        rateLimiter = new RateLimiter(configManager.get().maxMessagesPerMinute);
        if (client) { stopListener(); startListener(context); }
      }
    })
  );

  const cfg = configManager.get();
  if (cfg.autoStart && cfg.topic) {
    startListener(context);
  } else if (cfg.autoStart && !cfg.topic) {
    outputChannel.info('Ntfy Listener: no topic configured. Run "Ntfy: Configure" to set up.');
  }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

async function startListener(context: vscode.ExtensionContext): Promise<void> {
  const cfg = configManager.get();

  if (!cfg.topic) {
    const action = await vscode.window.showWarningMessage(
      'Ntfy Listener: No topic configured.', 'Configure Now'
    );
    if (action) { await configManager.configure(context); }
    return;
  }

  stopListener();

  const [accessToken] = await Promise.all([configManager.getAccessToken()]);
  client = new NtfyClient(cfg.serverUrl, cfg.topic, accessToken ?? undefined);

  client.on('connected',    ()          => { setStatus('connected', cfg.topic);  outputChannel.info(`Connected to topic: ${cfg.topic}`); });
  client.on('disconnected', ()          => { setStatus('idle');                  outputChannel.info('Disconnected.'); });
  client.on('reconnecting', (ms:number) => { setStatus('reconnecting');           outputChannel.warn(`Reconnecting in ${ms}ms…`); });
  client.on('error',        (e:Error)   => { outputChannel.error(`Error: ${e.message}`); });
  client.on('message',      (m:NtfyMessage) => handleIncoming(m));

  client.start();
}

function stopListener(): void {
  client?.stop();
  client = null;
  setStatus('idle');
  outputChannel.info('Listener stopped.');
}

// ── Incoming message handler ─────────────────────────────────────────────────

async function handleIncoming(msg: NtfyMessage): Promise<void> {
  const cfg = configManager.get();
  outputChannel.info(`Received [${msg.id}] "${msg.message}"`);

  // Step 1 — sanitize raw message
  const s1 = sanitize(msg.message);
  if (!s1.safe) {
    outputChannel.warn(`BLOCKED: ${s1.reason}`);
    vscode.window.showWarningMessage(`Ntfy: blocked message — ${s1.reason}`);
    return;
  }

  // Step 2 — if commands are disabled, treat everything as read-only
  if (!cfg.requirePassword) {
    vscode.window.showInformationMessage(`Ntfy: ${s1.sanitized}`);
    return;
  }

  const keyword = await configManager.getKeyword();
  if (!keyword) {
    // No keyword stored → read-only mode
    vscode.window.showInformationMessage(`Ntfy (read-only): ${s1.sanitized}`);
    return;
  }

  const extracted = extractCommand(s1.sanitized, keyword);
  if (extracted === null) {
    // Notification without the keyword prefix — display only
    vscode.window.showInformationMessage(`Ntfy: ${s1.sanitized}`);
    return;
  }

  // Step 3 — sanitize the extracted command a second time
  const s2 = sanitize(extracted);
  if (!s2.safe) {
    outputChannel.warn(`BLOCKED command after extraction: ${s2.reason}`);
    vscode.window.showWarningMessage(`Ntfy: blocked command — ${s2.reason}`);
    return;
  }

  // Step 4 — rate-limit check
  if (!rateLimiter.allow()) {
    outputChannel.warn('Rate limit exceeded — command dropped.');
    vscode.window.showWarningMessage('Ntfy: rate limit exceeded. Command ignored.');
    return;
  }

  outputChannel.info(`Executing command: "${s2.sanitized}"`);
  await injectIntoTerminal(s2.sanitized, cfg.terminalInjectionEnabled);
}

async function injectIntoTerminal(text: string, enabled: boolean): Promise<void> {
  if (!enabled) {
    vscode.window.showInformationMessage(`Ntfy command (injection disabled): ${text}`, 'Copy')
      .then(a => { if (a === 'Copy') { vscode.env.clipboard.writeText(text); } });
    return;
  }

  const terminal = vscode.window.activeTerminal ?? vscode.window.terminals[0];
  if (!terminal) {
    vscode.window.showWarningMessage('Ntfy: no active terminal to inject into.');
    return;
  }

  terminal.show(false);             // reveal without stealing keyboard focus
  terminal.sendText(text, true);    // append \n — confirms the prompt
  outputChannel.info(`Injected into terminal "${terminal.name}": ${text}`);
  vscode.window.showInformationMessage(`Ntfy → terminal: ${text}`);
}

// ── Phone → VS Code publisher ────────────────────────────────────────────────

async function sendToPhone(): Promise<void> {
  const cfg = configManager.get();
  if (!cfg.topic || !client) {
    vscode.window.showWarningMessage('Ntfy Listener is not connected.');
    return;
  }

  const message = await vscode.window.showInputBox({
    prompt: 'Message to send to your phone',
    placeHolder: 'Claude is waiting for your answer…',
  });
  if (!message) { return; }

  const pubToken = await configManager.getPublishToken();
  try {
    const status = await client.publish({
      message,
      title: 'VS Code',
      priority: 3,
      tags: ['vscode', 'bell'],
      token: pubToken,
    });
    if (status >= 200 && status < 300) {
      outputChannel.info(`Published to phone: "${message}"`);
    } else {
      vscode.window.showWarningMessage(`Ntfy publish failed with HTTP ${status}`);
    }
  } catch (e) {
    vscode.window.showErrorMessage(`Ntfy publish error: ${(e as Error).message}`);
  }
}

/**
 * Called by the Claude Code Stop hook (via a shell helper) to notify the phone
 * that Claude has paused and is awaiting input.
 *
 * Exposed as an extension API so the companion hook script can call it via
 * `code --command ntfy-listener.sendToPhone` without needing a separate process.
 */
export async function notifyClaudeStop(context: string): Promise<void> {
  const cfg = configManager.get();
  if (!cfg.notifyOnClaudeStop || !client || !cfg.topic) { return; }

  const pubToken = await configManager.getPublishToken();
  await client.publish({
    message: `Claude Code has stopped and awaits your input.\n\n${context}`.trim(),
    title: 'Claude Code — Input Required',
    priority: 4,
    tags: ['claude', 'bell', 'warning'],
    token: pubToken,
  });
  outputChannel.info('Sent Claude-stop notification to phone.');
}

// ── Status helpers ───────────────────────────────────────────────────────────

function setStatus(state: 'idle' | 'connected' | 'reconnecting', topic?: string): void {
  const icons = { idle: '$(circle-slash)', connected: '$(radio-tower)', reconnecting: '$(sync~spin)' } as const;
  statusBarItem.text = `${icons[state]} Ntfy${topic ? ':' + topic.slice(0, 16) : ''}`;
  statusBarItem.tooltip = `Ntfy Listener — ${state}`;
  statusBarItem.backgroundColor = state === 'connected'
    ? undefined
    : new vscode.ThemeColor('statusBarItem.warningBackground');
}

function showStatus(): void {
  const cfg = configManager.get();
  outputChannel.show(true);
  outputChannel.info(`State: ${client ? 'running' : 'stopped'} | Server: ${cfg.serverUrl} | Topic: ${cfg.topic || '(none)'}`);
}

export function deactivate(): void {
  stopListener();
}
