import * as vscode from 'vscode';

export interface NtfyConfig {
  serverUrl: string;
  topic: string;
  requirePassword: boolean;
  autoStart: boolean;
  terminalInjectionEnabled: boolean;
  maxMessagesPerMinute: number;
  notifyOnClaudeStop: boolean;
}

/**
 * Manages extension configuration. Sensitive values (keyword, access token)
 * are stored in VS Code's SecretStorage which delegates to the OS keychain
 * (Windows Credential Manager / macOS Keychain / libsecret).
 */
export class ConfigManager {
  private static readonly KEY_KEYWORD      = 'ntfy-listener.keyword';
  private static readonly KEY_ACCESS_TOKEN = 'ntfy-listener.accessToken';
  private static readonly KEY_PUB_TOKEN    = 'ntfy-listener.publishToken';

  constructor(private readonly secrets: vscode.SecretStorage) {}

  get(): NtfyConfig {
    const cfg = vscode.workspace.getConfiguration('ntfyListener');
    return {
      serverUrl:               cfg.get<string> ('serverUrl',              'https://ntfy.sh'),
      topic:                   cfg.get<string> ('topic',                  ''),
      requirePassword:         cfg.get<boolean>('requirePassword',        true),
      autoStart:               cfg.get<boolean>('autoStart',              true),
      terminalInjectionEnabled:cfg.get<boolean>('terminalInjectionEnabled',true),
      maxMessagesPerMinute:    cfg.get<number> ('maxMessagesPerMinute',   10),
      notifyOnClaudeStop:      cfg.get<boolean>('notifyOnClaudeStop',     true),
    };
  }

  async getKeyword():      Promise<string | undefined> { return this.secrets.get(ConfigManager.KEY_KEYWORD); }
  async getAccessToken():  Promise<string | undefined> { return this.secrets.get(ConfigManager.KEY_ACCESS_TOKEN); }
  async getPublishToken(): Promise<string | undefined> { return this.secrets.get(ConfigManager.KEY_PUB_TOKEN); }

  async setKeyword(v: string):      Promise<void> { await this.secrets.store(ConfigManager.KEY_KEYWORD, v); }
  async setAccessToken(v: string):  Promise<void> { await this.secrets.store(ConfigManager.KEY_ACCESS_TOKEN, v); }
  async setPublishToken(v: string): Promise<void> { await this.secrets.store(ConfigManager.KEY_PUB_TOKEN, v); }
  async clearKeyword():             Promise<void> { await this.secrets.delete(ConfigManager.KEY_KEYWORD); }

  /** Interactive configuration wizard. */
  async configure(context: vscode.ExtensionContext): Promise<boolean> {
    // Topic
    const topic = await vscode.window.showInputBox({
      prompt: 'ntfy topic name (use a long random string — acts as a shared secret)',
      value: this.get().topic,
      placeHolder: 'e.g. vscode-abc123-def456-ghi789',
      ignoreFocusOut: true,
    });
    if (topic === undefined) { return false; }
    await vscode.workspace.getConfiguration('ntfyListener')
      .update('topic', topic, vscode.ConfigurationTarget.Global);

    // Keyword password
    const kwChoice = await vscode.window.showQuickPick(
      ['Set keyword password', 'Clear keyword', 'Keep existing'],
      { placeHolder: 'Keyword password (required to execute commands)' }
    );
    if (kwChoice === 'Set keyword password') {
      const kw = await vscode.window.showInputBox({
        prompt: 'Keyword password — messages must start with "keyword: " to run commands',
        password: true,
        ignoreFocusOut: true,
      });
      if (kw) {
        await this.setKeyword(kw);
        vscode.window.showInformationMessage('Keyword saved to OS keychain.');
      }
    } else if (kwChoice === 'Clear keyword') {
      await this.clearKeyword();
      vscode.window.showInformationMessage('Keyword cleared. Extension will operate in read-only mode.');
    }

    // ntfy access token (for subscribing to private/protected topics)
    const tokenChoice = await vscode.window.showQuickPick(
      ['Set subscribe token', 'Set publish token', 'Skip'],
      { placeHolder: 'ntfy access token (optional — for private topics)' }
    );
    if (tokenChoice === 'Set subscribe token') {
      const tok = await vscode.window.showInputBox({ prompt: 'ntfy subscribe/read access token', password: true, ignoreFocusOut: true });
      if (tok) { await this.setAccessToken(tok); }
    } else if (tokenChoice === 'Set publish token') {
      const tok = await vscode.window.showInputBox({ prompt: 'ntfy publish/write access token', password: true, ignoreFocusOut: true });
      if (tok) { await this.setPublishToken(tok); }
    }

    // Require password toggle
    const pwMode = await vscode.window.showQuickPick(
      ['Yes — require keyword for commands (recommended)', 'No — read-only notifications only'],
      { placeHolder: 'Require keyword to execute commands?' }
    );
    if (pwMode) {
      await vscode.workspace.getConfiguration('ntfyListener')
        .update('requirePassword', pwMode.startsWith('Yes'), vscode.ConfigurationTarget.Global);
    }

    vscode.window.showInformationMessage('Ntfy Listener configured. Restart listener to apply changes.');
    return true;
  }
}
