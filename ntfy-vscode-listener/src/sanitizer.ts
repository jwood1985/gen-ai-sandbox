/**
 * Input sanitization for ntfy messages.
 *
 * Defence-in-depth: even if the keyword is compromised, injected shell
 * metacharacters or Unicode tricks cannot escape into the terminal.
 */

const MAX_MESSAGE_LENGTH = 512;

// Patterns that must never appear in a message destined for terminal injection.
// These cover shell metacharacters, script injection, control chars, and
// Unicode direction-override characters used in visual-spoofing attacks.
const BLOCKED_PATTERNS: RegExp[] = [
  /[;&|`$\\]/,                   // Shell metacharacters
  /<script/i,                    // Script injection attempt
  /javascript:/i,                // JS protocol
  /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/, // Non-printable control chars
  /\u202e|\u202d|\u202c|\u202b|\u202a/, // Bidi override/embedding chars
  /\u2028|\u2029/,               // Unicode line/paragraph separators
  /\u0000/,                      // Null byte
  /%00|%0[aAdD]/,                // URL-encoded newline / null
];

export interface SanitizationResult {
  safe: boolean;
  sanitized: string;
  reason?: string;
}

export function sanitize(input: string): SanitizationResult {
  if (typeof input !== 'string' || input.length === 0) {
    return { safe: false, sanitized: '', reason: 'Empty or non-string input' };
  }

  if (input.length > MAX_MESSAGE_LENGTH) {
    return { safe: false, sanitized: '', reason: `Exceeds max length (${MAX_MESSAGE_LENGTH})` };
  }

  for (const pattern of BLOCKED_PATTERNS) {
    if (pattern.test(input)) {
      return { safe: false, sanitized: '', reason: `Blocked pattern: ${pattern.toString()}` };
    }
  }

  // Collapse runs of whitespace and trim edges
  const sanitized = input.replace(/\s+/g, ' ').trim();

  return { safe: true, sanitized };
}

/**
 * Attempts to extract a command from a message that leads with the keyword.
 * Returns null if the keyword is not present.
 * Comparison is case-insensitive to be forgiving of mobile autocorrect.
 */
export function extractCommand(message: string, keyword: string): string | null {
  const prefix = `${keyword}:`;
  if (message.toLowerCase().startsWith(prefix.toLowerCase())) {
    return message.slice(prefix.length).trim();
  }
  return null;
}
