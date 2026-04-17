/**
 * Simple sliding-window rate limiter.
 * Prevents a compromised or replayed keyword from flooding the terminal.
 */
export class RateLimiter {
  private timestamps: number[] = [];

  constructor(private readonly maxPerMinute: number) {}

  /** Returns true if the action is allowed under the current rate limit. */
  allow(): boolean {
    const now = Date.now();
    const windowStart = now - 60_000;
    this.timestamps = this.timestamps.filter(t => t > windowStart);
    if (this.timestamps.length >= this.maxPerMinute) {
      return false;
    }
    this.timestamps.push(now);
    return true;
  }

  reset(): void {
    this.timestamps = [];
  }
}
