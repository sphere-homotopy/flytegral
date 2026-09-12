export const BUZZ_AUDIO_URL = 'https://upload.wikimedia.org/wikipedia/commons/c/ca/Bombus_buzz.ogg';

export class ThinkingBuzz {
  constructor({ audioFactory = (src) => new Audio(src), volume = 0.18 } = {}) {
    this.audio = audioFactory(BUZZ_AUDIO_URL);
    this.audio.loop = true;
    this.audio.preload = 'auto';
    this.audio.volume = volume;
    this.desiredPlaying = false;
  }

  async start() {
    this.desiredPlaying = true;
    try {
      await this.audio.play();
      return true;
    } catch {
      // Browsers may block autoplay until the first user gesture.
      return false;
    }
  }

  stop() {
    this.desiredPlaying = false;
    this.audio.pause();
    this.audio.currentTime = 0;
  }

  async unlock() {
    if (this.desiredPlaying) return this.start();
    try {
      await this.audio.play();
      this.audio.pause();
      this.audio.currentTime = 0;
      return true;
    } catch {
      return false;
    }
  }
}
