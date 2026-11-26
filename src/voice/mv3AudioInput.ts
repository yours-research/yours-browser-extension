import { AUDIO_WORKLET_CAPTURE_PROCESSOR_NAME } from '@yours/voice/audio';
import type { AudioInputSource } from '@yours/voice/contracts';

export interface Mv3AudioWorkletInputOptions {
  workletUrl?: string;
  deviceId?: string;
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
  sampleRate?: number;
}

function resolveDefaultWorkletUrl(): string {
  if (typeof chrome !== 'undefined' && chrome.runtime?.getURL) {
    return chrome.runtime.getURL('voice-capture-worklet.js');
  }
  return 'voice-capture-worklet.js';
}

/**
 * Strict MV3 CSP-compliant microphone `AudioInputSource`.
 * Loads the self-hosted static `voice-capture-worklet.js` module rather than creating
 * inline `Blob:` URLs (which `script-src 'self'` blocks).
 */
export class Mv3AudioWorkletInputSource implements AudioInputSource {
  private readonly options: Mv3AudioWorkletInputOptions;
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private silentGainNode: GainNode | null = null;
  private active = false;
  private currentSampleRate: number;

  constructor(options: Mv3AudioWorkletInputOptions = {}) {
    this.options = options;
    this.currentSampleRate = options.sampleRate ?? 16000;
  }

  get sampleRate(): number {
    return this.currentSampleRate;
  }

  get isActive(): boolean {
    return this.active;
  }

  async start(
    onFrame: (samples: Float32Array, sampleRate: number, timestampMs: number) => void,
  ): Promise<void> {
    if (this.active) {
      return;
    }
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone capture (navigator.mediaDevices.getUserMedia) is unavailable.');
    }

    const constraints: MediaStreamConstraints = {
      audio: {
        deviceId: this.options.deviceId ? { exact: this.options.deviceId } : undefined,
        echoCancellation: this.options.echoCancellation ?? true,
        noiseSuppression: this.options.noiseSuppression ?? true,
        autoGainControl: this.options.autoGainControl ?? true,
      },
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.stream = stream;

    const AudioCtx =
      globalThis.AudioContext ??
      (globalThis as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) {
      await this.stop();
      throw new Error('Web Audio AudioContext is unavailable.');
    }

    const ctx = new AudioCtx({
      sampleRate: this.options.sampleRate,
    });
    this.ctx = ctx;
    this.currentSampleRate = ctx.sampleRate || 16000;

    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    const workletUrl = this.options.workletUrl ?? resolveDefaultWorkletUrl();
    if (!ctx.audioWorklet) {
      await this.stop();
      throw new Error('AudioWorklet is unavailable in this execution context.');
    }

    await ctx.audioWorklet.addModule(workletUrl);

    const sourceNode = ctx.createMediaStreamSource(stream);
    const workletNode = new AudioWorkletNode(ctx, AUDIO_WORKLET_CAPTURE_PROCESSOR_NAME);
    const silentGain = ctx.createGain();
    silentGain.gain.value = 0;

    workletNode.port.onmessage = (event: MessageEvent<unknown>) => {
      if (!this.active) return;
      const data = event.data as
        | Float32Array
        | { type?: string; samples?: Float32Array; sampleRate?: number };
      if (data instanceof Float32Array) {
        onFrame(data, this.currentSampleRate, performance.now());
        return;
      }
      if (data && data.samples instanceof Float32Array) {
        onFrame(data.samples, data.sampleRate ?? this.currentSampleRate, performance.now());
      }
    };

    sourceNode.connect(workletNode);
    workletNode.connect(silentGain);
    silentGain.connect(ctx.destination);

    this.sourceNode = sourceNode;
    this.workletNode = workletNode;
    this.silentGainNode = silentGain;
    this.active = true;
  }

  async stop(): Promise<void> {
    this.active = false;

    if (this.workletNode) {
      try {
        this.workletNode.port.onmessage = null;
        this.workletNode.disconnect();
      } catch {
        // Ignore teardown race
      }
      this.workletNode = null;
    }

    if (this.silentGainNode) {
      try {
        this.silentGainNode.disconnect();
      } catch {
        // Ignore
      }
      this.silentGainNode = null;
    }

    if (this.sourceNode) {
      try {
        this.sourceNode.disconnect();
      } catch {
        // Ignore
      }
      this.sourceNode = null;
    }

    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        try {
          track.stop();
        } catch {
          // Ignore
        }
      }
      this.stream = null;
    }

    if (this.ctx) {
      try {
        await this.ctx.close();
      } catch {
        // Ignore
      }
      this.ctx = null;
    }
  }

  async dispose(): Promise<void> {
    await this.stop();
  }
}
