import {
  ChromeRuntimePortTransport,
  Mv3MicPermissionHelper,
  Mv3SidePanelClient,
  WebAudioPlaybackSink,
} from '@yours/voice/adapters';
import type { TranscriptSegment, VoiceState } from '@yours/voice/contracts';
import { LocalVoicePipeline } from '@yours/voice/pipeline';
import { VadEngine } from '@yours/voice/vad';

export const VOICE_PORT_NAME = 'yours-voice-bridge';

export interface VoiceStatusSnapshot {
  state: VoiceState;
  active: boolean;
  micLevel: number;
  partialTranscript: string;
  finalTranscript: string;
  speaking: boolean;
  bargeInCount: number;
  error: string | null;
}

interface SpeechRecognitionAlternativeLike {
  transcript?: string;
}

interface SpeechRecognitionResultLike {
  isFinal?: boolean;
  0?: SpeechRecognitionAlternativeLike;
}

interface SpeechRecognitionEventLike {
  resultIndex?: number;
  results?: ArrayLike<SpeechRecognitionResultLike>;
}

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onerror: ((ev: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
}

/**
 * Coordinates local voice dictation (`@yours/voice` `LocalVoicePipeline` + `VadEngine` +
 * `Mv3SidePanelClient` offscreen bridge + `Mv3MicPermissionHelper`) and text-to-speech playback
 * inside the Chromium MV3 Side Panel.
 */
export class SidepanelVoiceController {
  private sidePanelClient: Mv3SidePanelClient | null = null;
  private pipeline: LocalVoicePipeline | null = null;
  private transport: ChromeRuntimePortTransport | null = null;
  private playbackSink: WebAudioPlaybackSink | null = null;
  private recognition: SpeechRecognitionLike | null = null;
  private listeners = new Set<(snapshot: VoiceStatusSnapshot) => void>();
  private snapshot: VoiceStatusSnapshot = {
    state: 'idle',
    active: false,
    micLevel: 0,
    partialTranscript: '',
    finalTranscript: '',
    speaking: false,
    bargeInCount: 0,
    error: null,
  };

  subscribe(listener: (snapshot: VoiceStatusSnapshot) => void): () => void {
    this.listeners.add(listener);
    listener(this.snapshot);
    return () => {
      this.listeners.delete(listener);
    };
  }

  getSnapshot(): VoiceStatusSnapshot {
    return this.snapshot;
  }

  private updateSnapshot(patch: Partial<VoiceStatusSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const listener of this.listeners) {
      listener(this.snapshot);
    }
  }

  private async ensurePipeline(): Promise<LocalVoicePipeline> {
    if (this.pipeline) {
      return this.pipeline;
    }

    if (typeof chrome === 'undefined' || !chrome.runtime?.connect) {
      throw new Error('Chrome runtime port bridge is unavailable in this environment.');
    }

    const port = chrome.runtime.connect({ name: VOICE_PORT_NAME });
    const transport = new ChromeRuntimePortTransport(port);
    const permissionHelper = new Mv3MicPermissionHelper({
      permissionsPageUrl: 'permissions.html',
      strategy: 'iframe-then-tab',
    });
    const playbackSink = new WebAudioPlaybackSink();
    const sidePanelClient = new Mv3SidePanelClient({
      transport,
      permissionHelper,
      fallbackPlaybackSink: playbackSink,
      useSidePanelFallbackPlayback: true,
    });

    const inputSource = sidePanelClient.createInputSource();
    const vad = new VadEngine({
      emitProbabilityEvents: true,
    });

    const pipeline = new LocalVoicePipeline({
      inputSource,
      vad,
      enableIncrementalPartials: true,
    });

    pipeline.on('stateChange', ({ current }) => {
      this.updateSnapshot({
        state: current,
        active: current !== 'idle',
        speaking: current === 'speaking',
      });
    });

    pipeline.on('vad', (ev) => {
      if (ev.type === 'probability' || ev.type === 'speech_frame' || ev.type === 'speech_start') {
        const level = Math.max(0.05, Math.min(1, ev.probability));
        this.updateSnapshot({ micLevel: level });
      } else if (ev.type === 'speech_end') {
        this.updateSnapshot({ micLevel: 0.08 });
      }
    });

    pipeline.on('transcript', (seg: TranscriptSegment) => {
      if (seg.isFinal) {
        this.updateSnapshot({
          finalTranscript: seg.text,
          partialTranscript: '',
        });
      } else {
        this.updateSnapshot({ partialTranscript: seg.text });
      }
    });

    pipeline.on('bargeIn', () => {
      this.stopSpeechSynthesis();
      this.updateSnapshot({
        bargeInCount: this.snapshot.bargeInCount + 1,
        speaking: false,
      });
    });

    pipeline.on('error', ({ message }) => {
      this.updateSnapshot({ error: message });
    });

    this.transport = transport;
    this.playbackSink = playbackSink;
    this.sidePanelClient = sidePanelClient;
    this.pipeline = pipeline;
    return pipeline;
  }

  private startBrowserSpeechRecognitionFallback(): void {
    const g = globalThis as unknown as {
      SpeechRecognition?: new () => SpeechRecognitionLike;
      webkitSpeechRecognition?: new () => SpeechRecognitionLike;
    };
    const Ctor = g.SpeechRecognition ?? g.webkitSpeechRecognition;
    if (!Ctor || this.recognition) {
      return;
    }

    try {
      const rec = new Ctor();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = 'en-US';
      rec.onresult = (ev) => {
        if (!ev.results) return;
        let interim = '';
        let finalText = '';
        for (let i = ev.resultIndex ?? 0; i < ev.results.length; i++) {
          const res = ev.results[i];
          const text = res?.[0]?.transcript ?? '';
          if (res?.isFinal) {
            finalText += text;
          } else {
            interim += text;
          }
        }
        if (interim) {
          this.updateSnapshot({ partialTranscript: interim.trim(), micLevel: 0.72 });
        }
        if (finalText.trim()) {
          this.updateSnapshot({
            finalTranscript: finalText.trim(),
            partialTranscript: '',
            micLevel: 0.25,
          });
        }
      };
      rec.onerror = () => {
        // Silent fallback: Silero/Energy VAD via offscreen pipeline still tracks voice activity
      };
      rec.onend = () => {
        this.recognition = null;
      };
      rec.start();
      this.recognition = rec;
    } catch {
      this.recognition = null;
    }
  }

  async startDictation(): Promise<void> {
    this.updateSnapshot({ error: null, partialTranscript: '', finalTranscript: '' });
    try {
      const pipeline = await this.ensurePipeline();
      await pipeline.startListening({ bargeInEnabled: true });
      this.startBrowserSpeechRecognitionFallback();
      this.updateSnapshot({ active: true, state: pipeline.state, micLevel: 0.2 });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this.updateSnapshot({ active: false, state: 'idle', error: msg });
    }
  }

  async stopDictation(): Promise<string> {
    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch {
        // Ignore
      }
      this.recognition = null;
    }

    if (this.pipeline) {
      await this.pipeline.stopListening().catch(() => {});
    }

    const transcript = (this.snapshot.finalTranscript || this.snapshot.partialTranscript).trim();
    this.updateSnapshot({
      active: false,
      state: 'idle',
      micLevel: 0,
    });
    return transcript;
  }

  async toggleDictation(): Promise<{ active: boolean; transcript: string }> {
    if (this.snapshot.active) {
      const transcript = await this.stopDictation();
      return { active: false, transcript };
    }
    await this.startDictation();
    return { active: true, transcript: '' };
  }

  speakText(text: string): void {
    const trimmed = text.trim();
    if (!trimmed) return;
    this.stopSpeechSynthesis();
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return;
    }
    try {
      const utterance = new SpeechSynthesisUtterance(trimmed.slice(0, 4000));
      utterance.onstart = () => this.updateSnapshot({ speaking: true });
      utterance.onend = () => this.updateSnapshot({ speaking: false });
      utterance.onerror = () => this.updateSnapshot({ speaking: false });
      window.speechSynthesis.speak(utterance);
    } catch {
      this.updateSnapshot({ speaking: false });
    }
  }

  stopSpeechSynthesis(): void {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // Ignore
      }
    }
    if (this.playbackSink) {
      this.playbackSink.stopAll();
    }
    this.updateSnapshot({ speaking: false });
  }

  async dispose(): Promise<void> {
    await this.stopDictation().catch(() => {});
    this.stopSpeechSynthesis();
    if (this.pipeline) {
      await this.pipeline.dispose().catch(() => {});
      this.pipeline = null;
    }
    if (this.sidePanelClient) {
      this.sidePanelClient.dispose();
      this.sidePanelClient = null;
    }
    if (this.playbackSink) {
      await this.playbackSink.dispose().catch(() => {});
      this.playbackSink = null;
    }
    if (this.transport) {
      this.transport.close();
      this.transport = null;
    }
  }
}
