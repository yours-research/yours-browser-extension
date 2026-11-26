class YoursVoiceCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const ch0 = input[0];
    if (!ch0 || ch0.length === 0) {
      return true;
    }
    const len = ch0.length;
    const mono = new Float32Array(len);
    if (input.length === 1) {
      mono.set(ch0);
    } else {
      const inv = 1 / input.length;
      for (let i = 0; i < len; i++) {
        let sum = 0;
        for (let c = 0; c < input.length; c++) {
          sum += input[c][i] || 0;
        }
        mono[i] = sum * inv;
      }
    }
    this.port.postMessage(
      { type: 'pcm', buffer: mono.buffer, sampleRate, currentTime },
      [mono.buffer],
    );
    return true;
  }
}

registerProcessor('yours-voice-capture-processor', YoursVoiceCaptureProcessor);
