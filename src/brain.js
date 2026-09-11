import { clamp, sliderToAnswer } from './math.js';
import { graphFeatureVector } from './environment.js';

export class GraphSensorEncoder {
  constructor({ sampleCount = 9 } = {}) {
    this.sampleCount = sampleCount;
  }

  encode(problem) {
    return Float32Array.from(graphFeatureVector(problem, { sampleCount: this.sampleCount }));
  }
}

export class SliderMotorDecoder {
  decode(normalizedPosition, problem) {
    const sliderPosition = clamp(Number(normalizedPosition), 0, 1);
    return {
      sliderPosition,
      value: sliderToAnswer(sliderPosition, problem.answerRange),
    };
  }
}

export class ConnectomeAgentAdapter {
  constructor({
    network,
    sensorEncoder = new GraphSensorEncoder(),
    motorDecoder = new SliderMotorDecoder(),
    kind = 'connectome-adapter',
  }) {
    if (!network || typeof network.run !== 'function') {
      throw new TypeError('ConnectomeAgentAdapter requires a network with run(input, context)');
    }
    this.network = network;
    this.sensorEncoder = sensorEncoder;
    this.motorDecoder = motorDecoder;
    this.kind = kind;
  }

  estimate(problem, options = {}) {
    const sensoryInput = this.sensorEncoder.encode(problem, options);
    const motorOutput = this.network.run(sensoryInput, { problem, ...options });
    const decoded = this.motorDecoder.decode(motorOutput, problem, options);

    return {
      kind: this.kind,
      value: decoded.value,
      sliderPosition: decoded.sliderPosition,
      confidence: decoded.confidence ?? null,
      trace: decoded.trace ?? [0.5, decoded.sliderPosition],
    };
  }
}
