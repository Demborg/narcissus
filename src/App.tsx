import React from "react";
import {
  LayersModel,
  loadLayersModel,
  tensor2d,
  Tensor4D,
} from "@tensorflow/tfjs";
import "./App.css";

interface DecoderProps {
  latent_dim: number;
}

interface DecoderState {
  latent: number[];
  model?: LayersModel;
}

interface SliderProps {
  value: number;
  updater(value: number): void;
}

const LatentSlider = (props: SliderProps) => {
  return (
    <div>
      <input
        type="range"
        className="Slider"
        min="-3"
        max="3"
        value={props.value}
        step="0.1"
        onChange={(e) => props.updater(parseFloat(e.target.value))}
      />
      <br />
    </div>
  );
};

class VAEDecoder extends React.Component<DecoderProps, DecoderState> {
  private canvasRef = React.createRef<HTMLCanvasElement>();

  constructor(props: DecoderProps) {
    super(props);
    let latent: number[] = [];
    for (let i = 0; i < props.latent_dim; i++) {
      latent.push(6 * Math.random() - 3);
    }

    this.state = { latent: latent };
  }

  async componentDidMount() {
    const model = await loadLayersModel(
      "https://raw.githubusercontent.com/Demborg/narcissus/master/public/decoder/model.json"
    );
    this.setState({ model: model });
  }

  async drawLatent() {
    const canvas = this.canvasRef.current;
    if (this.state.model != null && canvas != null) {
      const tensor = this.state.model.predict(
        tensor2d([this.state.latent])
      ) as Tensor4D;
      const imgTensor = tensor.reshape([
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
      ]);
      const [height, width] = imgTensor.shape.slice(0, 2);

      const data = await imgTensor.data();
      const bytes = new Uint8ClampedArray(width * height * 4);

      for (let i = 0; i < height * width; ++i) {
        let r, g, b;
        r = data[i * 3] * 255;
        g = data[i * 3 + 1] * 255;
        b = data[i * 3 + 2] * 255;

        const j = i * 4;
        bytes[j + 0] = Math.round(r);
        bytes[j + 1] = Math.round(g);
        bytes[j + 2] = Math.round(b);
        bytes[j + 3] = 255;
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      const imageData = new ImageData(bytes, width, height);
      if (ctx != null) {
        ctx.putImageData(imageData, 0, 0);
      }
    }
  }

  updateLatent(index: number, value: number) {
    const latent = this.state.latent.slice();
    latent[index] = value;
    this.setState({ latent: latent });
  }

  render() {
    this.drawLatent();
    return (
      <div>
        <canvas ref={this.canvasRef} className="Latent-Canvas" />
        <div>explore the latent space</div>
        <div>
          {this.state.latent.map((value, index) => (
            <LatentSlider
              value={value}
              updater={(value: number) => this.updateLatent(index, value)}
            />
          ))}
        </div>
      </div>
    );
  }
}

const App = () => {
  return (
    <div className="App">
      <header className="App-header">
        <h2>This portrait doesn't exist (duh)</h2>
        <VAEDecoder latent_dim={16} />
      </header>
    </div>
  );
};

export default App;
