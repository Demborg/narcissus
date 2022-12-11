import React, { useRef, useState, useEffect, useCallback } from "react";
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

interface SliderProps {
  updater(value: number, index: number): void;
  index: number;
}

const LatentSlider = (props: SliderProps) => {
  const [value, setValue] = useState(0);

  const updater = props.updater;

  useEffect(() => {
    const v = 6 * Math.random() - 3;
    setValue(v);
    updater(v, props.index);
  }, [updater, props.index]);
  return (
    <div>
      {value !== 0 && (
        <input
          type="range"
          className="Slider"
          min="-3"
          max="3"
          defaultValue={value}
          step="0.1"
          onChange={(e) =>
            props.updater(parseFloat(e.target.value), props.index)
          }
        />
      )}
      <br />
    </div>
  );
};

const drawLatent = async (
  canvas: HTMLCanvasElement,
  model: LayersModel,
  latent: number[]
) => {
  const tensor = model.predict(tensor2d([latent])) as Tensor4D;
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
};

const VAECanvas = (props: { latent: number[] }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<LayersModel>();
  const [offset, setOffset] = useState<[number, number]>([0, 0]);
  const latent = props.latent;
  console.log(latent);

  useEffect(() => {
    loadLayersModel(
      "https://raw.githubusercontent.com/Demborg/narcissus/master/public/decoder/model.json"
    ).then((model) => setModel(model));
  }, []);

  const onMouseMove = useCallback((e) => {
    const targetRect = e.currentTarget.getBoundingClientRect();
    setOffset([
      e.nativeEvent.offsetX / targetRect.width,
      e.nativeEvent.offsetY / targetRect.height,
    ]);
  }, []);

  if (model && canvasRef.current) {
    drawLatent(
      canvasRef.current,
      model,
      latent.map((v, i) => (i < 2 ? v + 2 * offset[i] : v))
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className="Latent-Canvas"
      onMouseMove={onMouseMove}
    />
  );
};

const VAEDecoder = (props: DecoderProps) => {
  const [latent, setLatent] = useState<number[]>(
    Array(props.latent_dim).fill(0)
  );
  const updater = useCallback(
    (value: number, index: number) =>
      setLatent((latent) => {
        let stuff = JSON.parse(JSON.stringify(latent));
        stuff[index] = value;
        return stuff;
      }),
    []
  );
  return (
    <div>
      <VAECanvas latent={latent} />
      <div>explore the latent space</div>
      <div>
        {latent.map((_, index) => (
          <LatentSlider updater={updater} index={index} key={index} />
        ))}
      </div>
    </div>
  );
};

const App = () => {
  const params = new URLSearchParams(window.location.search);
  if (params.get("fullscreen")) {
    return (
      <div className="Fullscreen">
        <VAECanvas
          latent={Array(16)
            .fill(0)
            .map(() => 3 - 6 * Math.random())}
        />
      </div>
    );
  }
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
