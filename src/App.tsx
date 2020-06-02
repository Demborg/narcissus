import React from 'react';
import * as tf from '@tensorflow/tfjs'
import modelJSON from './decoder/model.json'
import './App.css';

interface DecoderProps {
  size: number;
  latent_dim: number;
}

interface DecoderState {
  latent: number[];
  model?: tf.LayersModel;
}

interface SliderProps {
  value: number;
}


class LatentSlider extends React.Component<SliderProps, {}> {
  render () {
    return (
      <div>
        <input type="range" className="Slider" min="-3" max="3" value={this.props.value} step="0.1" id="slider1"/>
        <br/>
      </div>
    )
  }
}

class VAEDecoder extends React.Component<DecoderProps, DecoderState>{
  constructor(props: DecoderProps) {
    super(props);
    let latent: number[] = [];
    for (let i = 0; i < props.latent_dim; i++){
      latent.push(6 * Math.random() - 3)
    }

    this.state = {'latent': latent}
  }

  async componentDidMount() {
    const model = await tf.loadLayersModel('https://drive.google.com/file/d/1--BjrYN4fMRNbNq63XX1dphx5eIA7MaA')
    this.state = {'latent': this.state.latent, 'model': model}
  }

  render() {
    return (
      <div>
        <canvas
        ref="canvas"
        width={this.props.size}
        height={this.props.size}
        />
        <div>explore the latent space</div>
        <div>
          {this.state.latent.map(value => <LatentSlider value={value}/>)}
        </div>
      </div>
    )
  }
}

function App() {
  const model = tf.loadLayersModel('./decoder/model.json')
  return (
    <div className="App">
      <header className="App-header">
        <h2>
          This portrait doesn't exist (duh)
        </h2>
        <VAEDecoder size={128} latent_dim={16}/>
      </header>
    </div>
  );
}

export default App;
