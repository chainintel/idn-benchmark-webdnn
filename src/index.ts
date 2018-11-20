const { WebDNNBackend } = require('@idn/backend-webdnn/dist/node');

const { fromBuffer, toBuffer } = require('@idn/util-buffer');
const { Loader } = require('@idn/loader');

const { performance } = require('perf_hooks');
const async = require('async');
const { w } = require('@idn/util-promisify');

class WebDNNBenchmark {
  backend;
  numSamples;
  constructor(type = 'webdnn/fallback', numSamples = 25) {
    this.backend = new WebDNNBackend([type]);
    this.numSamples = numSamples;
  }
  async benchmark(modelPkg) {
    const loader = new Loader();
    const model = await loader.load(modelPkg);
    let runner = await this.backend.init(model);
    // prerun
    for (let i = 0; i < 1; i++) {
      let result = await runner.infer([
        toBuffer(new Float32Array(model.inputs[0].shape.reduce((a, b) => a * b)))
      ]);
    }
    let numSamples = this.numSamples;
    let start = performance.now();
    for (let i = 0; i < numSamples; i++) {
      let result = await runner.infer([
        toBuffer(new Float32Array(model.inputs[0].shape.reduce((a, b) => a * b)))
      ]);
    }
    let end = performance.now();
    let rate = numSamples / (end - start);
    return rate;
  }
  async getScore() {
    let pkgs = [
      '@idn/model-webdnn-benchmark-lenet',
      '@idn/model-webdnn-benchmark-alexnet',
      '@idn/model-webdnn-benchmark-vgg11',
      '@idn/model-webdnn-benchmark-vgg11_bn',
      '@idn/model-webdnn-benchmark-resnet18'
    ];
    return new Promise((resolve, reject) => {
      let rates = async.mapLimit(
        pkgs,
        1,
        async (pkg, cb) => {
          let [err, rate] = await w(this.benchmark(pkg));
          cb(err, rate);
        },
        (err, rates) => {
          if (err) {
            reject(err);
            return;
          }
          // calculate harmonic mean
          let score = rates.length / rates.reduce((rateA, rateB) => 1 / rateA + 1 / rateB);
          resolve((10000 * score).toFixed(0));
        }
      );
    });
  }
}

export { WebDNNBenchmark };
