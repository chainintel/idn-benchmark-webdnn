'use strict';
var expect = require('chai').expect;
const { WebDNNBenchmark } = require('../dist/index.js');

describe('Benchmark WebDNN', () => {
  it('should run fallback', async () => {
    const benchmark = new WebDNNBenchmark('webdnn/fallback', 10);
    let score = await benchmark.getScore();
    console.log('fallback score', score);
  }).timeout(120000);

  // it('should run gpu', async () => {
  //   const benchmark = new WebDNNBenchmark('webdnn/webassembly', 10);
  //   let score = await benchmark.getScore();
  //   console.log('gpu score', score);
  // }).timeout(30000);
});
