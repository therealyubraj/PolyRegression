let xs = [];
let ys = [];
let a, b, c, d;

const learningRate = 0.6;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(480, 600);
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));

}

function predict(xArr) {
  const xTen = tf.tensor1d(xArr);
  //y = ax3+bx2+cx+d
  const yTen = (xTen.pow(3).mul(a))
    .add(xTen.square().mul(b))
    .add(xTen.mul(c))
    .add(d);
  return yTen;
}

function loss(pred, label) {
  //const loss = (pred, label) => pred.sub(label).square().mean();
  return pred.sub(label).square().mean();
}


function draw() {
  background(0);
  textSize(32);
  strokeWeight(0);
  fill(255);
  text("Click to add dots ", 10, 30);
  tf.tidy(() => {
    if (ys.length > 0) {
      const yTen = tf.tensor1d(ys);
      optimizer.minimize(() => loss(predict(xs), yTen));
    }
  });

  for (let i = 0; i < xs.length; i++) {
    stroke(255);
    strokeWeight(5);
    let px = map(xs[i], -1, 1, 0, width);
    let py = map(ys[i], -1, 1, height, 0);
    point(px, py);
  }

  strokeWeight(2);
  tf.tidy(() => {
    let x = [];
    for (let i = -1; i <= 1; i += 0.05) {
      x.push(i);
    }
    const yTen = predict(x);
    let y = yTen.dataSync();
    noFill();
    beginShape();
    for (let i = 0; i < x.length; i++) {
      let px = map(x[i], -1, 1, 0, width);
      let py = map(y[i], -1, 1, height, 0);
      vertex(px, py);
    }
    endShape();
  });
}


function mousePressed() {
  console.log('HERE');
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, 1, -1);
  xs.push(x);
  ys.push(y);
}