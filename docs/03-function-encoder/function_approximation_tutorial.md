# Function Approximation

An interactive guide to Fourier basis, RBF, and neural networks.

## 1. Introduction to Function Approximation

Function approximation is the task of finding a simpler function that closely matches a complex target function. This is fundamental in machine learning, signal processing, and numerical analysis.

**Key Idea:** Represent any function f(x) as a combination of simpler "basis functions":

$$f(x) \approx \sum_i w_i \phi_i(x)$$

where $\phi_i(x)$ are basis functions and $w_i$ are weights/coefficients.

**Why Function Approximation?**

- **Compression:** Store complex functions using fewer parameters
- **Generalization:** Learn patterns from data and predict on new inputs
- **Analysis:** Understand function properties through basis decomposition
- **Computation:** Replace expensive function evaluations with fast approximations

## 2. Fourier Basis Approximation

Fourier approximation represents functions as sums of sine and cosine waves at different frequencies.

**Fourier Series:**

$$f(x) = a_0 + \sum_n [a_n\cos(nx) + b_n\sin(nx)]$$

**Coefficients:**

$$a_0 = \frac{1}{2\pi} \int f(x)dx$$

$$a_n = \frac{1}{\pi} \int f(x)\cos(nx)dx$$

$$b_n = \frac{1}{\pi} \int f(x)\sin(nx)dx$$

<div class="fa-container">
<div class="fa-section">
<div class="fa-header">
<h3>Interactive Fourier Approximation</h3>
</div>

<div class="fa-controls">
    <div class="fa-control-group">
        <label>Terms:</label>
        <input type="range" id="fourier-terms" min="1" max="20" value="5">
        <span id="fourier-terms-val">5</span>
    </div>
    <div class="fa-control-group">
        <label>Function:</label>
        <select id="fourier-func">
            <option value="square">Square Wave</option>
            <option value="sawtooth">Sawtooth Wave</option>
            <option value="triangle">Triangle Wave</option>
        </select>
    </div>
</div>

<div class="fa-grid-2">
    <div>
        <h4 style="text-align: center; color: #333; margin: 10px 0;">Approximation</h4>
        <canvas id="fourier-canvas" class="fa-canvas" style="height: 350px;"></canvas>
    </div>
    <div>
        <h4 style="text-align: center; color: #333; margin: 10px 0;">Individual Basis Functions</h4>
        <canvas id="fourier-basis-canvas" class="fa-canvas" style="height: 350px;"></canvas>
    </div>
</div>

<div class="fa-info-box">
<strong>Key Properties</strong>
<ul style="margin: 5px 0 0 20px;">
<li><strong>Global basis:</strong> Each term affects the entire domain</li>
<li><strong>Frequency interpretation:</strong> Low frequencies = trends, high frequencies = details</li>
<li><strong>Best for:</strong> Periodic signals, smooth functions</li>
<li><strong>Gibbs phenomenon:</strong> Overshoot at discontinuities (~9%)</li>
</ul>
</div>
</div>
</div>

## 3. Radial Basis Functions (RBF)

RBF networks represent functions as weighted sums of "bumps" centered at different points.

**RBF Approximation:**

$$f(x) = \sum_i w_i \phi(||x - c_i||)$$

**Common RBF Types:**

- Gaussian: $\phi(r) = \exp(-r^2/\sigma^2)$
- Multiquadric: $\phi(r) = \sqrt{r^2 + \sigma^2}$
- Inverse multiquadric: $\phi(r) = 1/\sqrt{r^2 + \sigma^2}$

<div class="fa-container">
<div class="fa-section">
<div class="fa-header">
<h3>Interactive RBF Approximation</h3>
</div>

<div class="fa-controls">
    <div class="fa-control-group">
        <label>Centers:</label>
        <input type="range" id="rbf-centers" min="3" max="15" value="7">
        <span id="rbf-centers-val">7</span>
    </div>
    <div class="fa-control-group">
        <label>Width (σ):</label>
        <input type="range" id="rbf-width" min="0.1" max="2" step="0.1" value="0.5">
        <span id="rbf-width-val">0.5</span>
    </div>
    <div class="fa-control-group">
        <label>Type:</label>
        <select id="rbf-type">
            <option value="gaussian">Gaussian</option>
            <option value="multiquadric">Multiquadric</option>
            <option value="inverse">Inverse Multiquadric</option>
        </select>
    </div>
</div>

<div class="fa-grid-2">
    <div>
        <h4 style="text-align: center; color: #333; margin: 10px 0;">Approximation</h4>
        <canvas id="rbf-canvas" class="fa-canvas" style="height: 350px;"></canvas>
    </div>
    <div>
        <h4 style="text-align: center; color: #333; margin: 10px 0;">Individual RBF Basis Functions</h4>
        <canvas id="rbf-basis-canvas" class="fa-canvas" style="height: 350px;"></canvas>
    </div>
</div>

<div class="fa-info-box">
<strong>Key Properties</strong>
<ul style="margin: 5px 0 0 20px;">
<li><strong>Local basis:</strong> Each RBF has limited influence</li>
<li><strong>Spatial interpretation:</strong> Centers determine which regions are captured</li>
<li><strong>Best for:</strong> Scattered data, non-periodic functions, interpolation</li>
<li><strong>Meshless:</strong> Works in multiple dimensions without structured grid</li>
</ul>
</div>
</div>
</div>

<style>
    .fa-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        margin: 20px auto;
        max-width: 1200px;
        background-color: #f9f9f9;
        padding: 20px;
        border: 1px solid #ccc;
        border-radius: 8px;
    }

    .fa-section {
        background-color: #fff;
        margin: 20px 0;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #ddd;
    }

    .fa-header {
        text-align: center;
        margin-bottom: 15px;
        color: #333;
    }

    .fa-header h3 {
        margin: 0 0 10px 0;
        color: #007bff;
    }

    .fa-controls {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 15px 0;
        flex-wrap: wrap;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }

    .fa-control-group {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #333;
    }

    .fa-control-group label {
        font-weight: 600;
        font-size: 14px;
    }

    .fa-control-group input[type="range"] {
        width: 150px;
    }

    .fa-control-group select {
        padding: 5px 10px;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 14px;
    }

    .fa-button {
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        background-color: #28a745;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    .fa-button:hover {
        background-color: #218838;
    }

    .fa-canvas {
        width: 100%;
        height: 400px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: white;
        margin: 10px 0;
    }

    .fa-info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 15px 0;
        border-radius: 4px;
        color: #333;
    }

    .fa-info-box strong {
        display: block;
        margin-bottom: 8px;
        color: #0056b3;
    }

    .fa-grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }

    @media (max-width: 768px) {
        .fa-grid-2 {
            grid-template-columns: 1fr;
        }
    }

    .fa-status {
        text-align: center;
        padding: 10px;
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin: 10px 0;
        color: #333;
        font-weight: 600;
    }
</style>

<script>
// ============================================
// UTILITY FUNCTIONS
// ============================================

function drawAxes(ctx, width, height, padding, xMin, xMax, yMin, yMax) {
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
}

function toCanvasX(x, xMin, xMax, width, padding) {
    return padding + ((x - xMin) / (xMax - xMin)) * (width - 2 * padding);
}

function toCanvasY(y, yMin, yMax, height, padding) {
    return height - padding - ((y - yMin) / (yMax - yMin)) * (height - 2 * padding);
}

function drawFunction(ctx, func, xMin, xMax, yMin, yMax, width, height, padding, color, lineWidth, dash = []) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash(dash);
    ctx.beginPath();

    const steps = 500;
    for (let i = 0; i <= steps; i++) {
        const x = xMin + (i / steps) * (xMax - xMin);
        const y = func(x);
        const canvasX = toCanvasX(x, xMin, xMax, width, padding);
        const canvasY = toCanvasY(y, yMin, yMax, height, padding);

        if (i === 0) ctx.moveTo(canvasX, canvasY);
        else ctx.lineTo(canvasX, canvasY);
    }
    ctx.stroke();
    ctx.setLineDash([]);
}

function drawPoints(ctx, points, xMin, xMax, yMin, yMax, width, height, padding, color, size) {
    ctx.fillStyle = color;
    points.forEach(p => {
        const canvasX = toCanvasX(p.x, xMin, xMax, width, padding);
        const canvasY = toCanvasY(p.y, yMin, yMax, height, padding);
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, size, 0, 2 * Math.PI);
        ctx.fill();
    });
}

function drawLabel(ctx, text, x, y, color = '#333') {
    ctx.fillStyle = color;
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(text, x, y);
}

function drawLegend(ctx, items, x, y) {
    let currentY = y;
    items.forEach(item => {
        ctx.fillStyle = item.color;
        ctx.fillRect(x, currentY - 8, 20, 3);
        if (item.dash) {
            ctx.setLineDash(item.dash);
            ctx.strokeStyle = item.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, currentY - 6);
            ctx.lineTo(x + 20, currentY - 6);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        ctx.fillStyle = '#333';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(item.label, x + 25, currentY);
        currentY += 20;
    });
}

// ============================================
// FOURIER APPROXIMATION
// ============================================

function squareWave(x) {
    return Math.sin(x) > 0 ? 1 : -1;
}

function sawtoothWave(x) {
    return 2 * (x / (2 * Math.PI) - Math.floor(x / (2 * Math.PI) + 0.5));
}

function triangleWave(x) {
    return 2 * Math.abs(sawtoothWave(x)) - 1;
}

function fourierTerm(x, k, funcType) {
    if (funcType === 'square') {
        let freq = 2*k - 1;
        return (4/(Math.PI * freq)) * Math.sin(freq * x);
    } else if (funcType === 'sawtooth') {
        return (2/Math.PI) * Math.pow(-1, k+1) * Math.sin(k * x) / k;
    } else if (funcType === 'triangle') {
        let freq = 2*k - 1;
        return (8/(Math.PI*Math.PI * freq*freq)) * Math.pow(-1, k+1) * Math.sin(freq * x);
    }
    return 0;
}

function fourierApprox(x, n, funcType) {
    let sum = 0;
    for (let k = 1; k <= n; k++) {
        sum += fourierTerm(x, k, funcType);
    }
    return sum;
}

function updateFourier() {
    const n = parseInt(document.getElementById('fourier-terms').value);
    const funcType = document.getElementById('fourier-func').value;
    document.getElementById('fourier-terms-val').textContent = n;

    const padding = 50;
    const xMin = -Math.PI;
    const xMax = Math.PI;
    const yMin = -1.5;
    const yMax = 1.5;

    // Update approximation canvas
    const canvas = document.getElementById('fourier-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    drawAxes(ctx, width, height, padding, xMin, xMax, yMin, yMax);

    let trueFunc;
    if (funcType === 'square') trueFunc = squareWave;
    else if (funcType === 'sawtooth') trueFunc = sawtoothWave;
    else trueFunc = triangleWave;

    drawFunction(ctx, trueFunc, xMin, xMax, yMin, yMax, width, height, padding, '#764ba2', 2);
    drawFunction(ctx, x => fourierApprox(x, n, funcType), xMin, xMax, yMin, yMax, width, height, padding, '#667eea', 2, [5, 5]);

    drawLegend(ctx, [
        {color: '#764ba2', label: 'True Function'},
        {color: '#667eea', label: `Fourier (${n} terms)`, dash: [5, 5]}
    ], width - 180, 30);

    drawLabel(ctx, 'x', width - padding + 10, height - padding + 5);
    drawLabel(ctx, 'f(x)', padding - 35, padding - 10);

    // Update basis functions canvas
    const basisCanvas = document.getElementById('fourier-basis-canvas');
    const basisCtx = basisCanvas.getContext('2d');
    basisCanvas.width = basisCanvas.clientWidth;
    basisCanvas.height = basisCanvas.clientHeight;

    const basisWidth = basisCanvas.width;
    const basisHeight = basisCanvas.height;

    basisCtx.clearRect(0, 0, basisWidth, basisHeight);
    drawAxes(basisCtx, basisWidth, basisHeight, padding, xMin, xMax, yMin, yMax);

    const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22', '#1abc9c'];
    for (let k = 1; k <= Math.min(n, 7); k++) {
        drawFunction(basisCtx, x => fourierTerm(x, k, funcType),
            xMin, xMax, yMin, yMax, basisWidth, basisHeight, padding,
            colors[(k-1) % colors.length], 1.5);
    }

    drawLabel(basisCtx, 'x', basisWidth - padding + 10, basisHeight - padding + 5);
    drawLabel(basisCtx, 'φᵢ(x)', padding - 35, padding - 10);
}

document.getElementById('fourier-terms').addEventListener('input', updateFourier);
document.getElementById('fourier-func').addEventListener('change', updateFourier);

// ============================================
// RBF APPROXIMATION
// ============================================

function rbfGaussian(r, sigma) {
    return Math.exp(-(r*r) / (sigma*sigma));
}

function rbfMultiquadric(r, sigma) {
    return Math.sqrt(r*r + sigma*sigma);
}

function rbfInverse(r, sigma) {
    return 1.0 / Math.sqrt(r*r + sigma*sigma);
}

function targetFunc(x) {
    return Math.sin(2*x) * Math.exp(-0.1*x*x);
}

function solveLinearSystem(A, b) {
    const n = A.length;
    const aug = A.map((row, i) => [...row, b[i]]);

    for (let i = 0; i < n; i++) {
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) maxRow = k;
        }
        [aug[i], aug[maxRow]] = [aug[maxRow], aug[i]];

        for (let k = i + 1; k < n; k++) {
            const factor = aug[k][i] / (aug[i][i] + 1e-10);
            for (let j = i; j <= n; j++) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        x[i] = aug[i][n];
        for (let j = i + 1; j < n; j++) {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= (aug[i][i] + 1e-10);
    }
    return x;
}

let currentCenters = [];
let currentWeights = [];

function updateRBF() {
    const numCenters = parseInt(document.getElementById('rbf-centers').value);
    const sigma = parseFloat(document.getElementById('rbf-width').value);
    const rbfType = document.getElementById('rbf-type').value;

    document.getElementById('rbf-centers-val').textContent = numCenters;
    document.getElementById('rbf-width-val').textContent = sigma.toFixed(1);

    const padding = 50;
    const xMin = -3;
    const xMax = 3;
    const yMin = -1.5;
    const yMax = 1.5;

    // Generate centers
    currentCenters = [];
    for (let i = 0; i < numCenters; i++) {
        currentCenters.push(xMin + (i / (numCenters - 1)) * (xMax - xMin));
    }

    // Build RBF matrix
    const Phi = [];
    for (let i = 0; i < currentCenters.length; i++) {
        const row = [];
        for (let j = 0; j < currentCenters.length; j++) {
            const r = Math.abs(currentCenters[i] - currentCenters[j]);
            let val;
            if (rbfType === 'gaussian') val = rbfGaussian(r, sigma);
            else if (rbfType === 'multiquadric') val = rbfMultiquadric(r, sigma);
            else val = rbfInverse(r, sigma);
            row.push(val);
        }
        Phi.push(row);
    }

    const fValues = currentCenters.map(c => targetFunc(c));
    currentWeights = solveLinearSystem(Phi, fValues);

    const rbfFunc = (x) => {
        let sum = 0;
        for (let j = 0; j < currentCenters.length; j++) {
            const r = Math.abs(x - currentCenters[j]);
            let rbfVal;
            if (rbfType === 'gaussian') rbfVal = rbfGaussian(r, sigma);
            else if (rbfType === 'multiquadric') rbfVal = rbfMultiquadric(r, sigma);
            else rbfVal = rbfInverse(r, sigma);
            sum += currentWeights[j] * rbfVal;
        }
        return sum;
    };

    // Update approximation canvas
    const canvas = document.getElementById('rbf-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    drawAxes(ctx, width, height, padding, xMin, xMax, yMin, yMax);

    drawFunction(ctx, targetFunc, xMin, xMax, yMin, yMax, width, height, padding, '#764ba2', 2);
    drawFunction(ctx, rbfFunc, xMin, xMax, yMin, yMax, width, height, padding, '#667eea', 2, [5, 5]);

    const centerPoints = currentCenters.map(c => ({x: c, y: targetFunc(c)}));
    drawPoints(ctx, centerPoints, xMin, xMax, yMin, yMax, width, height, padding, '#ff9800', 5);

    drawLegend(ctx, [
        {color: '#764ba2', label: 'True Function'},
        {color: '#667eea', label: 'RBF Approximation', dash: [5, 5]},
        {color: '#ff9800', label: 'Centers'}
    ], width - 180, 30);

    drawLabel(ctx, 'x', width - padding + 10, height - padding + 5);
    drawLabel(ctx, 'f(x)', padding - 35, padding - 10);

    // Update basis functions canvas
    const basisCanvas = document.getElementById('rbf-basis-canvas');
    const basisCtx = basisCanvas.getContext('2d');
    basisCanvas.width = basisCanvas.clientWidth;
    basisCanvas.height = basisCanvas.clientHeight;

    const basisWidth = basisCanvas.width;
    const basisHeight = basisCanvas.height;

    basisCtx.clearRect(0, 0, basisWidth, basisHeight);
    drawAxes(basisCtx, basisWidth, basisHeight, padding, xMin, xMax, -0.2, 1.5);

    const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22', '#1abc9c'];
    const maxBasisToShow = Math.min(7, numCenters);

    for (let i = 0; i < maxBasisToShow; i++) {
        const center = currentCenters[i];
        const basisFunc = (x) => {
            const r = Math.abs(x - center);
            if (rbfType === 'gaussian') return rbfGaussian(r, sigma);
            else if (rbfType === 'multiquadric') return rbfMultiquadric(r, sigma);
            else return rbfInverse(r, sigma);
        };

        drawFunction(basisCtx, basisFunc, xMin, xMax, -0.2, 1.5,
            basisWidth, basisHeight, padding, colors[i % colors.length], 1.5);
    }

    drawLabel(basisCtx, 'x', basisWidth - padding + 10, basisHeight - padding + 5);
    drawLabel(basisCtx, 'φᵢ(x)', padding - 35, padding - 10);
}

document.getElementById('rbf-centers').addEventListener('input', updateRBF);
document.getElementById('rbf-width').addEventListener('input', updateRBF);
document.getElementById('rbf-type').addEventListener('change', updateRBF);

// ============================================
// NEURAL NETWORK
// ============================================

let nnWeights = null;

function relu(x) { return Math.max(0, x); }
function tanh(x) { return Math.tanh(x); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function activation(x, type) {
    if (type === 'relu') return relu(x);
    if (type === 'tanh') return tanh(x);
    if (type === 'sigmoid') return sigmoid(x) * 2 - 1;
    return x;
}

function nnForward(x, weights, actType) {
    const {w1, b1, w2, b2} = weights;
    const hidden = [];
    for (let i = 0; i < w1.length; i++) {
        const z = w1[i] * x + b1[i];
        hidden.push(activation(z, actType));
    }
    let output = b2;
    for (let i = 0; i < w2.length; i++) {
        output += w2[i] * hidden[i];
    }
    return output;
}

function initializeNN(numUnits) {
    return {
        w1: Array(numUnits).fill(0).map(() => (Math.random() - 0.5) * 2),
        b1: Array(numUnits).fill(0).map(() => (Math.random() - 0.5) * 0.5),
        w2: Array(numUnits).fill(0).map(() => (Math.random() - 0.5) * 2),
        b2: (Math.random() - 0.5) * 0.5
    };
}

function trainNN() {
    const numUnits = parseInt(document.getElementById('nn-units').value);
    const actType = document.getElementById('nn-activation').value;

    nnWeights = initializeNN(numUnits);

    const xTrain = [];
    const yTrain = [];
    for (let i = 0; i < 100; i++) {
        const x = -3 + (i / 99) * 6;
        xTrain.push(x);
        yTrain.push(targetFunc(x));
    }

    const learningRate = 0.01;
    const epochs = 200;

    for (let epoch = 0; epoch < epochs; epoch++) {
        for (let i = 0; i < xTrain.length; i++) {
            const x = xTrain[i];
            const yTrue = yTrain[i];
            const yPred = nnForward(x, nnWeights, actType);
            const error = yPred - yTrue;
            const delta = error * learningRate;

            for (let j = 0; j < numUnits; j++) {
                nnWeights.w2[j] -= delta * 0.1;
                nnWeights.w1[j] -= delta * 0.01;
            }
        }
    }

    let totalLoss = 0;
    for (let i = 0; i < xTrain.length; i++) {
        const pred = nnForward(xTrain[i], nnWeights, actType);
        totalLoss += Math.pow(pred - yTrain[i], 2);
    }
    totalLoss /= xTrain.length;

    document.getElementById('nn-status').textContent = `Training MSE: ${totalLoss.toFixed(4)}`;
    updateNN();
}

function updateNN() {
    const canvas = document.getElementById('nn-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const numUnits = parseInt(document.getElementById('nn-units').value);
    const actType = document.getElementById('nn-activation').value;
    document.getElementById('nn-units-val').textContent = numUnits;

    if (!nnWeights) nnWeights = initializeNN(numUnits);

    const width = canvas.width;
    const height = canvas.height;
    const padding = 50;
    const xMin = -3;
    const xMax = 3;
    const yMin = -1.5;
    const yMax = 1.5;

    ctx.clearRect(0, 0, width, height);
    drawAxes(ctx, width, height, padding, xMin, xMax, yMin, yMax);

    drawFunction(ctx, targetFunc, xMin, xMax, yMin, yMax, width, height, padding, '#764ba2', 2);
    drawFunction(ctx, x => nnForward(x, nnWeights, actType), xMin, xMax, yMin, yMax, width, height, padding, '#667eea', 2, [5, 5]);

    drawLegend(ctx, [
        {color: '#764ba2', label: 'True Function'},
        {color: '#667eea', label: 'Neural Network', dash: [5, 5]}
    ], width - 180, 30);

    drawLabel(ctx, 'x', width - padding + 10, height - padding + 5);
    drawLabel(ctx, 'f(x)', padding - 35, padding - 10);
}

document.getElementById('nn-units').addEventListener('input', updateNN);
document.getElementById('nn-activation').addEventListener('change', updateNN);

// ============================================
// INITIALIZATION
// ============================================

window.addEventListener('load', () => {
    updateFourier();
    updateRBF();
    updateNN();
});
</script>
