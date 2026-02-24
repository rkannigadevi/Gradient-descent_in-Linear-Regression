// Dataset
const data = [
    { x: 1, y: 2 },
    { x: 2, y: 4 },
    { x: 3, y: 6 }
];

const n = data.length;

// State
let m = 0;
let c = 0;
let learningRate = 0.1;
let iteration = 0;
let history = [];
let costHistory = [];
let isRunning = false;
let animationTimeout = null;

// DOM Elements
const mInput = document.getElementById('initial-m');
const cInput = document.getElementById('initial-c');
const lrInput = document.getElementById('learning-rate');
const iterInput = document.getElementById('max-iterations');
const speedInput = document.getElementById('animation-speed');
const speedValue = document.getElementById('speed-value');

const runOneBtn = document.getElementById('run-one');
const runAllBtn = document.getElementById('run-all');
const resetBtn = document.getElementById('reset');

const mathContainer = document.getElementById('math-iterations-container');
const insightText = document.getElementById('insight-text');

// Charts
let regressionChart, costChart;

function initCharts() {
    const ctx1 = document.getElementById('regressionChart').getContext('2d');
    regressionChart = new Chart(ctx1, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Data Points',
                data: data.map(p => ({ x: p.x, y: p.y })),
                backgroundColor: '#e74c3c',
                pointRadius: 6
            }, {
                label: 'Regression Line',
                data: [],
                type: 'line',
                borderColor: '#3498db',
                borderWidth: 2,
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            scales: {
                x: { type: 'linear', position: 'bottom', min: 0, max: 4 },
                y: { min: 0, max: 8 }
            },
            animation: false
        }
    });

    const ctx2 = document.getElementById('costChart').getContext('2d');
    costChart = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Mean Squared Error (J)',
                data: [],
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            scales: {
                x: { title: { display: true, text: 'Iteration' } },
                y: { title: { display: true, text: 'Cost J' } }
            },
            animation: false
        }
    });
}

function updateCharts() {
    // Update Regression Line
    const lineData = [
        { x: 0, y: m * 0 + c },
        { x: 4, y: m * 4 + c }
    ];
    regressionChart.data.datasets[1].data = lineData;
    regressionChart.update();

    // Update Cost Chart
    costChart.data.labels = costHistory.map((_, i) => i);
    costChart.data.datasets[0].data = costHistory;
    costChart.update();
}

function calculateStep() {
    iteration++;

    // Step 2: Predictions and Errors
    const stepResults = data.map(p => {
        const yPred = m * p.x + c;
        const error = p.y - yPred;
        const sqError = error * error;
        const xError = p.x * error;
        return { x: p.x, y: p.y, yPred, error, sqError, xError };
    });

    // Step 3: Cost Calculation
    const sumSqError = stepResults.reduce((sum, r) => sum + r.sqError, 0);
    const cost = sumSqError / n;

    // Step 4: Gradients
    const sumXErr = stepResults.reduce((sum, r) => sum + r.xError, 0);
    const sumErr = stepResults.reduce((sum, r) => sum + r.error, 0);

    const dm = -(2 / n) * sumXErr;
    const dc = -(2 / n) * sumErr;

    // Step 5: Update Parameters
    const oldM = m;
    const oldC = c;
    m = oldM - learningRate * dm;
    c = oldC - learningRate * dc;

    costHistory.push(cost);

    renderMathStep(iteration, oldM, oldC, stepResults, cost, dm, dc, m, c);
    updateCharts();
    updateInsight(dm, dc, cost);

    return { cost, dm, dc };
}

function renderMathStep(iter, m_val, c_val, results, initialCost, dm, dc, m_new, c_new) {
    const template = document.getElementById('iteration-template');
    const clone = template.content.cloneNode(true);

    clone.querySelector('.iter-num').textContent = iter;

    // Step 1: Current Parameters LaTeX
    const step1HTML = String.raw`\[ m = ${m_val.toFixed(4)}, \quad c = ${c_val.toFixed(4)} \]`;
    clone.querySelector('.step1-calc').innerHTML = step1HTML;

    const tbody = clone.querySelector('.table-body');
    results.forEach(r => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>\\( ${r.x} \\)</td>
            <td>\\( ${r.y} \\)</td>
            <td>\\( ${m_val.toFixed(4)} \\times ${r.x} + ${c_val.toFixed(4)} = ${r.yPred.toFixed(4)} \\)</td>
            <td>\\( ${r.y} - ${r.yPred.toFixed(4)} = ${r.error.toFixed(4)} \\)</td>
            <td>\\( (${r.error.toFixed(4)})^2 = ${r.sqError.toFixed(4)} \\)</td>
            <td>\\( ${r.x} \\times (${r.error.toFixed(4)}) = ${r.xError.toFixed(4)} \\)</td>
        `;
        tbody.appendChild(row);
    });

    // Step 3: Cost LaTeX - Consolidated into a single clear derivation line
    const sumSqStr = results.map(r => r.sqError.toFixed(4)).join(' + ');
    const sumSqVal = results.reduce((s, r) => s + r.sqError, 0);
    const costCalcHTML = String.raw`
        <p>\[ J = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 \]</p>
        <p>\[ J = \frac{1}{${n}} (${sumSqStr}) = \frac{${sumSqVal.toFixed(4)}}{${n}} = ${initialCost.toFixed(6)} \]</p>
    `;
    const step3Container = clone.querySelector('.step3-calc');
    step3Container.innerHTML = costCalcHTML;
    step3Container.classList.add('highlight-math');

    // Step 4: Gradient LaTeX - Refined for clarity
    const sumXErrStr = results.map(r => `(${r.xError.toFixed(4)})`).join(' + ');
    const sumXErrVal = results.reduce((s, r) => s + r.xError, 0);
    const dmCalcHTML = String.raw`
        <p>\[ \frac{\partial J}{\partial m} = -\frac{2}{n} \sum x(y-\hat{y}) \]</p>
        <p>\[ \frac{\partial J}{\partial m} = -\frac{2}{${n}} [ ${sumXErrStr} ] \]</p>
        <p>\[ \frac{\partial J}{\partial m} = -\frac{2}{${n}} (${sumXErrVal.toFixed(4)}) = ${dm.toFixed(4)} \]</p>
    `;
    const step4MContainer = clone.querySelector('.step4-calc-m');
    step4MContainer.innerHTML = dmCalcHTML;
    step4MContainer.classList.add('highlight-math');

    const sumErrStr = results.map(r => `(${r.error.toFixed(4)})`).join(' + ');
    const sumErrVal = results.reduce((s, r) => s + r.error, 0);
    const dcCalcHTML = String.raw`
        <p>\[ \frac{\partial J}{\partial c} = -\frac{2}{n} \sum (y-\hat{y}) \]</p>
        <p>\[ \frac{\partial J}{\partial c} = -\frac{2}{${n}} [ ${sumErrStr} ] \]</p>
        <p>\[ \frac{\partial J}{\partial c} = -\frac{2}{${n}} (${sumErrVal.toFixed(4)}) = ${dc.toFixed(4)} \]</p>
    `;
    const step4CContainer = clone.querySelector('.step4-calc-c');
    step4CContainer.innerHTML = dcCalcHTML;
    step4CContainer.classList.add('highlight-math');

    // Step 5: Parameter Update LaTeX - Refined for clarity
    const mUpdateHTML = String.raw`
        <p>\[ m_{new} = m_{old} - \alpha \frac{\partial J}{\partial m} \]</p>
        <p>\[ m_{new} = ${m_val.toFixed(4)} - (${learningRate} \times ${dm.toFixed(4)}) \]</p>
        <p>\[ m_{new} = ${m_val.toFixed(4)} - (${(learningRate * dm).toFixed(4)}) = ${m_new.toFixed(4)} \]</p>
        <br>
        <p>\[ c_{new} = c_{old} - \alpha \frac{\partial J}{\partial c} \]</p>
        <p>\[ c_{new} = ${c_val.toFixed(4)} - (${learningRate} \times ${dc.toFixed(4)}) \]</p>
        <p>\[ c_{new} = ${c_val.toFixed(4)} - (${(learningRate * dc).toFixed(4)}) = ${c_new.toFixed(4)} \]</p>
    `;
    const step5Container = clone.querySelector('.step5-calc');
    step5Container.innerHTML = mUpdateHTML;
    step5Container.classList.add('highlight-math');

    // Convergence check
    if (Math.abs(dm) < 0.001 && Math.abs(dc) < 0.001) {
        clone.querySelector('.convergence-msg').style.display = 'block';
    }

    if (iteration === 1) mathContainer.innerHTML = '';

    const newStep = document.createElement('div');
    newStep.appendChild(clone);
    mathContainer.prepend(newStep);

    // Render LaTeX
    if (window.renderMathInElement) {
        renderMathInElement(newStep, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false
        });
    }
}

function updateInsight(dm, dc, cost) {
    let explanation = `Iteration ${iteration}: Cost is ${cost.toFixed(6)}. `;

    if (dm > 0) {
        explanation += `Since ∂J/∂m is positive (${dm.toFixed(3)}), we decrease m to go down the slope. `;
    } else {
        explanation += `Since ∂J/∂m is negative (${dm.toFixed(3)}), we increase m to go down the slope. `;
    }

    if (Math.abs(dm) < 0.01 && Math.abs(dc) < 0.01) {
        explanation = "Model has converged! The gradient is nearly flat, meaning we found the minimum error.";
        isRunning = false;
    }

    insightText.innerHTML = `<strong>Explanation:</strong> ${explanation}`;
}

function runOne() {
    if (iteration === 0) {
        m = parseFloat(mInput.value);
        c = parseFloat(cInput.value);
        learningRate = parseFloat(lrInput.value);
        costHistory = [];
        mathContainer.innerHTML = '';
    }
    calculateStep();
}

async function runAll() {
    if (isRunning) return;
    isRunning = true;

    m = parseFloat(mInput.value);
    c = parseFloat(cInput.value);
    learningRate = parseFloat(lrInput.value);
    const maxIter = parseInt(iterInput.value);

    iteration = 0;
    costHistory = [];
    mathContainer.innerHTML = '';

    while (isRunning && iteration < maxIter) {
        const { cost, dm, dc } = calculateStep();

        if (cost < 0.0001 || (Math.abs(dm) < 0.001 && Math.abs(dc) < 0.001)) {
            isRunning = false;
            break;
        }

        const delay = parseInt(speedInput.value);
        await new Promise(resolve => setTimeout(resolve, delay));
    }
    isRunning = false;
}

function reset() {
    isRunning = false;
    iteration = 0;
    m = parseFloat(mInput.value);
    c = parseFloat(cInput.value);
    costHistory = [];
    mathContainer.innerHTML = '<div class="placeholder-msg">Mathematical derivation will appear here once you start the iteration.</div>';
    insightText.textContent = 'Set your parameters and click "Run One Iteration" to start the learning process.';
    updateCharts();
}

// Event Listeners
runOneBtn.addEventListener('click', runOne);
runAllBtn.addEventListener('click', runAll);
resetBtn.addEventListener('click', reset);

speedInput.addEventListener('input', () => {
    speedValue.textContent = speedInput.value + 'ms';
});

// Initial Setup
window.onload = () => {
    initCharts();
    m = parseFloat(mInput.value);
    c = parseFloat(cInput.value);
    updateCharts();

    // Render initial LaTeX in the static parts of the page
    if (window.renderMathInElement) {
        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false }
            ]
        });
    }
};
