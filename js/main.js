// AI学习教程网站 - 主JavaScript文件

document.addEventListener('DOMContentLoaded', function() {
    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // 滚动时导航栏效果
    const header = document.querySelector('.header');
    if (header) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                header.style.boxShadow = '0 2px 20px rgba(0,0,0,0.2)';
            } else {
                header.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
            }
        });
    }

    // 代码高亮初始化
    initCodeHighlight();
});

// 简单的代码高亮功能
function initCodeHighlight() {
    document.querySelectorAll('.code-block pre').forEach(block => {
        let html = block.innerHTML;
        
        // Python关键字
        const keywords = ['import', 'from', 'def', 'class', 'return', 'if', 'else', 'elif', 
                         'for', 'while', 'in', 'not', 'and', 'or', 'True', 'False', 'None',
                         'try', 'except', 'finally', 'with', 'as', 'lambda', 'pass', 'break',
                         'continue', 'global', 'nonlocal', 'assert', 'yield', 'raise'];
        
        keywords.forEach(kw => {
            const regex = new RegExp(`\\b${kw}\\b`, 'g');
            html = html.replace(regex, `<span class="keyword">${kw}</span>`);
        });
        
        // 注释
        html = html.replace(/(#.*)$/gm, '<span class="comment">$1</span>');
        
        // 字符串
        html = html.replace(/(".*?"|'.*?')/g, '<span class="string">$1</span>');
        
        // 数字
        html = html.replace(/\b(\d+\.?\d*)\b/g, '<span class="number">$1</span>');
        
        block.innerHTML = html;
    });
}

// 线性回归可视化演示
function LinearRegressionDemo(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // 生成示例数据
    let points = [];
    let w = 0, b = 0; // 模型参数
    let learningRate = 0.01;
    let animationId;
    
    function generateData() {
        points = [];
        for (let i = 0; i < 30; i++) {
            const x = Math.random() * 8 + 1;
            const y = 2 * x + 3 + (Math.random() - 0.5) * 4;
            points.push({ x, y });
        }
    }
    
    function draw() {
        ctx.clearRect(0, 0, width, height);
        
        // 坐标轴
        ctx.strokeStyle = '#ddd';
        ctx.beginPath();
        ctx.moveTo(50, 10);
        ctx.lineTo(50, height - 30);
        ctx.lineTo(width - 10, height - 30);
        ctx.stroke();
        
        // 绘制数据点
        ctx.fillStyle = '#4a90d9';
        points.forEach(p => {
            const px = 50 + (p.x / 10) * (width - 60);
            const py = height - 30 - (p.y / 25) * (height - 40);
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill();
        });
        
        // 绘制回归线
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 2;
        ctx.beginPath();
        const x1 = 0, y1 = w * x1 + b;
        const x2 = 10, y2 = w * x2 + b;
        ctx.moveTo(50 + (x1 / 10) * (width - 60), height - 30 - (y1 / 25) * (height - 40));
        ctx.lineTo(50 + (x2 / 10) * (width - 60), height - 30 - (y2 / 25) * (height - 40));
        ctx.stroke();
        
        // 显示参数
        ctx.fillStyle = '#333';
        ctx.font = '14px sans-serif';
        ctx.fillText(`y = ${w.toFixed(2)}x + ${b.toFixed(2)}`, width - 150, 30);
    }
    
    function gradientDescent() {
        let dw = 0, db = 0;
        points.forEach(p => {
            const pred = w * p.x + b;
            const error = pred - p.y;
            dw += error * p.x;
            db += error;
        });
        dw /= points.length;
        db /= points.length;
        
        w -= learningRate * dw;
        b -= learningRate * db;
    }
    
    function animate() {
        gradientDescent();
        draw();
        animationId = requestAnimationFrame(animate);
    }
    
    // 初始化
    generateData();
    w = 0;
    b = 0;
    draw();
    
    // 绑定控制按钮
    const startBtn = document.getElementById('startDemo');
    const resetBtn = document.getElementById('resetDemo');
    
    if (startBtn) {
        startBtn.addEventListener('click', function() {
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
                this.textContent = '开始训练';
            } else {
                animate();
                this.textContent = '暂停';
            }
        });
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
                startBtn.textContent = '开始训练';
            }
            generateData();
            w = 0;
            b = 0;
            draw();
        });
    }
}

// 神经网络可视化演示
function NeuralNetworkDemo(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '300');
    svg.setAttribute('viewBox', '0 0 600 300');
    container.appendChild(svg);
    
    const layers = [3, 4, 4, 2]; // 网络结构
    const nodeRadius = 20;
    const layerGap = 150;
    const startX = 75;
    
    // 绘制连接线
    for (let l = 0; l < layers.length - 1; l++) {
        for (let i = 0; i < layers[l]; i++) {
            for (let j = 0; j < layers[l + 1]; j++) {
                const x1 = startX + l * layerGap;
                const y1 = 150 - ((layers[l] - 1) * 30) / 2 + i * 30;
                const x2 = startX + (l + 1) * layerGap;
                const y2 = 150 - ((layers[l + 1] - 1) * 30) / 2 + j * 30;
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', x1);
                line.setAttribute('y1', y1);
                line.setAttribute('x2', x2);
                line.setAttribute('y2', y2);
                line.setAttribute('stroke', '#ddd');
                line.setAttribute('stroke-width', '1');
                svg.appendChild(line);
            }
        }
    }
    
    // 绘制节点
    for (let l = 0; l < layers.length; l++) {
        for (let i = 0; i < layers[l]; i++) {
            const x = startX + l * layerGap;
            const y = 150 - ((layers[l] - 1) * 30) / 2 + i * 30;
            
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', nodeRadius / 2);
            circle.setAttribute('fill', l === 0 ? '#4a90d9' : l === layers.length - 1 ? '#e74c3c' : '#2ecc71');
            circle.setAttribute('stroke', '#fff');
            circle.setAttribute('stroke-width', '2');
            svg.appendChild(circle);
        }
    }
    
    // 添加层标签
    const labels = ['输入层', '隐藏层1', '隐藏层2', '输出层'];
    for (let l = 0; l < layers.length; l++) {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', startX + l * layerGap);
        text.setAttribute('y', 280);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '12');
        text.setAttribute('fill', '#666');
        text.textContent = labels[l];
        svg.appendChild(text);
    }
}

// 激活函数可视化
function ActivationFunctionDemo(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    function relu(x) {
        return Math.max(0, x);
    }
    
    function tanh_func(x) {
        return Math.tanh(x);
    }
    
    function draw(func, color, name) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let px = 0; px < width; px++) {
            const x = (px - width / 2) / 50;
            const y = func(x);
            const py = height / 2 - y * 50;
            
            if (px === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();
    }
    
    // 绘制坐标轴
    ctx.strokeStyle = '#ddd';
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();
    
    // 绘制三种激活函数
    draw(sigmoid, '#4a90d9', 'Sigmoid');
    draw(relu, '#e74c3c', 'ReLU');
    draw(tanh_func, '#2ecc71', 'Tanh');
    
    // 图例
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#4a90d9';
    ctx.fillText('Sigmoid', 10, 20);
    ctx.fillStyle = '#e74c3c';
    ctx.fillText('ReLU', 10, 35);
    ctx.fillStyle = '#2ecc71';
    ctx.fillText('Tanh', 10, 50);
}
