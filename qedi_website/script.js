document.addEventListener('DOMContentLoaded', () => {

    /* --- Hero Particle Reservoir Animation --- */
    const heroCanvas = document.getElementById('hero-particles');
    if (heroCanvas) {
        const ctx = heroCanvas.getContext('2d');
        let width, height;
        let particles = [];

        function resize() {
            width = heroCanvas.width = heroCanvas.parentElement.clientWidth;
            height = heroCanvas.height = heroCanvas.parentElement.clientHeight;
        }
        window.addEventListener('resize', resize);
        resize();

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 1.5;
                this.vy = (Math.random() - 0.5) * 1.5;
                this.color = Math.random() > 0.8 ? '#E3000F' : '#444'; // EPFL Red or Dark Gray
                this.radius = Math.random() * 2 + 1;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.fill();
            }
        }

        for (let i = 0; i < 80; i++) particles.push(new Particle());

        function animate() {
            ctx.clearRect(0, 0, width, height);

            for (let i = 0; i < particles.length; i++) {
                particles[i].update();
                particles[i].draw();

                // Draw connections
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < 120) {
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(227, 0, 15, ${1 - dist / 120})`;
                        if (particles[i].color === '#444' && particles[j].color === '#444') {
                            ctx.strokeStyle = `rgba(100, 100, 100, ${0.5 - (dist / 240)})`; // Faint gray
                        }
                        ctx.lineWidth = 1;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(animate);
        }
        animate();
    }


    /* --- Interactive QRC Data Imputer Sandbox --- */
    const sandboxCanvas = document.getElementById('draw-canvas');
    if (sandboxCanvas) {
        const ctx = sandboxCanvas.getContext('2d');
        const btnClear = document.getElementById('btn-clear');
        const btnImpute = document.getElementById('btn-impute');

        // Match internal resolution
        let width, height;
        function resizeSandbox() {
            width = sandboxCanvas.width = sandboxCanvas.parentElement.clientWidth;
            height = sandboxCanvas.height = 400;
        }
        window.addEventListener('resize', resizeSandbox);
        resizeSandbox();

        let isDrawing = false;
        let drawnSegments = []; // Array of {x, y} arrays
        let currentSegment = [];

        function renderBase() {
            ctx.clearRect(0, 0, width, height);

            // Draw grid lines
            ctx.strokeStyle = '#222';
            ctx.lineWidth = 1;
            for (let i = 1; i < 10; i++) {
                ctx.beginPath();
                ctx.moveTo(0, height * (i / 10));
                ctx.lineTo(width, height * (i / 10));
                ctx.stroke();
            }

            // Draw user segments
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 3;
            ctx.shadowBlur = 0;

            drawnSegments.forEach(seg => {
                if (seg.length > 1) {
                    ctx.beginPath();
                    ctx.moveTo(seg[0].x, seg[0].y);
                    for (let i = 1; i < seg.length; i++) ctx.lineTo(seg[i].x, seg[i].y);
                    ctx.stroke();
                }
            });

            if (currentSegment.length > 1) {
                ctx.beginPath();
                ctx.moveTo(currentSegment[0].x, currentSegment[0].y);
                for (let i = 1; i < currentSegment.length; i++) ctx.lineTo(currentSegment[i].x, currentSegment[i].y);
                ctx.stroke();
            }
        }

        renderBase();

        sandboxCanvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            const rect = sandboxCanvas.getBoundingClientRect();
            currentSegment = [{ x: e.clientX - rect.left, y: e.clientY - rect.top }];
        });

        sandboxCanvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const rect = sandboxCanvas.getBoundingClientRect();
            let newX = e.clientX - rect.left;

            // Ensure drawing only goes left-to-right to simulate time series
            const lastPoint = currentSegment[currentSegment.length - 1];
            if (newX > lastPoint.x) {
                currentSegment.push({ x: newX, y: e.clientY - rect.top });
                renderBase();
            }
        });

        sandboxCanvas.addEventListener('mouseup', () => {
            if (!isDrawing) return;
            isDrawing = false;
            drawnSegments.push(currentSegment);
            currentSegment = [];
        });

        sandboxCanvas.addEventListener('mouseleave', () => {
            if (!isDrawing) return;
            isDrawing = false;
            drawnSegments.push(currentSegment);
            currentSegment = [];
        });

        // The Quantum Imputation Magic
        btnImpute.addEventListener('click', () => {
            if (drawnSegments.length < 2) return; // Need at least a gap to impute

            // Sort segments by X coordinate
            drawnSegments.sort((a, b) => a[0].x - b[0].x);

            let imputePoints = [];

            // Connect the gaps smoothly using QRC simulation
            for (let i = 0; i < drawnSegments.length - 1; i++) {
                let endObj = drawnSegments[i][drawnSegments[i].length - 1];
                let startObj = drawnSegments[i + 1][0];

                // Gap exists! Simulate imputation
                let gapWidth = startObj.x - endObj.x;
                if (gapWidth > 5) { // Minimum 5px gap
                    let ctrl1X = endObj.x + gapWidth * 0.3;
                    let ctrl1Y = endObj.y;

                    let ctrl2X = startObj.x - gapWidth * 0.3;
                    let ctrl2Y = startObj.y;

                    imputePoints.push({
                        start: endObj,
                        end: startObj,
                        cp1: { x: ctrl1X, y: ctrl1Y },
                        cp2: { x: ctrl2X, y: ctrl2Y }
                    });
                }
            }

            // Animate Drawing
            let progress = 0;
            function animateImputation() {
                renderBase(); // Draw normal stuff

                ctx.strokeStyle = '#E3000F'; // EPFL Red
                ctx.lineWidth = 4;
                ctx.shadowColor = '#E3000F';
                ctx.shadowBlur = 10;

                imputePoints.forEach(curve => {
                    ctx.beginPath();
                    ctx.moveTo(curve.start.x, curve.start.y);

                    // Bezier interpolation limited by progress
                    for (let t = 0; t <= progress; t += 0.05) {
                        let _t = 1 - t;
                        let px = _t * _t * _t * curve.start.x + 3 * _t * _t * t * curve.cp1.x + 3 * _t * t * t * curve.cp2.x + t * t * t * curve.end.x;
                        let py = _t * _t * _t * curve.start.y + 3 * _t * _t * t * curve.cp1.y + 3 * _t * t * t * curve.cp2.y + t * t * t * curve.end.y;
                        ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                });

                if (progress < 1) {
                    progress += 0.02; // Speed of auto-regression
                    requestAnimationFrame(animateImputation);
                }
            }
            animateImputation();
        });

        btnClear.addEventListener('click', () => {
            drawnSegments = [];
            currentSegment = [];
            renderBase();
        });
    }

    /* --- Lightbox Functionality --- */
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxCaption = document.getElementById('lightbox-caption');
    const closeBtn = document.querySelector('.lightbox-close');

    if (lightbox) {
        document.querySelectorAll('.zoomable').forEach(img => {
            img.addEventListener('click', () => {
                lightbox.classList.add('show');
                lightboxImg.src = img.src;
                lightboxCaption.textContent = img.nextElementSibling.textContent;
            });
        });

        closeBtn.addEventListener('click', () => {
            lightbox.classList.remove('show');
        });

        // Click outside the image to close
        lightbox.addEventListener('click', (e) => {
            if (e.target !== lightboxImg) {
                lightbox.classList.remove('show');
            }
        });

        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === "Escape" && lightbox.classList.contains('show')) {
                lightbox.classList.remove('show');
            }
        });
    }

    /* --- Results Data Loading (Heatmap) --- */
    const plotHeatmap = document.getElementById('plot-heatmap');

    if (plotHeatmap) {
        fetch('assets/results_data.json')
            .then(response => response.json())
            .then(data => {
                // --- Draw Interactive Error Heatmap ---
                let errors = [];
                let maxError = 0;

                // Compute error array: (predicted - actual)
                for (let d = 0; d < data.pred_days; d++) {
                    let dayErrors = [];
                    for (let f = 0; f < data.features.length; f++) {
                        let err = data.predicted[d][f] - data.actual[d][f];
                        dayErrors.push(err);
                        if (Math.abs(err) > maxError) maxError = Math.abs(err);
                    }
                    errors.push(dayErrors);
                }

                const traceHeatmap = {
                    z: errors,
                    x: data.features,
                    y: Array.from({ length: data.pred_days }, (_, i) => `Day ${i + 1}`),
                    type: 'heatmap',
                    colorscale: 'RdBu',
                    zmin: -maxError,
                    zmax: maxError,
                    reversescale: true,
                    colorbar: { title: "Error" },
                    hovertemplate: 'Predictive Error: %{z}<br>Horizon: %{y}<br>Feature: %{x}<extra></extra>'
                };

                const layoutHeatmap = {
                    margin: { l: 60, r: 20, b: 60, t: 20 },
                    xaxis: {
                        title: 'Feature Space (224 Mapped Tenor & Maturity Swaptions)',
                        showticklabels: false
                    },
                    yaxis: { title: '' },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                };

                Plotly.newPlot('plot-heatmap', [traceHeatmap], layoutHeatmap, { responsive: true });
            })
            .catch(err => console.error("Error loading plot data:", err));
    }

    /* --- Animated PCA Trajectory Plot --- */
    const plotPcaAnim = document.getElementById('plot-pca-anim');
    const btnReplayAnim = document.getElementById('btn-replay-anim');

    if (plotPcaAnim) {
        fetch('assets/pca_anim_data.json')
            .then(response => response.json())
            .then(data => {
                const totalPoints = data.true_pc1.length;
                const xFrames = Array.from({ length: totalPoints }, (_, i) => i + 1);

                const halfIdx = Math.floor(totalPoints / 2);

                // Initialize the base plot
                // 1. The full target reality (faded grey, acts as the absolute background)
                const traceTrueBase = {
                    x: xFrames,
                    y: data.true_pc1,
                    mode: 'lines',
                    name: 'Target (Unknown Future)',
                    line: { color: 'rgba(68, 68, 68, 0.4)', width: 3 }
                };

                // 2. The known history ground truth (solid red for the first half, and grows dynamically)
                const traceKnownTruth = {
                    x: xFrames.slice(0, halfIdx),
                    y: data.true_pc1.slice(0, halfIdx),
                    mode: 'lines',
                    name: 'Known Market History',
                    line: { color: 'rgba(227, 0, 15, 1)', width: 3 }
                };

                // 3. The QRC Prediction (draws the next 6 days in green)
                const tracePredAnim = {
                    x: [xFrames[halfIdx - 1]],
                    y: [data.pred_pc1[halfIdx - 1]],
                    mode: 'lines',
                    name: 'QRC Prediction Engine',
                    line: { color: 'rgba(46, 204, 113, 1)', width: 4 }
                };

                const layout = {
                    margin: { l: 60, r: 20, b: 60, t: 40 },
                    xaxis: { title: 'Time (Days)', range: [0, totalPoints + 5] },
                    yaxis: {
                        title: 'Principal Component 1 (Swaption Variance)',
                        range: [
                            Math.min(...data.true_pc1, ...data.pred_pc1) - 1,
                            Math.max(...data.true_pc1, ...data.pred_pc1) + 1
                        ]
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    hovermode: 'x unified',
                    showlegend: true,
                    legend: { orientation: 'h', x: 0.5, y: 1.1, xanchor: 'center' }
                };

                Plotly.newPlot('plot-pca-anim', [traceTrueBase, traceKnownTruth, tracePredAnim], layout, { responsive: true });

                // Animation Engine
                let animReq;
                let animTimeout;
                function runAnimation() {
                    cancelAnimationFrame(animReq);
                    clearTimeout(animTimeout);

                    // Reset to initial state (halfIdx is the current "present day" anchor)
                    Plotly.update('plot-pca-anim', {
                        x: [xFrames, xFrames.slice(0, halfIdx), [xFrames[halfIdx - 1]]],
                        y: [data.true_pc1, data.true_pc1.slice(0, halfIdx), [data.pred_pc1[halfIdx - 1]]]
                    }, {}, [0, 1, 2]);

                    let anchorIdx = halfIdx - 1;

                    function animateStep() {
                        const stepSize = 6;
                        let nextAnchor = anchorIdx + stepSize;
                        if (nextAnchor >= totalPoints) nextAnchor = totalPoints - 1;

                        // Step 1: Predict 6 days into the future with Green Line (shooting out from the anchor)
                        Plotly.update('plot-pca-anim', {
                            x: [xFrames, xFrames.slice(0, anchorIdx + 1), xFrames.slice(anchorIdx, nextAnchor + 1)],
                            y: [data.true_pc1, data.true_pc1.slice(0, anchorIdx + 1), data.pred_pc1.slice(anchorIdx, nextAnchor + 1)]
                        }, {}, [0, 1, 2]);

                        // Step 2: Time moves forward (Wait 500ms, then the red line "Known History" catches up to the green prediction)
                        if (nextAnchor < totalPoints - 1) {
                            animTimeout = setTimeout(() => {
                                // Red line snaps forward 6 days (simulating time passing and the ground truth being revealed)
                                anchorIdx = nextAnchor;

                                // Erase the green line (since we are now at the new present day)
                                Plotly.update('plot-pca-anim', {
                                    x: [xFrames, xFrames.slice(0, anchorIdx + 1), [xFrames[anchorIdx]]],
                                    y: [data.true_pc1, data.true_pc1.slice(0, anchorIdx + 1), [data.pred_pc1[anchorIdx]]]
                                }, {}, [0, 1, 2]);

                                // Wait a beat, then predict the next 6 days
                                animTimeout = setTimeout(() => {
                                    animReq = requestAnimationFrame(animateStep);
                                }, 200);

                            }, 800); // How long the green prediction stays on screen
                        } else {
                            // Final frame has been reached, lock the red line to the end eventually
                            animTimeout = setTimeout(() => {
                                Plotly.update('plot-pca-anim', {
                                    x: [xFrames, xFrames.slice(0, nextAnchor + 1), []],
                                    y: [data.true_pc1, data.true_pc1.slice(0, nextAnchor + 1), []]
                                }, {}, [0, 1, 2]);
                            }, 800);
                        }
                    }

                    // Start the rolling forward-prediction animation
                    animTimeout = setTimeout(() => {
                        animReq = requestAnimationFrame(animateStep);
                    }, 500);
                }

                // Autoplay once loaded
                setTimeout(runAnimation, 500);

                if (btnReplayAnim) {
                    btnReplayAnim.addEventListener('click', runAnimation);
                }
            })
            .catch(err => console.error("Error loading PCA animation data:", err));
    }

    /* --- Interactive 2D MAE Surface Heatmap --- */
    const plotMaeSurface = document.getElementById('plot-mae-surface');
    if (plotMaeSurface) {
        fetch('assets/mae_surface.json')
            .then(response => response.json())
            .then(data => {
                const traceMae = {
                    z: data.surface,
                    x: data.maturities.map(m => m < 1 ? m.toFixed(2) : m.toFixed(1)),
                    y: data.tenors.map(t => t + 'Y'),
                    type: 'heatmap',
                    colorscale: 'Plasma',
                    zmin: 0, // strict 0 minimum for absolute error scaling
                    zsmooth: 'best', // exactly mimics interpolation="bilinear" from python
                    colorbar: { title: "MAE" },
                    hovertemplate: 'Maturity: %{x} Years<br>Tenor: %{y}<br>Mean Absolute Error: %{z:.5f}<extra></extra>'
                };

                const layoutMae = {
                    margin: { l: 60, r: 20, b: 60, t: 40 },
                    xaxis: {
                        title: 'Maturity (years)',
                        tickangle: -45,
                        type: 'category'
                    },
                    yaxis: {
                        title: 'Tenor (years)',
                        autorange: true,
                        type: 'category',
                        scaleanchor: 'x',    // Locks the y-axis physical length to match the x-axis
                        scaleratio: 1        // Forces exactly 1:1 perfect square aspect ratio per cell
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                };

                Plotly.newPlot('plot-mae-surface', [traceMae], layoutMae, { responsive: true });
            })
    }

    /* --- Interactive Model Performance Leaderboard --- */
    const plotModelComp = document.getElementById('plot-model-comparison');
    if (plotModelComp) {
        const rowLabels = [
            'Classical LSTM',
            'QSVR (Quantum SVR)',
            'Hybrid QNN',
            'Photonic Linear QRC',
            'Hybrid Photonic Linear QRC',
            '<b>🥇 Hybrid Photonic Temporal QRC</b>'
        ];

        // Data populated from experiment_logs.json + final_model.py + temporal_qrc.py outputs
        const mses = ['5.26e-05', '5.43e-04', '6.85e-05', '4.22e-05', '7.70e-06', '<b>7.58e-06</b>'];
        const rses = ['0.00725', '0.02330', '0.00827', '0.00650', '0.00277', '<b>0.00275</b>']; // RMSE = sqrt(MSE)
        const maes = ['0.00518', '0.01866', '0.00589', '0.00450', '0.00214', '<b>0.00195</b>']; // Temporal QRC MAE average

        const traceTable = {
            type: 'table',
            header: {
                values: [
                    "<b>Model Architecture</b>",
                    "<b>Mean Squared Error (MSE)</b>",
                    "<b>Root Mean Squared (RMSE)</b>",
                    "<b>Mean Absolute Error (MAE)</b>"
                ],
                align: "center",
                line: { width: 1, color: '#e2e8f0' },
                fill: { color: "#1e293b" }, // Dark Slate background
                font: { family: "Inter, sans-serif", size: 16, color: "white" },
                height: 50
            },
            cells: {
                values: [rowLabels, mses, rses, maes],
                align: "center",
                line: { color: "#e2e8f0", width: 1 },
                fill: {
                    color: [
                        // Column colors (rows 1-5: light grey, row 6: green highlight)
                        ['#f8fafc', '#f8fafc', '#f8fafc', '#f8fafc', '#f8fafc', 'rgba(46, 204, 113, 0.15)'], // Col 1
                        ['#f8fafc', '#f8fafc', '#f8fafc', '#f8fafc', '#f8fafc', 'rgba(46, 204, 113, 0.15)'], // Col 2
                        ['#f8fafc', '#f8fafc', '#f8fafc', '#f8fafc', '#f8fafc', 'rgba(46, 204, 113, 0.15)'], // Col 3
                        ['#f8fafc', '#f8fafc', '#f8fafc', '#f8fafc', '#f8fafc', 'rgba(46, 204, 113, 0.15)'], // Col 4
                    ]
                },
                font: { family: "Inter, sans-serif", size: 15, color: "#0f172a" },
                height: 45
            }
        };

        const layoutTable = {
            margin: { l: 10, r: 10, b: 10, t: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };

        Plotly.newPlot('plot-model-comparison', [traceTable], layoutTable, { responsive: true });
    }

    /* --- Imputer Toggle Button --- */
    const toggleImputerBtn = document.getElementById('toggle-imputer-btn');
    const imputerContent = document.getElementById('imputer-content');

    if (toggleImputerBtn && imputerContent) {
        toggleImputerBtn.addEventListener('click', () => {
            if (imputerContent.style.display === 'none') {
                imputerContent.style.display = 'block';
                // Trigger a resize to correctly recalculate sandbox canvas dimensions now that parent is visible
                window.dispatchEvent(new Event('resize'));
                setTimeout(() => {
                    imputerContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            } else {
                imputerContent.style.display = 'none';
            }
        });
    }

    /* --- Pipeline Animation Click Zoom --- */
    const pipelineAnim = document.getElementById('pipeline-anim-container');
    if (pipelineAnim) {
        pipelineAnim.addEventListener('click', () => {
            pipelineAnim.classList.toggle('window-expanded');
        });
    }

    /* --- Photonic Temporal QRC Circuit Animation --- */
    const qrcCanvas = document.getElementById('qrc-circuit-anim');
    if (qrcCanvas) {
        const ctx = qrcCanvas.getContext('2d');
        let t = 0;

        const N_MODES = 8;
        const N_INPUT = 5;
        const N_STEPS = 5; // Temporal window elements
        const N_VIRT = 3;  // Virtual node depths

        function drawCircuit() {
            // Resize handler inside drawing for flex/responsiveness
            const w = qrcCanvas.width = qrcCanvas.offsetWidth || 400;
            const h = qrcCanvas.height = qrcCanvas.offsetHeight || 180;
            ctx.clearRect(0, 0, w, h);

            const modeSpacing = h / (N_MODES + 1);

            // Draw Mode lines
            ctx.lineWidth = 1;
            for (let m = 0; m < N_MODES; m++) {
                const y = (m + 1) * modeSpacing;
                // Distinction: Input modes (0-4) are blue-ish, Memory modes (5-7) are purple-ish
                ctx.strokeStyle = m < N_INPUT ? 'rgba(52, 152, 219, 0.3)' : 'rgba(155, 89, 182, 0.3)';
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();

                // Labels
                ctx.fillStyle = m < N_INPUT ? 'rgba(52, 152, 219, 0.8)' : 'rgba(155, 89, 182, 0.8)';
                ctx.font = '9px Inter';
                ctx.fillText(m < N_INPUT ? 'In' : 'Mem', 5, y - 3);
            }

            const totalSegments = N_STEPS + N_VIRT;
            const segmentWidth = w / (totalSegments + 1);

            // Draw mixing interferometers & phase shifters
            for (let s = 0; s < totalSegments; s++) {
                const cx = (s + 1) * segmentWidth;

                // 1. Universal Mixing Interferometer (All 8 modes)
                ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
                ctx.beginPath();
                ctx.rect(cx - 8, modeSpacing - 5, 16, (N_MODES - 1) * modeSpacing + 10);
                ctx.fill();
                ctx.stroke();

                // Draw internal entanglement webs
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                for (let i = 0; i < 3; i++) {
                    ctx.moveTo(cx - 8, (1 + Math.random() * (N_MODES - 1)) * modeSpacing);
                    ctx.lineTo(cx + 8, (1 + Math.random() * (N_MODES - 1)) * modeSpacing);
                }
                ctx.stroke();

                // 2. Phase Shifters (ONLY if it's a temporal step, and ONLY on top 5 modes)
                if (s < N_STEPS) {
                    for (let m = 0; m < N_INPUT; m++) {
                        ctx.fillStyle = 'rgba(46, 204, 113, 0.8)';
                        ctx.beginPath();
                        ctx.arc(cx + 12, (m + 1) * modeSpacing, 3, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    // Step Labels
                    ctx.fillStyle = 'rgba(255,255,255,0.5)';
                    ctx.font = '8px Inter';
                    ctx.fillText(`T${s + 1}`, cx - 5, h - 5);
                } else {
                    // Virtual Depth Labels
                    ctx.fillStyle = 'rgba(231, 76, 60, 0.5)';
                    ctx.font = '8px Inter';
                    ctx.fillText(`V${s - N_STEPS + 1}`, cx - 5, h - 5);
                }
            }

            // Draw traveling photons
            // 3 Photons moving rightwards
            const speed = 0.005;
            const px = (t * speed * w) % (w + 50) - 25; // Loop across screen

            // Map photon position to closest column to figure out random jumps
            let currentSegment = Math.floor(px / segmentWidth);

            for (let p = 0; p < 3; p++) {
                // Pseudo-random y position based on time and photon ID to simulate interference mixing
                let pseudoRand = Math.sin(currentSegment * 12.3 + p * 4.5);
                let targetMode = Math.floor((pseudoRand + 1) / 2 * N_MODES);

                // Smoothing transition between modes
                let modeFrac = (px % segmentWidth) / segmentWidth;
                let lastMode = Math.floor((Math.sin((currentSegment - 1) * 12.3 + p * 4.5) + 1) / 2 * N_MODES);

                // If it's entering the first mixer, originate from top 5 (input limits)
                if (currentSegment <= 0) { lastMode = p % N_INPUT; targetMode = lastMode; }

                let renderMode = lastMode + (targetMode - lastMode) * (Math.sin(modeFrac * Math.PI - Math.PI / 2) * 0.5 + 0.5);
                let py = (renderMode + 1) * modeSpacing;

                ctx.shadowBlur = 10;
                ctx.shadowColor = 'rgba(231, 76, 60, 1)';
                ctx.fillStyle = '#ff7675';
                ctx.beginPath();
                ctx.arc(px, py, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.shadowBlur = 0;
            }

            t++;
            requestAnimationFrame(drawCircuit);
        }

        drawCircuit();
    }

});
