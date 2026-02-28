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
                lightboxCaption.innerText = img.nextElementSibling.innerText;
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

    /* --- Interactive 3D Plotly Swaption Surfaces --- */
    const plotActual = document.getElementById('plot-actual');
    const plotPredicted = document.getElementById('plot-predicted');

    if (plotActual && plotPredicted) {
        fetch('assets/results_data.json')
            .then(response => response.json())
            .then(data => {
                // Parse distinct Tenors and Maturities to build the grid
                const uniqueTenors = [...new Set(data.tenors)].sort((a, b) => a - b);
                const uniqueMaturities = [...new Set(data.maturities)].sort((a, b) => a - b);

                // Helper function to build 2D Z-matrix from flattened 224D array
                function buildZMatrix(dayIndex, sourceArray) {
                    const row = sourceArray[dayIndex];
                    let zMatrix = [];
                    for (let t = 0; t < uniqueTenors.length; t++) {
                        let zRow = [];
                        for (let m = 0; m < uniqueMaturities.length; m++) {
                            // Find the index in the flattened 224 array
                            const flatIndex = data.tenors.findIndex((ten, idx) => ten === uniqueTenors[t] && data.maturities[idx] === uniqueMaturities[m]);
                            zRow.push(flatIndex >= 0 ? row[flatIndex] : null);
                        }
                        zMatrix.push(zRow);
                    }
                    return zMatrix;
                }

                // Plotting Function
                function drawPlots(day) {
                    // Update buttons
                    document.querySelectorAll('.btn-secondary').forEach(b => b.classList.remove('active'));
                    document.getElementById(`btn-day${day}`).classList.add('active');

                    // Array indexing (Day 1 -> index 0)
                    const dayIndex = day - 1;
                    const zActual = buildZMatrix(dayIndex, data.actual);
                    const zPredicted = buildZMatrix(dayIndex, data.predicted);

                    const layoutTemplate = {
                        margin: { l: 0, r: 0, b: 0, t: 20 },
                        scene: {
                            xaxis: { title: 'Maturity (Years)', gridcolor: '#ddd' },
                            yaxis: { title: 'Tenor (Years)', gridcolor: '#ddd' },
                            zaxis: { title: 'Implied Volatility', gridcolor: '#ddd' },
                            camera: { eye: { x: 1.5, y: 1.5, z: 0.5 } }
                        },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };

                    const traceActual = {
                        z: zActual,
                        x: uniqueMaturities,
                        y: uniqueTenors,
                        type: 'surface',
                        colorscale: 'Viridis',
                        showscale: false
                    };

                    const tracePredicted = {
                        z: zPredicted,
                        x: uniqueMaturities,
                        y: uniqueTenors,
                        type: 'surface',
                        colorscale: 'RdBu',
                        showscale: false
                    };

                    Plotly.newPlot('plot-actual', [traceActual], Object.assign({}, layoutTemplate, { title: `Day ${day} Ground Truth` }), { responsive: true });
                    Plotly.newPlot('plot-predicted', [tracePredicted], Object.assign({}, layoutTemplate, { title: `Day ${day} MerLin QRC Prediction` }), { responsive: true });
                }

                // Initial Draw
                drawPlots(1);

                // Button Listeners
                document.getElementById('btn-day1').addEventListener('click', () => drawPlots(1));
                document.getElementById('btn-day3').addEventListener('click', () => drawPlots(3));
                document.getElementById('btn-day6').addEventListener('click', () => drawPlots(6));

                // --- Draw Interactive Error Heatmap ---
                const plotHeatmap = document.getElementById('plot-heatmap');
                if (plotHeatmap) {
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
                }

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

                // Initialize the base plot with full TRUE background and empty PRED
                const traceTrue = {
                    x: xFrames,
                    y: data.true_pc1,
                    mode: 'lines',
                    name: 'Market Reality (Target)',
                    line: { color: 'rgba(68, 68, 68, 0.4)', width: 3 } // Dark slate, faded background
                };

                const tracePred = {
                    x: [xFrames[0]],
                    y: [data.pred_pc1[0]],
                    mode: 'lines',
                    name: 'MerLin Tracking',
                    line: { color: 'rgba(227, 0, 15, 1)', width: 3 } // EPFL red
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

                Plotly.newPlot('plot-pca-anim', [traceTrue, tracePred], layout, { responsive: true });

                // Animation Engine
                let animReq;
                function runAnimation() {
                    // Reset
                    cancelAnimationFrame(animReq);
                    Plotly.update('plot-pca-anim', { x: [xFrames, [xFrames[0]]], y: [data.true_pc1, [data.pred_pc1[0]]] });

                    let frameIdx = 1;
                    function animateStep() {
                        const stepSize = 1; // 1 day per frame = ~3-4x slower
                        frameIdx += stepSize;
                        if (frameIdx >= totalPoints) frameIdx = totalPoints - 1;

                        Plotly.update('plot-pca-anim', {
                            x: [xFrames, xFrames.slice(0, frameIdx)],
                            y: [data.true_pc1, data.pred_pc1.slice(0, frameIdx)]
                        }, {}, [0, 1]);

                        if (frameIdx < totalPoints - 1) {
                            setTimeout(() => {
                                animReq = requestAnimationFrame(animateStep);
                            }, 10); // Adds a slight delay to ensure it stays consistently slow
                        }
                    }
                    animReq = requestAnimationFrame(animateStep);
                }

                // Autoplay once loaded
                setTimeout(runAnimation, 500);

                if (btnReplayAnim) {
                    btnReplayAnim.addEventListener('click', runAnimation);
                }
            })
            .catch(err => console.error("Error loading PCA animation data:", err));
    }

});
