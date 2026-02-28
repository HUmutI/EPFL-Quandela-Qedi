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

});
