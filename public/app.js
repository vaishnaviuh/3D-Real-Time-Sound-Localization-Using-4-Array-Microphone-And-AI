import * as THREE from 'three';

(() => {
  const statusEl = document.getElementById('status');
  const detectionStatusEl = document.getElementById('detectionStatus');
  const posEl = document.getElementById('pos');
  const angEl = document.getElementById('ang');
  const energyEl = document.getElementById('energy');
  const canvas = document.getElementById('canvas2d');
  const ctx = canvas.getContext('2d');
  const root3d = document.getElementById('threeRoot');

  let ws;
  let lastPoint = null;

  // 2D drawing helpers
  function worldToCanvas(x, y) {
    // Map world meters to canvas pixels (scale to fit)
    const scale = Math.min(canvas.width, canvas.height) / 2.5; // +/-1.25m
    const cx = canvas.width / 2 + x * scale;
    const cy = canvas.height / 2 - y * scale;
    return [cx, cy];
  }

  function draw2D() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // grid
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    for (let i = -1; i <= 1; i += 0.25) {
      const [x1, y1] = worldToCanvas(-1.25, i);
      const [x2, y2] = worldToCanvas(1.25, i);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      const [x3, y3] = worldToCanvas(i, -1.25);
      const [x4, y4] = worldToCanvas(i, 1.25);
      ctx.beginPath(); ctx.moveTo(x3, y3); ctx.lineTo(x4, y4); ctx.stroke();
    }
    // axes
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    // X axis
    let [xa1, ya1] = worldToCanvas(-1.25, 0);
    let [xa2, ya2] = worldToCanvas(1.25, 0);
    ctx.beginPath(); ctx.moveTo(xa1, ya1); ctx.lineTo(xa2, ya2); ctx.stroke();
    // Y axis
    let [yaX1, yaY1] = worldToCanvas(0, -1.25);
    let [yaX2, yaY2] = worldToCanvas(0, 1.25);
    ctx.beginPath(); ctx.moveTo(yaX1, yaY1); ctx.lineTo(yaX2, yaY2); ctx.stroke();
    // axis labels
    ctx.fillStyle = '#0f172a';
    ctx.font = '12px sans-serif';
    const [xlabX, xlabY] = worldToCanvas(1.2, 0.02);
    ctx.fillText('X', xlabX, xlabY);
    const [ylabX, ylabY] = worldToCanvas(0.02, 1.2);
    ctx.fillText('Y', ylabX, ylabY);
    // microphones (square ~4.5cm edge -> show schematic at +/-0.0225m)
    const micEdge = 0.0225;
    const micPositions = [
      [micEdge, micEdge],
      [-micEdge, micEdge],
      [-micEdge, -micEdge],
      [micEdge, -micEdge],
    ];
    ctx.fillStyle = '#111827';
    micPositions.forEach(([mx, my]) => {
      const [cx, cy] = worldToCanvas(mx, my);
      ctx.beginPath(); ctx.rect(cx - 6, cy - 6, 12, 12); ctx.fill();
    });
    
    // current point (red marker)
    if (lastPoint) {
      const p = lastPoint;
      const [cx, cy] = worldToCanvas(p.x, p.y);
      ctx.fillStyle = '#ef4444';
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
      // draw x,y,z label near the marker
      ctx.fillStyle = '#0f172a';
      ctx.font = '12px sans-serif';
      const label = `x:${p.x.toFixed(3)}  y:${p.y.toFixed(3)}  z:${p.z.toFixed(3)} m`;
      ctx.fillText(label, cx + 10, cy - 10);
    }
  }

  // 3D scene (Three.js)
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, root3d.clientWidth / root3d.clientHeight, 0.01, 20);
  camera.position.set(1.6, 1.2, 1.6);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(root3d.clientWidth, root3d.clientHeight);
  root3d.appendChild(renderer.domElement);
  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(1, 1, 1);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  // ground
  const grid = new THREE.GridHelper(4.0, 40, 0xcccccc, 0x444444);
  scene.add(grid);
  // axes helper (X=red, Y=green, Z=blue)
  const axes = new THREE.AxesHelper(0.5);
  scene.add(axes);
  // microphones
  const micGeo = new THREE.BoxGeometry(0.03, 0.03, 0.01);
  const micMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
  const micEdge3d = 0.0225;
  const mic3d = [
    new THREE.Mesh(micGeo, micMat), // top-right
    new THREE.Mesh(micGeo, micMat), // top-left
    new THREE.Mesh(micGeo, micMat), // bottom-left
    new THREE.Mesh(micGeo, micMat), // bottom-right
  ];
  mic3d[0].position.set(micEdge3d, micEdge3d, 0);
  mic3d[1].position.set(-micEdge3d, micEdge3d, 0);
  mic3d[2].position.set(-micEdge3d, -micEdge3d, 0);
  mic3d[3].position.set(micEdge3d, -micEdge3d, 0);
  mic3d.forEach(m => scene.add(m));
  // current point
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.03, 32, 32),
    new THREE.MeshStandardMaterial({ color: 0xef4444 })
  );
  scene.add(sphere);

  function render3D() {
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }

  function updateUI(p) {
    posEl.textContent = `(${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}) m`;
    angEl.textContent = `Azimuth ${p.azimuth.toFixed(1)}°, Elevation ${p.elevation.toFixed(1)}°`;
    const hEl = document.getElementById('height');
    if (hEl) hEl.textContent = `${(p.height ?? p.z).toFixed(3)} m`;
    const dEl = document.getElementById('distance');
    if (dEl && p.distance !== undefined) dEl.textContent = `${p.distance.toFixed(3)} m`;
    const body = document.getElementById('tdoaBody');
    if (body) {
      const pairs = p.tdoa_pairs || [];
      let rows = '';
      for (const row of pairs) {
        rows += `<tr>
          <td style="padding:6px;">${row.ch1}</td>
          <td style="padding:6px;">${row.ch2}</td>
          <td style="padding:6px; text-align:right;">${row.tdoa.toFixed(6)}</td>
        </tr>`;
      }
      body.innerHTML = rows;
    }
    // 3D overlay coords
    const overlay = document.getElementById('coord3d');
    if (overlay) {
      overlay.textContent = `x: ${p.x.toFixed(3)}  y: ${p.y.toFixed(3)}  z: ${p.z.toFixed(3)} m`;
    }
  }

  // WebSocket
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => { 
      statusEl.textContent = 'Connected'; 
      console.log('WebSocket connected - ready for updates');
    };
    ws.onclose = () => { 
      statusEl.textContent = 'Disconnected'; 
      console.log('WebSocket disconnected');
    };
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    ws.onmessage = (e) => {
      console.log('Received update'); // Debug log
      try {
        const p = JSON.parse(e.data);
        // Update detection status
        if (detectionStatusEl) {
          if (p.status === 'tracking') {
            detectionStatusEl.textContent = '✓ ' + (p.message || 'Source detected and tracking');
            detectionStatusEl.style.color = '#10b981';
          } else if (p.status === 'no_signal') {
            detectionStatusEl.textContent = '✗ ' + (p.message || 'No signal detected');
            detectionStatusEl.style.color = '#ef4444';
          } else if (p.status === 'triangulation_failed') {
            detectionStatusEl.textContent = '⚠ ' + (p.message || 'TDOA detected but triangulation failed');
            detectionStatusEl.style.color = '#f59e0b';
          } else if (p.status === 'tdoa_invalid') {
            detectionStatusEl.textContent = '⚠ ' + (p.message || 'TDOA value too large');
            detectionStatusEl.style.color = '#f59e0b';
          } else if (p.status === 'no_audio') {
            detectionStatusEl.textContent = '✗ ' + (p.message || 'No audio input detected');
            detectionStatusEl.style.color = '#ef4444';
          } else {
            detectionStatusEl.textContent = 'Waiting for signal...';
            detectionStatusEl.style.color = '#6b7280';
          }
        }
        // Update energy levels (show raw if available, otherwise filtered)
        if (energyEl) {
          const rawAvg = p.raw_energy_avg !== undefined ? p.raw_energy_avg : (p.energy_avg || 0);
          const rawMax = p.raw_energy_max !== undefined ? p.raw_energy_max : (p.energy_max || 0);
          const filtAvg = p.energy_avg || 0;
          const filtMax = p.energy_max || 0;
          
          let energyText = `Raw: avg=${rawAvg.toFixed(8)}, max=${rawMax.toFixed(8)}`;
          if (filtAvg > 0 || filtMax > 0) {
            energyText += ` | Filtered: avg=${filtAvg.toFixed(8)}, max=${filtMax.toFixed(8)}`;
          }
          
          // Show channel energies if available
          if (p.raw_channel_energies && p.raw_channel_energies.length > 0) {
            const chEnergies = p.raw_channel_energies.map((e, i) => `Ch${i}:${e.toFixed(6)}`).join(' ');
            energyText += ` | [${chEnergies}]`;
          }
          
          energyEl.textContent = energyText;
          
          // Color code: green if raw energy above threshold, red if below
          if (rawAvg > 0.000001) {
            energyEl.style.color = '#10b981';
          } else if (rawAvg > 1e-10) {
            energyEl.style.color = '#f59e0b';
          } else {
            energyEl.style.color = '#ef4444';
          }
        }
        // Update if we have position data - but only if position changed significantly
        if (p.x !== undefined && p.y !== undefined && p.z !== undefined) {
          // Check if position changed significantly (dead zone to prevent jitter)
          const posChanged = !lastPoint || 
            Math.abs(p.x - lastPoint.x) > 0.005 ||  // 5mm threshold
            Math.abs(p.y - lastPoint.y) > 0.005 ||
            Math.abs(p.z - lastPoint.z) > 0.005;
          
          if (posChanged) {
            lastPoint = p;
            
            // Redraw only when position changed
            draw2D();
            sphere.position.set(p.x, p.y, p.z);
            render3D();
            updateUI(p);
          } else {
            // Position didn't change much - just update UI (for status/energy)
            updateUI(p);
          }
        } else if (p.status === 'waiting' && lastPoint) {
          // Keep showing last position even when waiting for signal
          draw2D();
          sphere.position.set(lastPoint.x, lastPoint.y, lastPoint.z);
          render3D();
          if (detectionStatusEl) {
            detectionStatusEl.textContent = '⏳ ' + (p.message || 'Waiting for signal...');
            detectionStatusEl.style.color = '#6b7280';
          }
        }
      } catch (_) {}
    };
  }

  // resize handler
  window.addEventListener('resize', () => {
    renderer.setSize(root3d.clientWidth, root3d.clientHeight);
    camera.aspect = root3d.clientWidth / root3d.clientHeight;
    camera.updateProjectionMatrix();
    render3D();
  });

  document.getElementById('btnStart').addEventListener('click', async () => {
    await fetch('/start');
    if (!ws || ws.readyState !== WebSocket.OPEN) connect();
  });
  document.getElementById('btnStop').addEventListener('click', async () => {
    await fetch('/stop');
    if (ws) ws.close();
  });
  

  // initial draw
  draw2D();
  render3D();
})();

