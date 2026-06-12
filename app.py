# ============================================================
# TSLA FORECASTING HUB  |  app.py
# Model: CNN-GRU + Hurst Regime Detection + OU Mean-Reversion
# ============================================================

import os
import re
import warnings
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def clean_html(html_code: str) -> str:
    """Compacts HTML code into a single-line string with comments removed, 
    completely preventing Streamlit's markdown parser from falsely wrapping 
    multiline indented HTML segments as markdown blocks."""
    if not html_code:
        return ""
    # Strip HTML comments to avoid parser confusion
    html_code = re.sub(r'<!--.*?-->', '', html_code, flags=re.DOTALL)
    # Replace all kinds of newlines with spaces
    html_code = html_code.replace("\n", " ").replace("\r", " ")
    # Compact any multiple consecutive whitespace characters into a single space
    html_code = re.sub(r'\s+', ' ', html_code)
    return html_code.strip()

import streamlit.delta_generator
_original_dg_markdown = streamlit.delta_generator.DeltaGenerator.markdown
def safe_dg_markdown(self, body, *args, **kwargs):
    if isinstance(body, str) and body.strip().startswith("<"):
        body = clean_html(body)
        kwargs["unsafe_allow_html"] = True
    return _original_dg_markdown(self, body, *args, **kwargs)
streamlit.delta_generator.DeltaGenerator.markdown = safe_dg_markdown

warnings.filterwarnings("ignore")

secret_url = ""
try:
    if "model_config" in st.secrets and "gdrive_model_link" in st.secrets["model_config"]:
        secret_url = st.secrets["model_config"]["gdrive_model_link"]
except Exception:
    pass

def get_secret_model_link() -> str:
    return secret_url

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Hybrid Forecast Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── WebGL Ghost Cursor Engine (ReactBits Port to Streamlit) ───
st.markdown("""
<div id="rb-webgl-ghost-cursor-container" style="position:fixed; top:0; left:0; width:100vw; height:100vh; pointer-events:none; z-index:999999; overflow:hidden;">
    <canvas id="rb-webgl-ghost-cursor-canvas" style="position:fixed; top:0; left:0; width:100vw; height:100vh; pointer-events:none; display:block; mix-blend-mode:screen;"></canvas>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
(function() {
    var targetWindow = window.parent || window;
    var targetDocument = targetWindow.document;

    if (targetWindow.hasOwnProperty('__RB_WEBGL_CURSOR_ACTIVE__')) { return; }
    targetWindow.__RB_WEBGL_CURSOR_ACTIVE__ = true;

    var container = document.getElementById('rb-webgl-ghost-cursor-container');
    var canvas = document.getElementById('rb-webgl-ghost-cursor-canvas');
    if (!canvas || !window.THREE) { return; }

    /* Teleport canvas container layout directly into parent document root view */
    if (container && targetDocument.body) {
        targetDocument.body.appendChild(container);
    }

    var renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true,
        depth: false,
        stencil: false,
        premultipliedAlpha: false
    });
    renderer.setClearColor(0x000000, 0);

    var scene = new THREE.Scene();
    var camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    var geom = new THREE.PlaneGeometry(2, 2);

    /* Props configuration matched with your React template layout values */
    var trailLength = 50;
    var inertia = 0.5;
    var brightness = 2.0;
    var rawColor = '#B497CF';
    
    var trailBuf = Array.from({ length: trailLength }, function() { return new THREE.Vector2(0.5, 0.5); });
    var head = 0;
    var currentMouse = new THREE.Vector2(0.5, 0.5);
    var velocity = new THREE.Vector2(0, 0);
    var fadeOpacity = 1.0;
    var lastMoveTime = performance.now();
    var pointerActive = false;

    var baseColor = new THREE.Color(rawColor);

    /* Fragment Shader Port converting ThreeJS GLSL parameters perfectly */
    var fragmentShaderCode = [
        "uniform float iTime;",
        "uniform vec3  iResolution;",
        "uniform vec2  iMouse;",
        "uniform vec2  iPrevMouse[" + trailLength + "];",
        "uniform float iOpacity;",
        "uniform float iScale;",
        "uniform vec3  iBaseColor;",
        "uniform float iBrightness;",
        "varying vec2  vUv;",

        "float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7))) * 43758.5453123); }",
        "float noise(vec2 p){",
        "  vec2 i = floor(p), f = fract(p);",
        "  f *= f * (3. - 2. * f);",
        "  return mix(mix(hash(i + vec2(0.,0.)), hash(i + vec2(1.,0.)), f.x),",
        "             mix(hash(i + vec2(0.,1.)), hash(i + vec2(1.,1.)), f.x), f.y);",
        "}",
        "float fbm(vec2 p){",
        "  float v = 0.0; float a = 0.5; mat2 m = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));",
        "  for(int i=0;i<5;i++){ v += a * noise(p); p = m * p * 2.0; a *= 0.5; }",
        "  return v;",
        "}",

        "vec4 blob(vec2 p, vec2 mousePos, float intensity, float activity) {",
        "  vec2 q = vec2(fbm(p * iScale + iTime * 0.1), fbm(p * iScale + vec2(5.2,1.3) + iTime * 0.1));",
        "  vec2 r = vec2(fbm(p * iScale + q * 1.5 + iTime * 0.15), fbm(p * iScale + q * 1.5 + vec2(8.3,2.8) + iTime * 0.15));",
        "  float smoke = fbm(p * iScale + r * 0.8);",
        "  float radius = 0.5 + 0.3 * (1.0 / iScale);",
        "  float distFactor = 1.0 - smoothstep(0.0, radius * activity, length(p - mousePos));",
        "  float alpha = pow(smoke, 2.5) * distFactor;",
        "  vec3 color = mix(mix(iBaseColor, vec3(1.0), 0.15), mix(iBaseColor, vec3(0.8, 0.9, 1.0), 0.25), sin(iTime * 0.5) * 0.5 + 0.5);",
        "  return vec4(color * alpha * intensity, alpha * intensity);",
        "}",

        "void main() {",
        "  vec2 uv = (gl_FragCoord.xy / iResolution.xy * 2.0 - 1.0) * vec2(iResolution.x / iResolution.y, 1.0);",
        "  vec2 mouse = (iMouse * 2.0 - 1.0) * vec2(iResolution.x / iResolution.y, 1.0);",
        "  vec3 colorAcc = vec3(0.0); float alphaAcc = 0.0;",
        "  vec4 b = blob(uv, mouse, 1.0, iOpacity); colorAcc += b.rgb; alphaAcc += b.a;",
        "  for (int i = 0; i < " + trailLength + "; i++) {",
        "    vec2 pm = (iPrevMouse[i] * 2.0 - 1.0) * vec2(iResolution.x / iResolution.y, 1.0);",
        "    float t = 1.0 - float(i) / float(" + trailLength + "); t = pow(t, 2.0);",
        "    if (t > 0.01) { vec4 bt = blob(uv, pm, t * 0.8, iOpacity); colorAcc += bt.rgb; alphaAcc += bt.a; }",
        "  }",
        "  colorAcc *= iBrightness;",
        "  float outAlpha = clamp(alphaAcc * iOpacity, 0.0, 1.0);",
        "  gl_FragColor = vec4(colorAcc, outAlpha);",
        "}"
    ].join("\n");

    var material = new THREE.ShaderMaterial({
        uniforms: {
            iTime: { value: 0 },
            iResolution: { value: new THREE.Vector3(1, 1, 1) },
            iMouse: { value: new THREE.Vector2(0.5, 0.5) },
            iPrevMouse: { value: trailBuf.map(function(v) { return v.clone(); }) },
            iOpacity: { value: 1.0 },
            iScale: { value: 1.2 },
            iBaseColor: { value: new THREE.Vector3(baseColor.r, baseColor.g, baseColor.b) },
            iBrightness: { value: brightness }
        },
        vertexShader: [
            "varying vec2 vUv;",
            "void main() { vUv = uv; gl_Position = vec4(position, 1.0); }"
        ].join("\n"),
        fragmentShader: fragmentShaderCode,
        transparent: true,
        depthTest: false,
        depthWrite: false
    });

    var mesh = new THREE.Mesh(geom, material);
    scene.add(mesh);

    function resize() {
        var w = targetWindow.innerWidth;
        var h = targetWindow.innerHeight;
        var dpr = Math.min(targetWindow.devicePixelRatio || 1, 1.5);
        renderer.setPixelRatio(dpr);
        renderer.setSize(w, h, false);
        material.uniforms.iResolution.value.set(w * dpr, h * dpr, 1);
    }
    targetWindow.addEventListener('resize', resize);
    resize();

    /* Sync Global Parent Frame Input Coordinates */
    targetWindow.addEventListener('mousemove', function(e) {
        var w = targetWindow.innerWidth;
        var h = targetWindow.innerHeight;
        currentMouse.set(e.clientX / w, 1 - (e.clientY / h));
        pointerActive = true;
        lastMoveTime = performance.now();
    });

    var start = performance.now();
    function animate() {
        var now = performance.now();
        var t = (now - start) / 1000;

        if (pointerActive) {
            velocity.set(currentMouse.x - material.uniforms.iMouse.value.x, currentMouse.y - material.uniforms.iMouse.value.y);
            material.uniforms.iMouse.value.copy(currentMouse);
            fadeOpacity = 1.0;
        } else {
            velocity.multiplyScalar(inertia);
            if (velocity.lengthSq() > 1e-6) { material.uniforms.iMouse.value.add(velocity); }
            var dt = now - lastMoveTime;
            if (dt > 1000) {
                var k = Math.min(1, (dt - 1000) / 1500);
                fadeOpacity = Math.max(0, 1 - k);
            }
        }

        head = (head + 1) % trailLength;
        trailBuf[head].copy(material.uniforms.iMouse.value);
        var arr = material.uniforms.iPrevMouse.value;
        for (var i = 0; i < trailLength; i++) {
            var srcIdx = (head - i + trailLength) % trailLength;
            arr[i].copy(trailBuf[srcIdx]);
        }

        material.uniforms.iOpacity.value = fadeOpacity;
        material.uniforms.iTime.value = t;

        renderer.render(scene, camera);
        targetWindow.requestAnimationFrame(animate);
    }
    animate();
})();
</script>
""", unsafe_allow_html=True)

# ── Futuristic Theme — ReactBits-inspired animations ─────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   FONTS & BASE SYSTEM
═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ═══════════════════════════════════════════════
   LIQUID ETHER BACKGROUND (reactbits: LiquidEther)
   Dynamic swirling, morphing fluid mesh of ether gradients
═══════════════════════════════════════════════ */
@keyframes rb-orb-float-center {
  0% { transform: translate(-50%, -50%) translate(0, 0) scale(1.0); filter: blur(60px) hue-rotate(0deg); }
  50% { transform: translate(-50%, -50%) translate(40px, -60px) scale(1.15); filter: blur(45px) hue-rotate(20deg); }
  100% { transform: translate(-50%, -50%) translate(-30px, 40px) scale(0.92); filter: blur(75px) hue-rotate(-15deg); }
}

@keyframes rb-orb-float-glow {
  0% { opacity: 0.40; transform: translate(-50%, -50%) scale(1.0) rotate(0deg); }
  50% { opacity: 0.60; transform: translate(-50%, -50%) scale(1.2) rotate(180deg); }
  100% { opacity: 0.40; transform: translate(-50%, -50%) scale(1.0) rotate(360deg); }
}

@keyframes rb-suborb-orbit-1 {
  0% { transform: translate(-50%, -50%) rotate(0deg) translateX(300px) rotate(0deg) scale(0.8); }
  50% { transform: translate(-50%, -50%) rotate(180deg) translateX(360px) rotate(-180deg) scale(1.1); }
  100% { transform: translate(-50%, -50%) rotate(360deg) translateX(300px) rotate(-360deg) scale(0.8); }
}

@keyframes rb-suborb-orbit-2 {
  0% { transform: translate(-50%, -50%) rotate(120deg) translateX(400px) rotate(-120deg) scale(1.1); }
  50% { transform: translate(-50%, -50%) rotate(300deg) translateX(320px) rotate(-300deg) scale(0.9); }
  100% { transform: translate(-50%, -50%) rotate(480deg) translateX(400px) rotate(-480deg) scale(1.1); }
}

.stApp {
  background-color: #03050c !important;
  color: #e2e8f5 !important;
  font-family: 'Inter', sans-serif !important;
  overflow-x: hidden;
  position: relative;
}

/* Background overlay of Orbs */
.rb-background-orb-center {
  position: fixed;
  top: 55%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 620px;
  height: 620px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #ff007f 0%, #7c3aed 35%, #06b6d4 70%, transparent 100%);
  mix-blend-mode: screen;
  z-index: 0;
  pointer-events: none;
  animation: rb-orb-float-center 22s ease-in-out infinite alternate;
  opacity: 0.35;
}

.rb-background-orb-glow {
  position: fixed;
  top: 55%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 1000px;
  height: 1000px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(245, 158, 11, 0.15) 0%, rgba(99, 102, 241, 0.10) 45%, rgba(13, 148, 136, 0.05) 75%, transparent 100%);
  mix-blend-mode: color-dodge;
  z-index: 0;
  pointer-events: none;
  animation: rb-orb-float-glow 30s ease-in-out infinite alternate;
  filter: blur(100px);
}

.rb-background-suborb-1 {
  position: fixed;
  top: 55%;
  left: 50%;
  width: 180px;
  height: 180px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(16, 185, 129, 0.35) 0%, rgba(16, 185, 129, 0) 70%);
  filter: blur(25px);
  mix-blend-mode: screen;
  z-index: 0;
  pointer-events: none;
  animation: rb-suborb-orbit-1 38s linear infinite;
  opacity: 0.55;
}

.rb-background-suborb-2 {
  position: fixed;
  top: 55%;
  left: 50%;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(239, 68, 68, 0.30) 0%, rgba(239, 68, 68, 0) 70%);
  filter: blur(35px);
  mix-blend-mode: screen;
  z-index: 0;
  pointer-events: none;
  animation: rb-suborb-orbit-2 48s linear infinite;
  opacity: 0.45;
}

/* Background grid details */
.stApp::before {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(circle, rgba(255,255,255,0.035) 1.2px, transparent 1.2px);
  background-size: 28px 28px;
  opacity: 0.85;
}

/* Futuristic Fine Dots Pattern Overlay (reactbits: DotGrid) */
.stApp > div {
  background-image: radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size: 24px 24px;
  position: relative; z-index: 1;
}

/* ═══════════════════════════════════════════════
   CRITICAL SCROLLING FIX FOR THE SIDEBAR
   Streamlit sidebars lock overflow by default,
   preventing input parameter access. We force scroll!
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: rgba(8, 10, 16, 0.94) !important;
  border-right: 1px solid rgba(59,130,246,0.18) !important;
  backdrop-filter: blur(24px) !important;
  -webkit-backdrop-filter: blur(24px) !important;
}

/* Targets the core structural containers in modern Streamlit to allow scrolling */
[data-testid="stSidebarContent"], 
[data-testid="stSidebarUserContent"], 
section[data-testid="stSidebar"] > div {
  overflow-y: auto !important;
  max-height: 100vh !important;
}

/* Customizable scrollbar for sleek cyber look */
[data-testid="stSidebarContent"]::-webkit-scrollbar {
  width: 5px;
}
[data-testid="stSidebarContent"]::-webkit-scrollbar-track {
  background: rgba(0,0,0,0.1);
}
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
  background: rgba(59,130,246,0.22);
  border-radius: 4px;
}
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {
  background: rgba(59,130,246,0.45);
}

/* ═══════════════════════════════════════════════
   TABS STYLE (Premium animated state)
═══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px !important;
  background: rgba(8,10,18,0.70) !important;
  border-bottom: 1px solid rgba(59,130,246,0.15) !important;
  backdrop-filter: blur(12px);
  padding: 4px 12px 0 12px !important;
  border-radius: 12px 12px 0 0;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: rgba(148,163,184,0.7) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  padding: 12px 24px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.3s ease, border-color 0.3s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #ffaa11 !important;
}
.stTabs [aria-selected="true"] {
  color: #ffcc00 !important;
  border-bottom: 2px solid #ffcc00 !important;
  background: rgba(255, 204, 0, 0.04) !important;
  text-shadow: 0 0 12px rgba(255, 204, 0, 0.30);
}

/* ═══════════════════════════════════════════════
   SHINY TEXT GRADIENTS (reactbits: ShinyText)
═══════════════════════════════════════════════ */
@keyframes shiny-glow {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.shiny-text {
  background: linear-gradient(120deg, #e2e8f0 25%, #ffcc00 50%, #e2e8f0 75%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shiny-glow 4s linear infinite;
  font-weight: 700;
}

/* ═══════════════════════════════════════════════
   METRIC CARDS (reactbits: Spotlight / Shimmer Card)
═══════════════════════════════════════════════ */
@keyframes shimmer-sweep {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.metric-card {
  position: relative;
  background: rgba(12,16,28,0.80) !important;
  border: 1px solid rgba(59,130,246,0.18) !important;
  border-radius: 12px !important;
  padding: 18px 16px !important;
  text-align: center;
  min-height: 100px;
  backdrop-filter: blur(16px);
  overflow: hidden;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease;
}
.metric-card::before {
  content: "";
  position: absolute; inset: 0; border-radius: 12px;
  background: linear-gradient(105deg,
    transparent 35%,
    rgba(59,130,246,0.08) 50%,
    transparent 65%);
  background-size: 200% 100%;
  animation: shimmer-sweep 4s linear infinite;
  pointer-events: none;
}
.metric-card:hover {
  border-color: rgba(59,130,246,0.45) !important;
  box-shadow: 0 8px 24px rgba(59,130,246,0.15);
  transform: translateY(-2px);
}
.metric-label {
  color: #94a3b8;
  font-size: 0.70rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 6px;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
}
.metric-value {
  color: #ffffff;
  font-size: 1.5rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  word-break: break-all;
  text-shadow: 0 0 15px rgba(59,130,246,0.20);
}
.metric-delta-up   { color: #10b981; font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 500;}
.metric-delta-down { color: #f43f5e; font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 500;}
.metric-muted      { color: #64748b; font-size: 0.78rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 500; }

/* ═══════════════════════════════════════════════
   PULSING SIGNAL BADGE (reactbits: Glowing Ring)
═══════════════════════════════════════════════ */
@keyframes ring-buy {
  0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.5), 0 0 15px rgba(16,185,129,0.2); }
  70%  { box-shadow: 0 0 0 10px rgba(16,185,129,0), 0 0 25px rgba(16,185,129,0.1); }
  100% { box-shadow: 0 0 0 0 rgba(16,185,129,0), 0 0 15px rgba(16,185,129,0.2); }
}
@keyframes ring-sell {
  0%   { box-shadow: 0 0 0 0 rgba(244,63,94,0.5), 0 0 15px rgba(244,63,94,0.2); }
  70%  { box-shadow: 0 0 0 10px rgba(244,63,94,0), 0 0 25px rgba(244,63,94,0.1); }
  100% { box-shadow: 0 0 0 0 rgba(244,63,94,0), 0 0 15px rgba(244,63,94,0.2); }
}
@keyframes ring-hold {
  0%   { box-shadow: 0 0 0 0 rgba(245,158,11,0.5), 0 0 15px rgba(245,158,11,0.2); }
  70%  { box-shadow: 0 0 0 10px rgba(245,158,11,0), 0 0 25px rgba(245,158,11,0.1); }
  100% { box-shadow: 0 0 0 0 rgba(245,158,11,0), 0 0 15px rgba(245,158,11,0.2); }
}
.signal-buy {
  background: rgba(16,185,129,0.1) !important;
  color: #10b981 !important;
  border: 2px solid rgba(16,185,129,0.7) !important;
  border-radius: 8px !important;
  padding: 10px 32px !important;
  font-weight: 700 !important;
  font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important;
  letter-spacing: 0.1em !important;
  display: inline-block !important;
  animation: ring-buy 2s infinite !important;
}
.signal-sell {
  background: rgba(244,63,94,0.1) !important;
  color: #f43f5e !important;
  border: 2px solid rgba(244,63,94,0.7) !important;
  border-radius: 8px !important;
  padding: 10px 32px !important;
  font-weight: 700 !important;
  font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important;
  letter-spacing: 0.1em !important;
  display: inline-block !important;
  animation: ring-sell 2s infinite !important;
}
.signal-hold {
  background: rgba(245,158,11,0.1) !important;
  color: #f59e0b !important;
  border: 2px solid rgba(245,158,11,0.7) !important;
  border-radius: 8px !important;
  padding: 10px 32px !important;
  font-weight: 700 !important;
  font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important;
  letter-spacing: 0.1em !important;
  display: inline-block !important;
  animation: ring-hold 2s infinite !important;
}

/* ═══════════════════════════════════════════════
   SECTION HEADERS (Scanning light animation)
═══════════════════════════════════════════════ */
@keyframes scanning-glow {
  0%   { left: -100%; }
  100% { left:  120%; }
}
.section-header {
  position: relative;
  color: #ffcc00;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-top: 24px;
  margin-bottom: 12px;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  overflow: hidden;
  padding-bottom: 6px;
}
.section-header::after {
  content: "";
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(59,130,246,0.3), transparent);
}
.section-header::before {
  content: "";
  position: absolute;
  bottom: 0;
  width: 35%;
  height: 1px;
  background: linear-gradient(90deg, transparent, #ffcc00, transparent);
  animation: scanning-glow 4s linear infinite;
}

/* ═══════════════════════════════════════════════
   HEADER HERO STRIP (reactbits: StarBorder vibe)
═══════════════════════════════════════════════ */
@keyframes border-sweep {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.app-header-banner {
  position: relative;
  background: rgba(10,14,26,0.72);
  border: 1px solid rgba(59,130,246,0.22);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 16px;
  overflow: hidden;
  backdrop-filter: blur(20px);
}
.app-header-banner::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 1.5px;
  background: linear-gradient(90deg, transparent, rgba(59,130,246,0.7), #ffcc00, transparent);
  background-size: 200% auto;
  animation: border-sweep 6s linear infinite;
}
.app-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.5rem;
  color: #ffffff;
  letter-spacing: -0.01em;
}
.app-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.70rem;
  color: #64748b;
  letter-spacing: 0.08em;
  margin-top: 4px;
}

/* ═══════════════════════════════════════════════
   INDICATOR COMPONENT
═══════════════════════════════════════════════ */
.indicator-row {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px;
  margin-bottom: 6px;
  border-radius: 8px;
  background: rgba(255,255,255,0.015);
  border: 1px solid rgba(255,255,255,0.03);
  font-family: 'Inter', sans-serif;
  font-size: 0.82rem;
  color: #e2e8f0;
  transition: background 0.25s ease, border-color 0.25s ease;
}
.indicator-row:hover {
  background: rgba(59,130,246,0.05);
  border-color: rgba(59,130,246,0.15);
}
.ind-icon { font-size: 0.9rem; flex-shrink: 0; }
.ind-name { font-weight: 500; flex: 1; }
.ind-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.05em;
  padding: 3px 10px;
  border-radius: 6px;
  font-weight: 600;
}
.badge-bull { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
.badge-bear { background: rgba(244,63,94,0.12);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.25); }
.badge-neut { background: rgba(245,158,11,0.10); color: #f59e0b; border: 1px solid rgba(245,158,11,0.20); }
.badge-warn { background: rgba(244,63,94,0.18);  color: #ffa5b5; border: 1px solid rgba(244,63,94,0.30); }

/* ═══════════════════════════════════════════════
   CHART DECORATION CONTAINER
═══════════════════════════════════════════════ */
.chart-wrap {
  position: relative;
  border: 1px solid rgba(59,130,246,0.16);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(8,10,18,0.40);
  padding: 6px;
  transition: border-color 0.3s ease;
}
.chart-wrap:hover { border-color: rgba(59,130,246,0.35); }

/* ═══════════════════════════════════════════════
   SLIDER, BUTTONS, SELECTIONS (Clean aesthetic adjustments)
═══════════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, #0b1530 0%, #152554 100%) !important;
  border: 1px solid rgba(59,130,246,0.40) !important;
  color: #60a5fa !important;
  border-radius: 8px !important;
  padding: 10px 24px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.80rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
  width: 100%;
}
.stButton > button:hover {
  border-color: rgba(59,130,246,0.80) !important;
  box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
  color: #93c5fd !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #10204d 0%, #1e3a8a 100%) !important;
  border-color: rgba(59,130,246,0.60) !important;
  color: #93c5fd !important;
}

/* Miscellaneous corrections */
hr { border-color: rgba(59,130,246,0.12) !important; }
#MainMenu, footer { visibility: hidden; }

/* =======================================================
   REACTBITS.DEV 27 PREMIUM COMPONENTS SHADOW CSS PROTOCOLS
   ======================================================= */

/* 1. AURORA */
.rb-aurora-viewport {
  position: relative; overflow: hidden; height: 180px; border-radius: 12px;
  background: #020408; border: 1px solid rgba(59,130,246,0.20);
}
.rb-aurora-blend {
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 30% 30%, rgba(147,51,234,0.3) 0%, transparent 60%),
              radial-gradient(ellipse at 70% 70%, rgba(6,182,212,0.25) 0%, transparent 60%);
}
.rb-aurora-blob {
  position: absolute; width: 140px; height: 140px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,170,17,0.3) 0%, transparent 70%);
  animation: rb-aurora-spin 12s linear infinite alternate;
  top: 10%; left: 35%;
}
@keyframes rb-aurora-spin {
  0% { transform: translate(-30px, -20px) scale(0.9); }
  100% { transform: translate(30px, 20px) scale(1.2); }
}

/* 2. DOTGRID */
.rb-dotgrid {
  background-image: radial-gradient(circle, rgba(255,170,17,0.14) 1px, transparent 1px);
  background-size: 14px 14px; padding: 24px; border-radius: 12px; border: 1px solid rgba(255,170,17,0.12);
  min-height: 140px; display: flex; flex-direction: column; justify-content: center;
}

/* 3. SHINYTEXT */
.rb-shinytext {
  background: linear-gradient(120deg, #94a3b8 30%, #ffaa11 50%, #94a3b8 70%);
  background-size: 200% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: rb-shiny-glow 3s linear infinite; font-weight: 700; display: inline-block;
}
@keyframes rb-shiny-glow { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }

/* 4. BLURTEXT */
.rb-blurtext {
  font-family: 'Space Grotesk', sans-serif;
  animation: rb-blur-reveal 2.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes rb-blur-reveal {
  0% { filter: blur(12px); opacity: 0; transform: translateY(4px); }
  100% { filter: blur(0); opacity: 1; transform: translateY(0); }
}

/* 5. SPOTLIGHTCARD */
.rb-spotlightcard {
  position: relative; background: #0c101c; border: 1px solid rgba(59,130,246,0.15);
  border-radius: 12px; padding: 24px; overflow: hidden; transition: all 0.3s ease;
  min-height: 140px;
}
.rb-spotlightcard:hover {
  border-color: rgba(6,182,212,0.5) !important;
  box-shadow: 0 0 20px rgba(6,182,212,0.15);
}
.rb-spotlightcard::after {
  content: ""; position: absolute; inset: -40px; pointer-events: none; opacity: 0;
  background: radial-gradient(circle 100px at 50% 50%, rgba(6,182,212,0.15), transparent 80%);
  transition: opacity 0.5s ease;
}
.rb-spotlightcard:hover::after { opacity: 1; }

/* 6. TILTEDCARD */
.rb-tiltedcard {
  background: rgba(12,16,28,0.8); border: 1px solid rgba(59,130,246,0.15);
  border-radius: 12px; padding: 20px; transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  transform-style: preserve-3d; min-height: 140px;
}
.rb-tiltedcard:hover {
  transform: perspective(400px) rotateX(8deg) rotateY(-8deg) scale(1.02);
  border-color: rgba(168,85,247,0.4) !important;
}

/* 7. STARBORDER */
.rb-starborder-container {
  position: relative; padding: 1.5px; border-radius: 12px; overflow: hidden; display: inline-block; width: 100%;
}
.rb-starborder-anim {
  position: absolute; width: 200%; height: 200%; top: -50%; left: -50%;
  background: conic-gradient(from 0deg, transparent 40%, #ffaa11, #ff3366, transparent 60%);
  animation: rb-star-orbit 5s linear infinite; pointer-events: none;
}
.rb-starborder-content {
  position: relative; background: #080a10; border-radius: 11px; padding: 16px; color: #fff; min-height: 120px;
}
@keyframes rb-star-orbit { 100% { transform: rotate(360deg); } }

/* 8. SPLITTEXT */
.rb-splittext {
  font-family: 'Space Grotesk', sans-serif; display: inline-block; transition: letter-spacing 0.30s ease, color 0.30s ease;
}
.rb-splittext:hover { letter-spacing: 0.12em !important; color: #ffaa11 !important; }

/* 9. SCRAMBLETEXT */
.rb-scrambletext {
  font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; color: #60a5fa;
  animation: rb-scramble-flicker 1.5s steps(3) infinite; display: inline-block;
}
@keyframes rb-scramble-flicker {
  0% { text-shadow: 0 0 2px rgba(96,165,250,0.5); filter: hue-rotate(0deg); }
  50% { opacity: 0.85; filter: hue-rotate(30deg); }
  100% { text-shadow: 0 0 4px rgba(96,165,250,0.8); filter: hue-rotate(0deg); }
}

/* 10. GLOWINGRING */
.rb-glowingring-box {
  width: 44px; height: 44px; border-radius: 50%; background: rgba(16,185,129,0.18);
  border: 1.5px solid #10b981; display: flex; align-items: center; justify-content: center;
  position: relative; animation: rb-ring-pulse 2s infinite; margin: 0 auto;
}
@keyframes rb-ring-pulse {
  0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.6); }
  70% { box-shadow: 0 0 0 12px rgba(16,185,129,0); }
  100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}

/* 11. RIPPLE */
.rb-ripple-btn {
  position: relative; overflow: hidden; background: rgba(59,130,246,0.1);
  border: 1px solid rgba(59,130,246,0.3); color: #60a5fa; border-radius: 8px;
  padding: 12px 20px; font-weight: 600; cursor: pointer; text-align: center; width: 100%;
}
.rb-ripple-btn::before {
  content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,0.15);
  width: 10px; height: 10px; opacity: 0; left: 50%; top: 50%; transform: scale(1) translate(-50%, -50%);
  transform-origin: 0 0;
}
.rb-ripple-btn:active::before {
  animation: rb-ripple-out 0.8s ease-out;
}
@keyframes rb-ripple-out {
  0% { transform: scale(1) translate(-50%, -50%); opacity: 1; }
  100% { transform: scale(32) translate(-50%, -50%); opacity: 0; }
}

/* 12. NOISE */
.rb-noise-bg {
  background-image: radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size: 16px 16px; position: relative; border-radius: 12px; padding: 20px; min-height: 140px;
}
.rb-noise-bg::before {
  content: ""; position: absolute; inset: 0; opacity: 0.05; border-radius: 12px;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  pointer-events: none;
}

/* 13. SHIMMERBUTTON */
.rb-shimmer-btn {
  background: linear-gradient(90deg, #10204d, #1e3a8a, #10204d);
  background-size: 200% auto; border: 1.5px solid rgba(59,130,246,0.6);
  border-radius: 30px; color: #fff; padding: 12px 30px; font-weight: 600;
  letter-spacing: 0.06em; animation: rb-shimmer-pass 2s linear infinite;
  display: inline-block; cursor: pointer; text-decoration: none; text-align: center; width: 100%;
}
@keyframes rb-shimmer-pass {
  0% { background-position: 0% center; }
  100% { background-position: -200% center; }
}

/* 14. TRUEFOCUS */
.rb-truefocus-item {
  font-family: 'Space Grotesk', sans-serif; display: inline-block; font-size: 1.02rem; color: rgba(255,255,255,0.3);
  filter: blur(2.5px); transition: filter 0.3s ease, color 0.3s ease, transform 0.3s ease, background 0.3s ease, border-color 0.3s ease;
}
.rb-truefocus-item.active {
  filter: blur(0); color: #ffcc00; font-weight: 700; transform: scale(1.04);
}
/* TrueFocus group hover centering: when hovering the container, blur all rows slightly, except the hovered one */
.rb-consensus-matrix-container:hover .rb-truefocus-item {
  filter: blur(1.8px) !important;
  opacity: 0.45 !important;
  transform: scale(0.97) !important;
}
.rb-consensus-matrix-container:hover .rb-truefocus-item:hover {
  filter: blur(0) !important;
  opacity: 1 !important;
  transform: scale(1.02) !important;
  background: rgba(59,130,246,0.08) !important;
  border-color: rgba(59,130,246,0.22) !important;
}

/* 15. ROLLINGCHARACTERS */
.rb-roller {
  display: inline-block; height: 1.5em; overflow: hidden; vertical-align: bottom; font-family: 'JetBrains Mono', monospace;
}
.rb-roller-list {
  animation: rb-roll-up 6s cubic-bezier(0.16, 1, 0.3, 1) infinite;
}
@keyframes rb-roll-up {
  0%, 20% { transform: translateY(0); }
  25%, 45% { transform: translateY(-1.21em); }
  50%, 70% { transform: translateY(-2.42em); }
  75%, 95% { transform: translateY(-3.63em); }
  100% { transform: translateY(0); }
}

/* 16. PIXELCARD */
.rb-pixelgrid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; padding: 10px;
  background: rgba(8,10,18,0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px;
}
.rb-pixel {
  height: 24px; border-radius: 3px; background: rgba(59,130,246,0.12);
  transition: background 0.1s ease, box-shadow 0.1s ease;
}
.rb-pixel:hover {
  background: #ffaa11; box-shadow: 0 0 10px #ffaa11;
}

/* 17. GRADIENTTEXT */
.rb-gradienttext {
  font-family: 'Space Grotesk', sans-serif; font-weight: 800;
  background: linear-gradient(45deg, #ffaa11, #ff3366, #ff00cc, #33ccff, #10b981);
  background-size: 400% 400%; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: rb-grad-shift 12s ease infinite;
}
@keyframes rb-grad-shift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* 18. TECHCARDDECORATOR (Gilded Corners) */
.rb-tech-bracket {
  position: absolute; width: 14px; height: 14px; border: 2.2px solid #ffcc00; pointer-events: none;
}
.rb-bracket-tl { top: -2.5px; left: -2.5px; border-right: none !important; border-bottom: none !important; }
.rb-bracket-tr { top: -2.5px; right: -2.5px; border-left: none !important; border-bottom: none !important; }
.rb-bracket-bl { bottom: -2.5px; left: -2.5px; border-right: none !important; border-top: none !important; }
.rb-bracket-br { bottom: -2.5px; right: -2.5px; border-left: none !important; border-top: none !important; }

/* 19. BOUNCECARD */
.rb-bounce-card {
  background: #080d19; border: 1px solid rgba(59,130,246,0.2); border-radius: 12px; padding: 20px;
  cursor: pointer; transition: transform 0.25s cubic-bezier(0.175, 0.885, 0.45, 1.4); min-height: 140px;
}
.rb-bounce-card:hover {
  transform: translateY(-8px) scale(1.02);
  border-color: #10b981 !important; box-shadow: 0 12px 30px rgba(16,185,129,0.15);
}

/* 20. STACKEDCARDS */
.rb-stack-container {
  position: relative; height: 140px; width: 100%; transition: height 0.4s ease;
}
.rb-stack-card {
  position: absolute; width: 90%; left: 5%; height: 80px; border-radius: 10px; padding: 12px;
  background: #0b0e17; border: 1px solid rgba(255,255,255,0.06); transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.rb-stack-c1 { top: 0px; z-index: 3; transform: scale(1); }
.rb-stack-c2 { top: 15px; z-index: 2; transform: scale(0.94); opacity: 0.75; }
.rb-stack-c3 { top: 30px; z-index: 1; transform: scale(0.88); opacity: 0.45; }

/* Elegant downward separating fan out to make all cards 100% readable */
.rb-stack-container:hover {
  height: 290px !important;
}
.rb-stack-container:hover .rb-stack-c1 {
  transform: translateY(0px) scale(1) !important;
  opacity: 1 !important;
  z-index: 10 !important;
  box-shadow: 0 8px 24px rgba(255,204,0,0.15) !important;
}
.rb-stack-container:hover .rb-stack-c2 {
  transform: translateY(90px) scale(1) !important;
  opacity: 1 !important;
  z-index: 9 !important;
  box-shadow: 0 8px 24px rgba(59,130,246,0.12) !important;
}
.rb-stack-container:hover .rb-stack-c3 {
  transform: translateY(180px) scale(1) !important;
  opacity: 1 !important;
  z-index: 8 !important;
  box-shadow: 0 8px 24px rgba(244,63,94,0.12) !important;
}

/* 21. LIQUIDPROGRESS */
.rb-liquid-ball {
  width: 90px; height: 90px; border-radius: 50%; background: #060a12;
  border: 3.5px solid #ffcc00; position: relative; overflow: hidden; margin: 0 auto;
}
.rb-liquid-wave {
  position: absolute; bottom: 0; left: -50%; width: 200%; height: 60%;
  background: rgba(255,204,0,0.65); border-radius: 38%;
  animation: rb-wave-motion 5s linear infinite;
}
@keyframes rb-wave-motion { 100% { transform: rotate(360deg); } }

/* 22. ORB */
.rb-orb-viewport {
  position: relative; height: 130px; background: #020408; border-radius: 8px; overflow: hidden;
}
.rb-orb-light {
  position: absolute; border-radius: 50%; width: 66px; height: 66px;
  background: radial-gradient(circle, #a855f7 0%, transparent 70%);
  animation: rb-orb-float 6s ease-in-out infinite alternate;
}
@keyframes rb-orb-float {
  0% { left: 10%; top: 10%; transform: scale(0.8); }
  100% { left: 65%; top: 40%; transform: scale(1.35); }
}

/* 23. MAGNET */
.rb-magnet-pill {
  display: inline-block; padding: 10px 20px; border-radius: 20px; background: rgba(59,130,246,0.15);
  border: 1px solid rgba(59,130,246,0.3); color: #fff; text-align: center;
  transition: transform 0.25s cubic-bezier(0.25, 1, 0.5, 1); cursor: pointer; width: 100%;
}
.rb-magnet-pill:hover {
  transform: scale(1.08) translate(4px, -2px);
  border-color: #ffaa11 !important;
}

/* 24. COUNTUP */
.rb-odometer {
  display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; color: #10b981; font-weight: 700;
  text-shadow: 0 0 10px rgba(16,185,129,0.25);
}

/* 25. GRIDMOTION */
.rb-gridmotion {
  height: 110px; background: repeating-linear-gradient(0deg, transparent, transparent 10px, rgba(59,130,246,0.06) 11px, rgba(59,130,246,0.06) 12px);
  position: relative; overflow: hidden; border-radius: 6px; border: 1.5px solid rgba(59,130,246,0.1);
}
.rb-grid-lines {
  position: absolute; inset: 0;
  background: repeating-linear-gradient(90deg, transparent, transparent 20px, rgba(6,182,212,0.12) 21px, rgba(6,182,212,0.12) 22px);
  animation: rb-grid-scroll 12s linear infinite;
}
@keyframes rb-grid-scroll { 100% { background-position: 100px 0; } }

/* 26. DIGITALPULSE */
.rb-pulse {
  width: 14px; height: 14px; border-radius: 50%; background: #10b981; display: inline-block; position: relative;
}
.rb-pulse::after {
  content: ""; position: absolute; inset: -4px; border-radius: 50%; border: 2px solid #10b981;
  animation: rb-pulse-ring 1.5s cubic-bezier(0.16, 1, 0.3, 1) infinite;
}
@keyframes rb-pulse-ring { 100% { transform: scale(2.4); opacity: 0; } }

/* 27. DIAGNOSTICCONSOLE */
.rb-console {
  background: #04060b; border: 1px solid #10b981; border-radius: 8px; padding: 18px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #10b981; line-height: 1.5;
  box-shadow: inset 0 0 10px rgba(16,185,129,0.15); position: relative; min-height: 140px;
}
.rb-console::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.05), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.05));
  background-size: 100% 4px, 6px 100%; border-radius: 8px;
}

/* ═══════════════════════════════════════════════
   28. STAGGERED MENU (reactbits: StaggeredMenu)
═══════════════════════════════════════════════ */
.rb-menu-container {
  position: relative;
  width: 100%;
  margin-bottom: 24px;
}
.rb-menu-toggle {
  display: flex !important;
  align-items: center;
  justify-content: space-between !important;
  width: 100%;
  padding: 14px 18px !important;
  background: rgba(59,130,246,0.06) !important;
  border: 1.5px solid rgba(59,130,246,0.22) !important;
  border-radius: 10px !important;
  color: #ffcc00 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.08em !important;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
  user-select: none;
}
.rb-menu-toggle:hover {
  background: rgba(59,130,246,0.12) !important;
  border-color: #ffaa11 !important;
  box-shadow: 0 0 20px rgba(255,204,0,0.18) !important;
}
.rb-menu-dropdown {
  display: flex !important;
  flex-direction: column !important;
  gap: 10px !important;
  margin-top: 10px !important;
  perspective: 1000px;
  transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.rb-menu-item {
  display: flex !important;
  align-items: center !important;
  gap: 12px !important;
  padding: 12px 16px !important;
  background: rgba(12,16,28,0.85) !important;
  border: 1px solid rgba(59,130,246,0.12) !important;
  border-radius: 8px !important;
  color: #a1b0cb !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  text-decoration: none !important;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
  transform-style: preserve-3d;
  opacity: 0;
  transform: rotateX(-15deg) translateY(-10px);
  animation: rb-stagger-in 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) forwards !important;
}
.rb-menu-item:hover {
  background: rgba(255,204,0,0.06) !important;
  border-color: #ffcc00 !important;
  color: #ffffff !important;
  transform: scale(1.02) translateZ(15px) !important;
}
.rb-menu-item-icon {
  color: #ffcc00 !important;
  font-size: 0.95rem !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
@keyframes rb-stagger-in {
  to {
    opacity: 1;
    transform: rotateX(0) translateY(0);
  }
}
.rb-menu-item:nth-child(1) { animation-delay: 0.08s !important; }
.rb-menu-item:nth-child(2) { animation-delay: 0.16s !important; }
.rb-menu-item:nth-child(3) { animation-delay: 0.24s !important; }
.rb-menu-item:nth-child(4) { animation-delay: 0.32s !important; }
.rb-menu-item:nth-child(5) { animation-delay: 0.40s !important; }

/* Checkbox hack to toggle the menu smoothly without rerun */
#rb-menu-toggle-chk:checked ~ .rb-menu-dropdown {
  display: flex !important;
}
#rb-menu-toggle-chk:not(:checked) ~ .rb-menu-dropdown {
  max-height: 0 !important;
  opacity: 0 !important;
  overflow: hidden !important;
  margin-top: 0 !important;
  pointer-events: none !important;
}
#rb-menu-toggle-chk:checked ~ .rb-menu-dropdown {
  max-height: 500px !important;
  opacity: 1 !important;
}
#rb-menu-toggle-chk:checked ~ .rb-menu-toggle .rb-menu-arrow {
  transform: rotate(180deg) !important;
}
.rb-menu-arrow {
  transition: transform 0.3s ease !important;
}

/* ═══════════════════════════════════════════════
   29. FLYING POSTERS (reactbits: FlyingPosters)
═══════════════════════════════════════════════ */
@keyframes rb-poster-float-1 {
  0%, 100% { transform: translateY(0px) rotate(0.6deg); }
  50%      { transform: translateY(-7px) rotate(-0.6deg); }
}
@keyframes rb-poster-float-2 {
  0%, 100% { transform: translateY(-6px) rotate(-0.8deg); }
  50%      { transform: translateY(4px) rotate(0.8deg); }
}
@keyframes rb-poster-float-3 {
  0%, 100% { transform: translateY(3px) rotate(0.5deg); }
  50%      { transform: translateY(-5px) rotate(-0.5deg); }
}

.flying-posters-viewport {
  display: flex !important;
  overflow-x: auto !important;
  overflow-y: visible !important;
  gap: 28px !important;
  padding: 16px 14px 28px 14px !important;
  scroll-behavior: smooth !important;
  perspective: 1200px;
  -webkit-overflow-scrolling: touch !important;
  width: 100% !important;
}
/* Cyberpunk thin custom scrollbar for the dynamic flying posters gallery */
.flying-posters-viewport::-webkit-scrollbar {
  height: 6px !important;
}
.flying-posters-viewport::-webkit-scrollbar-track {
  background: rgba(12,16,28,0.25) !important;
  border-radius: 10px !important;
}
.flying-posters-viewport::-webkit-scrollbar-thumb {
  background: rgba(59,130,246,0.22) !important;
  border-radius: 10px !important;
  border: 1px solid rgba(59,130,246,0.15) !important;
}
.flying-posters-viewport::-webkit-scrollbar-thumb:hover {
  background: rgba(255,204,0,0.45) !important;
}

@keyframes rb-border-glow-rotate {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.flying-poster-card {
  flex: 0 0 450px !important;
  min-width: 450px !important;
  background: rgba(10,13,22,0.85) !important;
  border-radius: 14px !important;
  padding: 18px !important;
  box-shadow: 0 12px 34px rgba(0,0,0,0.55) !important;
  transition: box-shadow 0.4s ease, transform 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
  transform-style: preserve-3d;
  will-change: transform;
  position: relative !important;
  overflow: visible !important;
}

/* Beautiful ReactBits-inspired Border Glow */
.flying-poster-card::before {
  content: "" !important;
  position: absolute !important;
  inset: -1.5px !important;
  border-radius: 15px !important;
  padding: 1.5px !important;
  background: linear-gradient(135deg, #3b82f6, #a855f7, #ffcc00, #3b82f6) !important;
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
  -webkit-mask-composite: xor !important;
  mask-composite: exclude !important;
  background-size: 200% auto !important;
  animation: rb-border-glow-rotate 5s linear infinite !important;
  pointer-events: none !important;
  opacity: 0.38 !important;
  transition: opacity 0.4s ease, inset 0.4s ease, padding 0.4s ease !important;
}

.flying-poster-card:nth-child(3n+1) {
  animation: rb-poster-float-1 6.5s ease-in-out infinite !important;
}
.flying-poster-card:nth-child(3n+2) {
  animation: rb-poster-float-2 7.5s ease-in-out infinite !important;
}
.flying-poster-card:nth-child(3n) {
  animation: rb-poster-float-3 5.8s ease-in-out infinite !important;
}

.flying-poster-card:hover {
  transform: translateY(-14px) rotateY(4.5deg) scale(1.025) !important;
  box-shadow: 0 24px 44px rgba(59,130,246,0.22) !important;
  z-index: 99 !important;
  animation-play-state: paused !important;
}
.flying-poster-card:hover::before {
  opacity: 1.0 !important;
  inset: -2.5px !important;
  padding: 2.5px !important;
  border-radius: 16px !important;
}

/* ═══════════════════════════════════════════════
   30. INFINITE SCROLL MATRIX
═══════════════════════════════════════════════ */
.rb-infinite-scroll-container {
  overflow: hidden !important;
  position: relative !important;
  height: 220px !important;
  width: 100% !important;
  border: 1px solid rgba(59,130,246,0.12) !important;
  border-radius: 8px !important;
  background: rgba(12,16,28,0.3) !important;
  box-sizing: border-box !important;
  padding: 8px !important;
}
.rb-infinite-scroll-content {
  display: flex !important;
  flex-direction: column !important;
  gap: 8px !important;
  animation: rb-vertical-scroll 25s linear infinite !important;
}
@keyframes rb-vertical-scroll {
  0% { transform: translateY(0); }
  100% { transform: translateY(-50%); }
}
.rb-infinite-scroll-container:hover .rb-infinite-scroll-content {
  animation-play-state: paused !important;
}

/* Beautiful Shimmer/Animation Button style for graph titles on top of charts with Hover Tooltip Pop-ups */
.rb-graph-title-btn {
  background: linear-gradient(90deg, #090c15, #17223b, #090c15);
  background-size: 200% auto;
  border: 1.2px solid rgba(255, 204, 0, 0.45) !important;
  animation: rb-shimmer-pass 3s linear infinite !important;
  border-radius: 20px !important;
  color: #ffcc00 !important;
  padding: 6px 16px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.77rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  display: inline-block !important;
  text-align: center !important;
  margin: 0 auto 12px auto !important;
  text-transform: uppercase !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
  position: relative !important;
  cursor: help !important;
}

/* Tooltip Popup Box */
.rb-graph-title-btn::after {
  content: "Click inside chart area to inspect data coordinates";
  position: absolute;
  bottom: 125%;
  left: 50%;
  transform: translateX(-50%) translateY(5px);
  background: #111726 !important;
  border: 1px solid #ffcc00 !important;
  color: #ffffff !important;
  padding: 6px 12px !important;
  border-radius: 6px !important;
  font-size: 0.72rem !important;
  font-family: 'JetBrains Mono', monospace !important;
  white-space: nowrap !important;
  opacity: 0 !important;
  pointer-events: none !important;
  transition: opacity 0.2s ease, transform 0.2s ease !important;
  z-index: 99999 !important;
  box-shadow: 0 4px 15px rgba(0,0,0,0.6) !important;
}

/* Hover-Pop State Trigger */
.rb-graph-title-btn:hover::after {
  opacity: 1 !important;
  transform: translateX(-50%) translateY(0) !important;
}

/* Enhanced hover trigger support for Project Scenario Stack expansion */
.rb-stack-hover-trigger:hover .rb-stack-container {
  height: 290px !important;
}
.rb-stack-hover-trigger:hover .rb-stack-c1 {
  transform: translateY(0px) scale(1) !important;
  opacity: 1 !important;
  z-index: 10 !important;
  box-shadow: 0 8px 24px rgba(255,204,0,0.15) !important;
}
.rb-stack-hover-trigger:hover .rb-stack-c2 {
  transform: translateY(90px) scale(1) !important;
  opacity: 1 !important;
  z-index: 9 !important;
  box-shadow: 0 8px 24px rgba(59,130,246,0.12) !important;
}
.rb-stack-hover-trigger:hover .rb-stack-c3 {
  transform: translateY(180px) scale(1) !important;
  opacity: 1 !important;
  z-index: 8 !important;
  box-shadow: 0 8px 24px rgba(244,63,94,0.12) !important;
}
</style>
""", unsafe_allow_html=True)

# Render the backgrounds: Orb centerpiece background of reactbits (Orbiting sub-orbs and centerpieces)
st.markdown("""
<div class="rb-background-orb-glow"></div>
<div class="rb-background-orb-center"></div>
<div class="rb-background-suborb-1"></div>
<div class="rb-background-suborb-2"></div>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║                    CONSTANTS                             ║
# ╚══════════════════════════════════════════════════════════╝

CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TSLA_1.csv")
LOOKBACK  = 60
PLOT_BG   = "#080a10"
GRID_COL  = "#1a2135"
FONT_COL  = "#e2e8f5"
ACCENT    = "#ffcc00"
GREEN     = "#10b981"
RED       = "#f43f5e"
BLUE      = "#3b82f6"
PURPLE    = "#a855f7"
MUTED     = "#64748b"

# ╔══════════════════════════════════════════════════════════╗
# ║                    HELPERS                               ║
# ╚══════════════════════════════════════════════════════════╝

def safe_float(val, fallback=0.0) -> float:
    try:
        v = float(val)
        return fallback if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return fallback

def empty_state(icon: str, msg: str):
    st.markdown(f"""
    <div class="rb-starborder-container" style="margin-top: 10px; width: 100%;">
        <div class="rb-starborder-anim"></div>
        <div class="rb-starborder-content" style="background:#0a0d16; display:flex; align-items:center; flex-direction:column; padding:36px; text-align:center; min-height: 180px; position: relative; overflow: hidden;">
            <!-- ParticlesBg (Component 27) / DotGrid (Component 2) coordinate drifting backplane simulated visually -->
            <div style="position: absolute; inset: 0; background-image: radial-gradient(circle, rgba(255,255,255,0.06) 1px, transparent 1px); background-size: 20px 20px; opacity: 0.55; pointer-events: none;" class="rb-dotgrid"></div>
            
            <div style="font-size: 2.8rem; margin-bottom: 12px; z-index: 2;">{icon}</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.90rem; max-width: 480px; margin: 0 auto; line-height: 1.6; color: #a1b0cb; font-weight: 500; z-index: 2;">{msg}</div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: #64748b; margin-top: 14px; letter-spacing: 0.05em; z-index: 2;">COSMOS PARTICLES VECTOR BACKPLANE STABILIZED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label: str, value: str, delta: str = "", delta_cls: str = "metric-muted", show_scramble=False) -> str:
    # Golden corners TechCardDecorator (Component 15 / 18)
    decorator = """
    <div class="rb-tech-bracket rb-bracket-tl" style="border-color: rgba(255,204,0,0.3);"></div>
    <div class="rb-tech-bracket rb-bracket-tr" style="border-color: rgba(255,204,0,0.3);"></div>
    <div class="rb-tech-bracket rb-bracket-bl" style="border-color: rgba(255,204,0,0.3);"></div>
    <div class="rb-tech-bracket rb-bracket-br" style="border-color: rgba(255,204,0,0.3);"></div>
    """
    
    # Check if we show a gold shine or sliding text glow
    val_inner = f'<span class="rb-shinytext">{value}</span>' if show_scramble else f'<span class="rb-blurtext" style="color: #ffffff;">{value}</span>'
    
    # SpotlightCard (Component 12) + BounceCard (Component 16) + Noise Overlay (Component 3) + TechCardDecorator (Component 15) + BlurText (Component 8):
    html_content = f"""
    <div class="metric-card rb-spotlightcard rb-bounce-card" style="position: relative; overflow: visible; padding: 18px 16px !important; min-height: 110px; background: #0c101c !important;">
        {decorator}
        <div class="rb-noise-bg" style="padding: 0; min-height: 0;">
            <div class="metric-label" style="font-family: 'Space Grotesk', sans-serif; font-size: 0.70rem; color: #94a3b8; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px;">{label}</div>
            <div class="metric-value" style="font-size: 1.45rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 2px 0;">{val_inner}</div>
            <div class="{delta_cls}" style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 500;">{delta}</div>
        </div>
    </div>
    """
    return clean_html(html_content)

def base_layout(height: int = 350, title: str = "", override_yaxis=None) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,16,28,0.40)",
        font_color=FONT_COL, height=height,
        margin=dict(l=40, r=20, t=40, b=30),
        title=dict(text=title, font=dict(size=12, color=MUTED, family="Space Grotesk")),
        xaxis=dict(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL),
    )
    if override_yaxis is not None:
        layout["yaxis"].update(override_yaxis)
    return layout

# ╔══════════════════════════════════════════════════════════╗
# ║                    DATA LAYER                            ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> tuple[pd.DataFrame, list[str]]:
    warnings_out = []
    df = pd.DataFrame()
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
        except Exception as e:
            warnings_out.append(f"Could not read TSLA_1.csv: {e}")
    else:
        warnings_out.append("TSLA_1.csv unified dataset not found. Initializing highly realistic historical simulation data automatically.")
        # Fallback simulation generator of TSLA stock movement
        np.random.seed(42)
        base_price = 220.0
        start_date = pd.Timestamp("2025-06-11")
        biz_dates = pd.bdate_range(start=start_date, periods=252)
        
        prices = [base_price]
        current = base_price
        drift = 0.0006
        volatility = 0.022
        
        for _ in range(1, len(biz_dates)):
            daily_return = drift + volatility * np.random.normal()
            current = max(current * np.exp(daily_return), 15.0)
            prices.append(current)
            
        prices = np.array(prices)
        opens = prices * (1.0 + (np.random.rand(len(biz_dates)) - 0.5) * 0.015)
        spreads = prices * (0.01 + np.random.rand(len(biz_dates)) * 0.04)
        highs = np.maximum(opens, prices) + spreads * 0.35
        lows = np.minimum(opens, prices) - spreads * 0.65
        volumes = (12 + np.random.rand(len(biz_dates)) * 26) * 1000000
        
        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Adj Close": prices,
            "Volume": volumes.astype(int)
        }, index=biz_dates)
        df.index.name = "Date"

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        return pd.DataFrame(), [f"Dataset is missing columns: {', '.join(missing)}"]

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df["Spread"]    = df["High"] - df["Low"]
    df["MA30"]      = df["Close"].rolling(30).mean()
    df["MA90"]      = df["Close"].rolling(90).mean()
    df["MA200"]     = df["Close"].rolling(200).mean()
    df["EMA12"]     = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"]     = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = df["EMA12"] - df["EMA26"]
    df["MACDSig"]   = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"]  = df["MACD"] - df["MACDSig"]
    df["DailyReturn"] = df["Close"].pct_change() * 100

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    return df, warnings_out

def build_scaler(df: pd.DataFrame):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    adj_vals = df[["Adj Close"]].dropna().values
    if len(adj_vals) == 0:
        raise ValueError("No valid 'Adj Close' values to fit the scaler.")
    scaler.fit(adj_vals)
    return scaler

# ╔══════════════════════════════════════════════════════════╗
# ║                    MODEL LAYER                           ║
# ╚══════════════════════════════════════════════════════════╝

def extract_gdrive_id(url: str) -> str | None:
    url = url.strip()
    for pattern in [
        r"/file/d/([a-zA-Z0-9_-]{20,})",
        r"[?&]id=([a-zA-Z0-9_-]{20,})",
        r"/d/([a-zA-Z0-9_-]{20,})/",
        r"^([a-zA-Z0-9_-]{20,})$",
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

@st.cache_resource(show_spinner=False)
def load_model_cached(file_id: str):
    try: import gdown
    except ImportError: raise RuntimeError("'gdown' is missing from requirements.txt.")
    try: import tensorflow as tf
    except ImportError: raise RuntimeError("'tensorflow' is missing from requirements.txt.")

    download_url = f"https://drive.google.com/uc?id={file_id}"
    tmp_path     = os.path.join(tempfile.gettempdir(), f"tsla_model_{file_id[:8]}.keras")

    if not os.path.exists(tmp_path):
        result = gdown.download(download_url, tmp_path, quiet=True)
        if result is None or not os.path.exists(tmp_path):
            raise RuntimeError("Download failed. Verify link share options.")

    try:
        model = tf.keras.models.load_model(tmp_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Model load exception: {e}")

    return model

# ╔══════════════════════════════════════════════════════════╗
# ║            HYBRID REVISION ENGINE                        ║
# ║  Return-space recursion + Hurst regime detection +       ║
# ║  Ornstein-Uhlenbeck mean-reversion damping               ║
# ╚══════════════════════════════════════════════════════════╝

def _hurst_exponent(series: np.ndarray) -> float:
    n = len(series)
    if n < 20:
        return 0.5
    try:
        lags   = [2, 4, 8, 16, 32] if n >= 64 else [2, 4, 8]
        rs_vals = []
        for lag in lags:
            chunks = [series[i:i+lag] for i in range(0, n - lag + 1, lag)]
            if not chunks:
                continue
            rs_chunk = []
            for c in chunks:
                mean_c  = np.mean(c)
                deviate = np.cumsum(c - mean_c)
                r       = deviate.max() - deviate.min()
                s       = np.std(c, ddof=1)
                if s > 0:
                    rs_chunk.append(r / s)
            if rs_chunk:
                rs_vals.append((lag, np.mean(rs_chunk)))
        if len(rs_vals) < 2:
            return 0.5
        lags_arr  = np.log([v[0] for v in rs_vals])
        rs_arr    = np.log([v[1] for v in rs_vals])
        H         = np.polyfit(lags_arr, rs_arr, 1)[0]
        return float(np.clip(H, 0.1, 0.9))
    except Exception:
        return 0.5

def _compute_regime(history_series: pd.Series) -> dict:
    prices = history_series.dropna().values[-252:]
    if len(prices) < 30:
        return dict(hurst=0.5, trend_slope=0.0, mean_price=prices[-1],
                    vol=0.02, ou_speed=0.05)

    log_rets  = np.diff(np.log(prices))
    vol       = float(np.std(log_rets)) if len(log_rets) > 1 else 0.02
    H         = _hurst_exponent(prices)

    x          = np.arange(len(prices))
    log_prices = np.log(prices)
    slope      = float(np.polyfit(x, log_prices, 1)[0]) * 252  # annualised

    ema_span   = min(200, len(prices))
    weights    = np.exp(np.linspace(-1, 0, ema_span))
    weights   /= weights.sum()
    mean_price = float(np.convolve(prices, weights[::-1], mode='valid')[-1])

    try:
        if len(log_rets) > 10:
            phi    = np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1]
            phi    = np.clip(phi, -0.99, 0.99)
            ou_speed = float(-np.log(abs(phi))) if abs(phi) < 1.0 else 0.05
        else:
            ou_speed = 0.05
    except Exception:
        ou_speed = 0.05

    return dict(hurst=H, trend_slope=slope, mean_price=mean_price,
                vol=vol, ou_speed=ou_speed)

def _single_step_forecast(model, scaler, lookback_context: list) -> float:
    x_sc  = scaler.transform(np.array(lookback_context[-LOOKBACK:]).reshape(-1, 1)).flatten()
    x_in  = np.array(x_sc, dtype=np.float32).reshape(1, LOOKBACK, 1)
    raw   = float(np.clip(model.predict(x_in, verbose=0)[0, 0], 0.0, 1.0))
    return float(scaler.inverse_transform([[raw]])[0, 0])

def _revise_prediction(
    raw_pred:       float,
    prev_price:     float,
    anchor_price:   float,
    regime:         dict,
    step:           int,
    total_steps:    int,
) -> float:
    H           = regime["hurst"]
    mean_price  = regime["mean_price"]
    ou_speed    = regime["ou_speed"]
    vol         = regime["vol"]

    if prev_price <= 0 or anchor_price <= 0:
        return raw_pred

    raw_log_ret = np.log(max(raw_pred, 1e-6) / max(prev_price, 1e-6))
    cumulative_log_ret = np.log(max(prev_price, 1e-6) / max(anchor_price, 1e-6))

    drift_budget = 2.0 * vol * np.sqrt(step)
    drift_excess = abs(cumulative_log_ret) - drift_budget
    if drift_excess > 0:
        correction_sign     = -np.sign(cumulative_log_ret)
        drift_correction    = correction_sign * drift_excess * 0.15
    else:
        drift_correction = 0.0

    price_gap_pct = abs(prev_price - mean_price) / mean_price
    if price_gap_pct > 0.05:
        ou_pull = -ou_speed * np.log(max(prev_price, 1e-6) / max(mean_price, 1e-6))
        ou_scale = float(np.interp(H, [0.35, 0.50, 0.65], [0.35, 0.20, 0.08]))
    else:
        ou_pull  = 0.0
        ou_scale = 0.0

    final_log_ret = raw_log_ret + drift_correction + (ou_scale * ou_pull)
    final_log_ret = float(np.clip(final_log_ret, -4.0 * vol, 4.0 * vol))

    revised_price = prev_price * np.exp(final_log_ret)
    revised_price = float(np.clip(revised_price, mean_price * 0.20, mean_price * 3.50))
    return revised_price

def dynamic_timeline_forecasting(
    model, scaler, df: pd.DataFrame, start_date: pd.Timestamp, n_days: int
) -> tuple:
    db_max_date  = df.index.max()
    target_start = pd.Timestamp(start_date)
    biz_dates    = pd.bdate_range(start=target_start, periods=n_days)

    recent_ret = df["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    daily_vol  = (recent_ret.std() / 100) if len(recent_ret) >= 5 else 0.02

    preds_prices: list = []
    bridge_dates, bridge_prices, bridge_lo, bridge_hi = [], [], [], []

    # ── PATH A: start_date is within or at the database boundary ─────────────
    if target_start <= db_max_date:
        anchor_price_a = safe_float(df["Adj Close"].iloc[-1])
        future_step    = 0

        for curr_date in biz_dates:
            if curr_date <= db_max_date:
                pos_idx = df.index.get_indexer([curr_date], method="pad")[0]
                pos_idx = max(pos_idx, 0)
                preds_prices.append(float(df.iloc[pos_idx]["Adj Close"]))
            else:
                future_step += 1
                pos_idx = df.index.get_indexer([curr_date], method="pad")[0]
                if pos_idx != -1 and pos_idx >= LOOKBACK:
                    history_slice = df.iloc[:pos_idx]
                else:
                    history_slice = df.head(LOOKBACK)

                lookback_context = history_slice.tail(LOOKBACK)["Adj Close"].tolist()
                idx_offset = len(preds_prices) - 1
                while len(lookback_context) < LOOKBACK and idx_offset >= 0:
                    lookback_context.insert(0, preds_prices[idx_offset])
                    idx_offset -= 1

                raw_pred = _single_step_forecast(model, scaler, lookback_context)
                regime   = _compute_regime(history_slice.tail(252)["Adj Close"])
                prev_p   = lookback_context[-1]
                revised  = _revise_prediction(
                    raw_pred, prev_p, anchor_price_a, regime, future_step, n_days
                )
                preds_prices.append(revised)

    # ── PATH B: start_date is strictly beyond the database boundary ───────────
    else:
        working_df = df[["Adj Close"]].copy()
        gap_range  = pd.bdate_range(
            start=db_max_date + pd.Timedelta(days=1),
            end=target_start - pd.Timedelta(days=1),
        )

        if len(gap_range) > 0:
            regime_bridge  = _compute_regime(working_df.tail(252)["Adj Close"])
            anchor_bridge  = safe_float(working_df["Adj Close"].iloc[-1])
            total_bridge   = len(gap_range)

            for b_step, g_date in enumerate(gap_range, start=1):
                seed_vals = working_df.tail(LOOKBACK)["Adj Close"].tolist()
                raw_pred  = _single_step_forecast(model, scaler, seed_vals)
                prev_p    = seed_vals[-1]
                revised   = _revise_prediction(
                    raw_pred, prev_p, anchor_bridge, regime_bridge, b_step, total_bridge
                )

                working_df.loc[g_date, "Adj Close"] = revised
                bridge_dates.append(g_date)
                bridge_prices.append(revised)

                band_frac = np.clip(daily_vol * np.sqrt(b_step), 0, 0.45)
                bridge_lo.append(revised * (1 - band_frac))
                bridge_hi.append(revised * (1 + band_frac))

        regime_target = _compute_regime(working_df.tail(252)["Adj Close"])
        anchor_target = safe_float(working_df["Adj Close"].iloc[-1])
        for t_step, b_date in enumerate(biz_dates, start=1):
            seed_vals = working_df.tail(LOOKBACK)["Adj Close"].tolist()
            raw_pred  = _single_step_forecast(model, scaler, seed_vals)
            prev_p    = seed_vals[-1]
            revised   = _revise_prediction(
                raw_pred, prev_p, anchor_target, regime_target, t_step, n_days
            )
            preds_prices.append(revised)
            working_df.loc[b_date, "Adj Close"] = revised

    # ── Confidence bands ─────────────────────────────────────────────────────
    preds_prices = np.array(preds_prices, dtype=np.float32)
    lower_bounds, upper_bounds = [], []

    bridge_steps_count = len(bridge_prices)
    for idx in range(len(preds_prices)):
        total_depth = bridge_steps_count + idx + 1
        band_frac   = np.clip(daily_vol * np.sqrt(total_depth), 0, 0.45)
        lower_bounds.append(preds_prices[idx] * (1 - band_frac))
        upper_bounds.append(preds_prices[idx] * (1 + band_frac))

    return (
        biz_dates,
        preds_prices,
        np.array(lower_bounds),
        np.array(upper_bounds),
        pd.DatetimeIndex(bridge_dates),
        np.array(bridge_prices),
        np.array(bridge_lo),
        np.array(bridge_hi),
    )

# ╔══════════════════════════════════════════════════════════╗
# ║                    LOAD INITIALIZER                      ║
# ╚══════════════════════════════════════════════════════════╝

with st.spinner("Decoding dataset configuration matrix…"):
    df, data_warnings = load_data()

if df.empty:
    st.error("⛔ Dataset parsing exception. Check structure status.")
    st.stop()

for w in data_warnings:
    st.warning(f"⚠️ {w}")

current_price  = safe_float(df["Close"].iloc[-1])
prev_price     = safe_float(df["Close"].iloc[-2], fallback=current_price)
price_change   = current_price - prev_price
price_change_p = (price_change / prev_price * 100) if prev_price != 0 else 0.0

try:
    scaler = build_scaler(df)
    scaler_ok = True
except Exception as _scaler_err:
    scaler_ok = False
    st.warning(f"⚠️ Scaler core fault: {_scaler_err}")

# ╔══════════════════════════════════════════════════════════╗
# ║                    SIDEBAR CONTROLS                      ║
# ╚══════════════════════════════════════════════════════════╝

# Implicit dynamic model config values
gdrive_url = ""
load_btn = False

# Critical configuration sidebar with optimized scrolling styling and interactive nodes
with st.sidebar:
    # ShinyText on header
    st.markdown("## <span class='rb-shinytext' style='font-family: \"Space Grotesk\", sans-serif; font-size:1.6rem; font-weight:700;'>⚡ TSLA TRANSCEIVER</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Beautiful interactive ReactBits StaggeredMenu integration
    st.markdown("""
    <div class="rb-menu-container">
        <input type="checkbox" id="rb-menu-toggle-chk" style="display:none;" checked />
        <label for="rb-menu-toggle-chk" class="rb-menu-toggle">
            <span>🌐 NETWORK HUB DIRECTORY</span>
            <span class="rb-menu-arrow">▼</span>
        </label>
        <div class="rb-menu-dropdown">
            <div class="rb-menu-item">
                <span class="rb-menu-item-icon">🛰️</span>
                <span>Real-Time Core Telemetry</span>
            </div>
            <div class="rb-menu-item">
                <span class="rb-menu-item-icon">🔮</span>
                <span>Hybrid Forecast Engine</span>
            </div>
            <div class="rb-menu-item">
                <span class="rb-menu-item-icon">📶</span>
                <span>Oscillator Velocity Metrics</span>
            </div>
            <div class="rb-menu-item">
                <span class="rb-menu-item-icon">🎛️</span>
                <span>Transceiver Calibration Target</span>
            </div>
            <div class="rb-menu-item">
                <span class="rb-menu-item-icon">🛡️</span>
                <span>Alpha Risk Stabilization Matrix</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_status_slot = st.empty()
    if st.session_state.get("model_loaded", False):
        model_status_slot.markdown(clean_html("""
        <div class="rb-bounce-card" style="background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.3); padding: 8px 12px; border-radius: 8px; display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span class="rb-pulse"></span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #10b981; font-weight: 700;">🟢 CORE PROTOCOLS ONLINE</span>
        </div>
        """), unsafe_allow_html=True)
    else:
        model_status_slot.markdown(clean_html("""
        <div class="rb-bounce-card" style="background: rgba(244,63,94,0.08); border: 1px solid rgba(244,63,94,0.3); padding: 8px 12px; border-radius: 8px; display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span class="rb-pulse" style="background:#f43f5e; box-shadow: 0 0 8px #f43f5e;"></span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #f43f5e; font-weight: 700;">⏳ Transceiver Link Pending</span>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("---")
    # Magnet Pill (Component 19)
    st.markdown('<p class="section-header"><span class="rb-gradienttext">Forecast Space Parameters</span></p>', unsafe_allow_html=True)
    st.markdown('<div class="rb-magnet-pill" style="padding: 2px 8px; font-size:0.65rem; color:#60a5fa; font-weight:600; display:inline-block; border-radius:20px; border:1px solid rgba(59,130,246,0.15)">GRAVITY ANCHOR POINT</div>', unsafe_allow_html=True)
    chosen_start_date = st.date_input("Anchor Execution Date", value=df.index[-1].date())
    forecast_days     = st.slider("Forecast Temporal Reach", 5, 60, 30, 5)

    st.markdown("---")
    st.markdown('<p class="section-header"><span class="rb-gradienttext">Alpha Risk Matrix</span></p>', unsafe_allow_html=True)
    entry_price  = st.number_input("Target Entry Level ($)", min_value=0.0, value=0.0, step=0.01)
    position_qty = st.number_input("Unit Target Density", min_value=1, value=10, step=1)
    risk_pct     = st.slider("Risk Tolerance Clip (%)", 1, 20, 5)

if not st.session_state.get("model_loaded", False):
    url_to_load = gdrive_url.strip() if gdrive_url else secret_url.strip()
    if url_to_load:
        f_id = extract_gdrive_id(url_to_load)
        if f_id:
            try:
                st.session_state["model_obj"] = load_model_cached(f_id)
                st.session_state["model_loaded"] = True
                model_status_slot.markdown(clean_html("""
                <div class="rb-bounce-card" style="background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.3); padding: 8px 12px; border-radius: 8px; display: flex; align-items: center; gap: 8px; margin-top: 10px;">
                    <span class="rb-pulse"></span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #10b981; font-weight: 700;">🟢 CORE PROTOCOLS ONLINE</span>
                </div>
                """), unsafe_allow_html=True)
            except Exception:
                st.session_state["model_loaded"] = False

if load_btn:
    url_clean = gdrive_url.strip() if gdrive_url else ""
    if url_clean:
        f_id = extract_gdrive_id(url_clean)
        if f_id:
            with st.sidebar:
                with st.spinner("Downloading dynamic core matrix..."):
                    try:
                        st.session_state["model_obj"] = load_model_cached(f_id)
                        st.session_state["model_loaded"] = True
                        model_status_slot.markdown(clean_html("""
                        <div class="rb-bounce-card" style="background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.4); padding: 8px 12px; border-radius: 8px; display: flex; align-items: center; gap: 8px; margin-top: 10px;">
                            <span class="rb-pulse"></span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #10b981; font-weight: 700;">✅ DEPLOYMENT STABILIZED</span>
                        </div>
                        """), unsafe_allow_html=True)
                    except Exception as _me:
                        model_status_slot.markdown(clean_html(f"""
                        <div class="rb-bounce-card" style="background: rgba(244,63,94,0.12); border: 1px solid rgba(244,63,94,0.4); padding: 8px 12px; border-radius: 8px; display: flex; align-items: center; gap: 8px; margin-top: 10px;">
                            <span class="rb-pulse" style="background:#f43f5e;"></span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #f43f5e; font-weight: 700;">❌ {_me}</span>
                        </div>
                        """), unsafe_allow_html=True)

model = st.session_state.get("model_obj", None)
eff_entry = entry_price if entry_price > 0.0 else current_price

f_dates, f_prices, f_lower, f_upper = None, None, None, None
b_dates, b_prices, b_lower, b_upper = None, None, None, None

if model is not None and scaler_ok:
    with st.spinner("Executing structural timeline simulations..."):
        try:
            f_dates, f_prices, f_lower, f_upper, b_dates, b_prices, b_lower, b_upper = dynamic_timeline_forecasting(
                model, scaler, df, pd.Timestamp(chosen_start_date), n_days=forecast_days
            )
        except Exception as _err:
            st.error(f"Matrix strategy fault: {_err}")

# ╔══════════════════════════════════════════════════════════╗
# ║                    MAIN DASHBOARD HERO                   ║
# ╚══════════════════════════════════════════════════════════╝

st.markdown(clean_html("""
<div class="app-header-banner rb-spotlightcard rb-noise-bg" style="position: relative; overflow: hidden; padding: 28px 32px; border-radius: 12px; border: 1px solid rgba(59,130,246,0.22); background: #0c101c;">
    <!-- GridMotion Background Grid (Component 5) -->
    <div class="rb-gridmotion" style="position: absolute; inset: 0; height: 100%; border: none; opacity: 0.16; pointer-events: none; z-index: 1;">
        <div class="rb-grid-lines"></div>
    </div>
    
    <div style="position: relative; z-index: 2; display: flex; justify-content: space-between; align-items: center; width: 100%; flex-wrap: wrap; gap: 16px;">
        <div>
            <!-- ShinyText Title (Component 6) -->
            <h1 class="app-title" style="margin: 0;"><span class="rb-shinytext" style="font-size: 1.85rem; font-family: 'Space Grotesk', sans-serif; font-weight: 700;">TSLA HYBRID FORECAST TERMINAL</span></h1>
            <!-- GradientText Subtitle (Component 7) -->
            <p class="app-subtitle" style="margin: 6px 0 0 0; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em;"><span class="rb-gradienttext">CNN-GRU NEURAL PREDICTIONS • HURST EXPOSURE REGIME COMPLIANCE • RECTIFIED DECAY SYSTEM</span></p>
        </div>
        <div style="text-align: right; display: flex; align-items: center; gap: 8px;">
            <!-- ScrambleText Status Tag (Component 10) -->
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: rgba(255,255,255,0.45); background: rgba(255,255,255,0.03); padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.06); margin-right: 8px;" class="rb-scrambletext">[ONLINE_ALPHA_TRANSCEIVER]</span>
            <!-- DigitalPulse Live Indicator (Component 21) -->
            <div style="background: rgba(16,185,129,0.07); border: 1px solid rgba(16,185,129,0.30); padding: 4px 12px; border-radius: 8px; display: inline-flex; align-items: center; gap: 8px;">
                <span class="rb-pulse"></span>
                <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; color: #10b981; letter-spacing: 0.1em; font-weight: 700;">TRANSCEIVER LIVE</span>
            </div>
        </div>
    </div>
</div>
"""), unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║                    KPI METRICS BANNER                    ║
# ╚══════════════════════════════════════════════════════════╝

col_t, col_p, col_d, col_v, col_sp = st.columns([2.5, 2, 2, 2, 2.5])
with col_t:
    st.markdown(metric_card("Database Terminal Target", "TSLA (NASDAQ)", f"Sync Anchor: {df.index[-1].strftime('%d %b %Y')}", "metric-muted"), unsafe_allow_html=True)
with col_p:
    d_cls = "metric-delta-up" if price_change >= 0 else "metric-delta-down"
    arrow = "▲" if price_change >= 0 else "▼"
    st.markdown(metric_card("Last Trading Close", f"${current_price:.2f}", f"{arrow} ${abs(price_change):.2f} ({price_change_p:+.2f}%)", d_cls), unsafe_allow_html=True)
with col_d:
    st.markdown(metric_card("52-Week High Threshold", f"${safe_float(df['High'].tail(252).max()):.2f}"), unsafe_allow_html=True)
with col_v:
    st.markdown(metric_card("52-Week Low Threshold", f"${safe_float(df['Low'].tail(252).min()):.2f}"), unsafe_allow_html=True)
with col_sp:
    v_val = df["Volume"].tail(20).mean()
    st.markdown(metric_card("20-Day Mean Volume", f"{v_val/1e6:.2f}M" if v_val>=1e6 else f"{v_val/1e3:.0f}K"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📡 SIGNAL RECEPTOR", "🔮 QUANT FORECAST ENGINE", "📊 TELEMETRY MATRIX"])

# ════════════════════════════════════════════════════════════
#  TAB 1 — SIGNAL COMPILER
# ════════════════════════════════════════════════════════════
with tab1:
    rsi_now  = safe_float(df["RSI"].dropna().iloc[-1] if df["RSI"].dropna().any() else np.nan)
    macd_now = safe_float(df["MACD"].dropna().iloc[-1] if df["MACD"].dropna().any() else np.nan)
    sig_now  = safe_float(df["MACDSig"].dropna().iloc[-1] if df["MACDSig"].dropna().any() else np.nan)
    ma30_now = safe_float(df["MA30"].dropna().iloc[-1] if df["MA30"].dropna().any() else np.nan)
    ma90_now = safe_float(df["MA90"].dropna().iloc[-1] if df["MA90"].dropna().any() else np.nan)
    bb_up    = safe_float(df["BB_Upper"].dropna().iloc[-1] if df["BB_Upper"].dropna().any() else np.nan)
    bb_lo    = safe_float(df["BB_Lower"].dropna().iloc[-1] if df["BB_Lower"].dropna().any() else np.nan)

    eff_entry = entry_price if entry_price > 0.0 else current_price

    tech_scores = {
        "RSI Indicator (14)": 1 if rsi_now < 35 else (-1 if rsi_now > 65 else 0),
        "MACD Convergence Zone": 1 if macd_now > sig_now else -1,
        "MA-30/90 Structural Crossing": 1 if ma30_now > ma90_now else -1,
        "BB Volatility Bounds": 1 if current_price < bb_lo else (-1 if current_price > bb_up else 0)
    }

    if f_prices is not None:
        model_target = safe_float(f_prices[min(4, len(f_prices)-1)])
        model_pct = (model_target - eff_entry) / eff_entry * 100
        tech_scores["NN Neural Forecast Drift"] = 1 if model_pct > 1.2 else (-1 if model_pct < -1.2 else 0)

    valuation_premium = (eff_entry - current_price) / current_price * 100
    if valuation_premium > 5.0:
        tech_scores["Structural Entry Guard"] = -2 
    elif valuation_premium < -5.0:
        tech_scores["Structural Entry Guard"] = 1
    else:
        tech_scores["Structural Entry Guard"] = 0

    total_score = sum(tech_scores.values())
    if total_score >= 2:    signal_label, signal_css = "BUY",  "signal-buy"
    elif total_score <= -2: signal_label, signal_css = "SELL", "signal-sell"
    else:                   signal_label, signal_css = "HOLD", "signal-hold"

    stop_loss    = eff_entry * (1 - risk_pct / 100)
    take_profit  = eff_entry + max(eff_entry - stop_loss, 0.01) * 2.0

    left, right = st.columns([1, 1.8], gap="large")
    with left:
        # Wrap Consolidated Alpha Vector inside a StarBorder Card (Component 14) 
        # with an Orb Light background glow (Component 4) 
        # and Floating GlowingRing animation (Component 22) 
        # and Bottom RollingCharacters Ticker (Component 25)!
        
        rolling_ticker_html = """
        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 14px; margin-top: 14px;">
            <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; color: #64748b; font-weight: 500; letter-spacing: 0.05em;">SYS TICK STATE (Component 25):</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #fff;">
                <span class="rb-roller" style="width: 50px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 4px; padding: 2px 6px; height: 1.45em; text-align: center;">
                    <span class="rb-roller-list" style="display: flex; flex-direction: column; height: 4.84em; line-height: 1.25em;">
                        <span style="color:#10b981;">BULL</span>
                        <span style="color:#f43f5e;">BEAR</span>
                        <span style="color:#ffcc00;">NEUT</span>
                        <span style="color:#3b82f6;">HALT</span>
                    </span>
                </span>
            </span>
        </div>
        """

        star_border_html = f"""
        <div class="rb-starborder-container" style="margin-top: 10px;">
            <div class="rb-starborder-anim"></div>
            <div class="rb-starborder-content" style="background: #060912; padding: 24px;">
                <!-- Component 4: Orb Light background blur -->
                <div class="rb-orb-viewport" style="position: absolute; inset: 0; background: transparent; pointer-events: none; height: 100%; width: 100%;">
                    <div class="rb-orb-light" style="opacity: 0.18; width: 110px; height: 110px; background: radial-gradient(circle, #ffaa11 0%, transparent 70%);"></div>
                </div>
                
                <h3 style="margin: 0 0 16px 0; font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 600; text-align: center; letter-spacing: 0.12em;"><span class="rb-gradienttext">CONSOLIDATED ALPHA VECTOR</span></h3>
                <div style="text-align: center; padding: 12px 0;">
                    <!-- Component 22: GlowingRing breathing pulse around the signal badge -->
                    <div class="{signal_css}" style="display: inline-block;">{signal_label}</div>
                </div>
                <p style="color: #abc; font-size: 0.72rem; text-align: center; margin: 12px 0 0 0; line-height: 1.4; font-family: 'Inter', sans-serif;">
                    Unified sentiment recommendation generated by combining technical boundaries and neural model drift indicators.
                </p>
                {rolling_ticker_html}
            </div>
        </div>
        """
        st.markdown(clean_html(star_border_html), unsafe_allow_html=True)
        
        st.markdown(clean_html('<p class="section-header" style="margin-top: 24px;"><span class="rb-gradienttext">Consensus Matrix Details</span></p>'), unsafe_allow_html=True)
        
        # Style list of technical factor scores using TrueFocus (Component 11) focus filters
        single_items_html = ""
        for name, sc in tech_scores.items():
            is_active = (sc >= 1 or sc <= -1)
            focus_class = "rb-truefocus-item active" if is_active else "rb-truefocus-item"
            
            if sc >= 1:
                single_items_html += f"""
                <div class="indicator-row {focus_class}" style="overflow: visible; width: 100%;">
                    <span class="ind-icon">🟢</span>
                    <span class="ind-name" style="font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem; font-weight: 600; color: #fff;">{name}</span>
                    <span class="ind-badge badge-bull" style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem;">ACCELERATIVE</span>
                </div>
                """
            elif sc == -1:
                single_items_html += f"""
                <div class="indicator-row {focus_class}" style="overflow: visible; width: 100%;">
                    <span class="ind-icon">🔴</span>
                    <span class="ind-name" style="font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem; font-weight: 600; color: #fff;">{name}</span>
                    <span class="ind-badge badge-bear" style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem;">DEPRESSIVE</span>
                </div>
                """
            elif sc <= -2:
                single_items_html += f"""
                <div class="indicator-row {focus_class}" style="overflow: visible; width: 100%;">
                    <span class="ind-icon">⚠️</span>
                    <span class="ind-name" style="font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem; font-weight: 600; color: #fff;">{name}</span>
                    <span class="ind-badge badge-warn" style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem;">OUTLIER LOCKOUT</span>
                </div>
                """
            else:
                # Inactive factor slightly blurred representing TrueFocus blur filter
                single_items_html += f"""
                <div class="indicator-row {focus_class}" style="overflow: visible; width: 100%;">
                    <span class="ind-icon">🟡</span>
                    <span class="ind-name" style="font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem; color: rgba(255,255,255,0.4);">{name}</span>
                    <span class="ind-badge badge-neut" style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem;">STABILIZED</span>
                </div>
                """
        
        indicator_details_html = f"""
        <div class="rb-infinite-scroll-container rb-consensus-matrix-container">
            <div class="rb-infinite-scroll-content">
                <div style="display: flex; flex-direction: column; gap: 8px; padding-bottom: 8px;">
                    {single_items_html}
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; padding-bottom: 8px;">
                    {single_items_html}
                </div>
            </div>
        </div>
        """
        st.markdown(clean_html(indicator_details_html), unsafe_allow_html=True)

        # Component 26: DiagnosticConsole Green CRT terminal displaying real calculations logs
        console_logs = f"""
        <div class="rb-console" style="margin-top: 18px; min-height: 50px; font-size: 0.72rem; overflow: hidden; max-height: 70px; padding: 12px 18px;">
            <div style="font-family: 'JetBrains Mono', monospace; color: #10b981; line-height: 1.4;">
                <span style="color:#fff">[signal]</span> System compiled score: {total_score:+} &rarr; Recommendation: {signal_label}
                <span class="rb-pulse" style="width: 6px; height: 6px; vertical-align: middle; margin-left: 4px;"></span>
            </div>
        </div>
        """
        st.markdown(clean_html(console_logs), unsafe_allow_html=True)

    with right:
        st.markdown(clean_html('<p class="section-header"><span class="rb-gradienttext">Dynamic Execution Space Map</span></p>'), unsafe_allow_html=True)
        
        # Upgrade execution points into distinct 3D TiltedCard widgets (Component 13) 
        # with an interactive LiquidProgress ball (Component 23) in stop/loss limit!
        t1_col1, t1_col2, t1_col3 = st.columns(3)
        with t1_col1:
            st.markdown(clean_html(f"""
            <div class="rb-tiltedcard" style="padding: 16px; border: 1.5px solid rgba(255,204,0,0.25); background: rgba(255,204,0,0.02); border-radius: 12px; min-height: 125px;">
                <div style="font-size: 0.65rem; color: #ffcc00; font-family: 'Space Grotesk', sans-serif; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Calibrated Entry</div>
                <div style="font-size: 1.55rem; font-weight: 700; color: #fff; font-family: 'JetBrains Mono', monospace; margin: 8px 0;" class="rb-shinytext">
                    ${eff_entry:.2f}
                </div>
                <div style="font-size: 0.68rem; color: #64748b; font-family: 'Inter', sans-serif;">Trigger Level Target</div>
            </div>
            """), unsafe_allow_html=True)
        with t1_col2:
            st.markdown(clean_html(f"""
            <div class="rb-tiltedcard" style="padding: 16px; border: 1.5px solid rgba(244,63,94,0.25); background: rgba(244,63,94,0.02); border-radius: 12px; min-height: 125px;">
                <div style="font-size: 0.65rem; color: #f43f5e; font-family: 'Space Grotesk', sans-serif; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Stop Loss Limit</div>
                <div style="font-size: 1.55rem; font-weight: 700; color: #f43f5e; font-family: 'JetBrains Mono', monospace; margin: 8px 0;">
                    ${stop_loss:.2f}
                </div>
                <div style="font-size: 0.68rem; color: #f43f5e; font-family: 'JetBrains Mono', monospace; font-weight: 600;">−{risk_pct}% Tolerance</div>
            </div>
            """), unsafe_allow_html=True)
        with t1_col3:
            # Map risk_pct range (1-20) to reservoir visual height range (15% to 85%) for LiquidProgress
            risk_fill_percentage = min(max(int(risk_pct * 4.5), 15), 85)
            st.markdown(clean_html(f"""
            <div class="rb-tiltedcard" style="padding: 12px 14px; border: 1.5px solid rgba(16,185,129,0.25); background: rgba(16,185,129,0.02); border-radius: 12px; min-height: 125px; display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                <div style="flex: 1;">
                    <div style="font-size: 0.65rem; color: #10b981; font-family: 'Space Grotesk', sans-serif; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Reward Target</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #10b981; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">
                        ${take_profit:.2f}
                    </div>
                    <div style="font-size: 0.68rem; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight: 600;">+{risk_pct*2}% Target</div>
                </div>
                <!-- Component 23: LiquidProgress loading ball representing actual risk percentage -->
                <div style="flex-shrink: 0; text-align: center;">
                    <div class="rb-liquid-ball" style="width: 52px; height: 52px; border-width: 1.5px;">
                        <div class="rb-liquid-wave" style="height: {risk_fill_percentage}%; background: rgba(16,185,129,0.72);"></div>
                        <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:0.62rem; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,0.6); z-index:4;">{risk_pct}%</div>
                    </div>
                    <div style="font-size: 0.52rem; color: #64748b; font-family: 'Space Grotesk', monospace; margin-top: 4px; font-weight:600;">RISK CLIP</div>
                </div>
            </div>
            """), unsafe_allow_html=True)
        
        fig_t = go.Figure()
        ctx_df = df.tail(90)
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["Close"], name="Close Core", line=dict(color=ACCENT, width=1.8)))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["MA30"], name="MA30 Node", line=dict(color=BLUE, width=1, dash="dash")))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["BB_Upper"], name="BB Upper Band", line=dict(color=MUTED, width=0.8, dash="dot")))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["BB_Lower"], name="BB Lower Band", line=dict(color=MUTED, width=0.8, dash="dot"), fill="tonexty", fillcolor="rgba(100,116,139,0.02)"))
        
        fig_t.add_hline(y=eff_entry, line_color=ACCENT, line_width=1.2, line_dash="solid", annotation_text="Calculated Entry Target", annotation_position="top left", annotation_font=dict(color=ACCENT, size=9, family="Space Grotesk"))
        fig_t.add_hline(y=take_profit, line_color=GREEN, line_width=1.2, line_dash="dash", annotation_text="Take Profit Threshold (1:2)", annotation_position="top left", annotation_font=dict(color=GREEN, size=9, family="Space Grotesk"))
        fig_t.add_hline(y=stop_loss, line_color=RED, line_width=1.2, line_dash="dash", annotation_text="Risk Stop Loss Line", annotation_position="bottom left", annotation_font=dict(color=RED, size=9, family="Space Grotesk"))

        fig_t.update_layout(**base_layout(320, "Price Action Vector vs Boundary Limits", override_yaxis=dict(tickprefix="$")))
        fig_t.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5
            )
        )
        
        # Wrap Plotly Chart with TechCardDecorator corners (Component 15)
        st.markdown('<div class="chart-wrap" style="position: relative; overflow: visible;">', unsafe_allow_html=True)
        st.markdown(clean_html("""
        <div class="rb-tech-bracket rb-bracket-tl" style="border-width: 2.5px 0 0 2.5px; border-color: #ffaa11;"></div>
        <div class="rb-tech-bracket rb-bracket-tr" style="border-width: 2.5px 2.5px 0 0; border-color: #ffaa11;"></div>
        <div class="rb-tech-bracket rb-bracket-bl" style="border-width: 0 0 2.5px 2.5px; border-color: #ffaa11;"></div>
        <div class="rb-tech-bracket rb-bracket-br" style="border-width: 0 2.5px 2.5px 0; border-color: #ffaa11;"></div>
        """), unsafe_allow_html=True)
        st.plotly_chart(fig_t, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — FORECAST STRATEGY
# ════════════════════════════════════════════════════════════
with tab2:
    if model is None:
        empty_state("🔮 NETWORK OFFLINE", "Model telemetry connection pending. Initialize the neural transceiver from the control console.")
    elif f_prices is None:
        empty_state("⛔ INTERFERENCE FAULT", "Timeline core compile mismatch.")
    else:
        st.markdown(f'<div style="background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.22); border-radius: 8px; padding: 12px 18px; font-family: \'JetBrains Mono\', monospace; font-size: 0.78rem; color: #a1b0cb; margin-bottom: 20px;">🌐 <strong>TEMPORAL SYNCHRONIZATION POINT:</strong> {pd.Timestamp(chosen_start_date).strftime("%A, %d %b %Y")} (UTC)</div>', unsafe_allow_html=True)
        
        f_end = safe_float(f_prices[-1])
        f_chg = ((f_end - current_price) / current_price * 100)

        # Style metric headers as responsive SpotlightCards using metric_card
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Simulation Anchor Price", f"${current_price:.2f}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Terminal Horizon Price", f"${f_end:.2f}", f"{f_chg:+.2f}%", "metric-delta-up" if f_chg>=0 else "metric-delta-down", show_scramble=True), unsafe_allow_html=True)
        c3.markdown(metric_card("Timeline Peak Bound", f"${safe_float(f_prices.max()):.2f}"), unsafe_allow_html=True)
        c4.markdown(metric_card("Timeline Trough Bound", f"${safe_float(f_prices.min()):.2f}"), unsafe_allow_html=True)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="Historical Real-Time Core", line=dict(color=ACCENT, width=1.8)))
        
        if b_dates is not None and len(b_dates) > 0:
            b_x = list(b_dates) + list(b_dates[::-1])
            b_y = list(b_upper) + list(b_lower[::-1])
            fig_fc.add_trace(go.Scatter(x=b_x, y=b_y, fill="toself", fillcolor="rgba(168,85,247,0.05)", line=dict(color="rgba(0,0,0,0)"), name="Bridge Uncertainty Band"))
            fig_fc.add_trace(go.Scatter(x=b_dates, y=b_prices, name="Context Bridge Trajectory", line=dict(color=PURPLE, width=1.5, dash="dash")))

        fx = list(f_dates) + list(f_dates[::-1])
        fy = list(f_upper) + list(f_lower[::-1])
        fig_fc.add_trace(go.Scatter(x=fx, y=fy, fill="toself", fillcolor="rgba(59,130,246,0.10)", line=dict(color="rgba(0,0,0,0)"), name="Forecast Uncertainty Band"))
        fig_fc.add_trace(go.Scatter(x=f_dates, y=f_prices, name="Model Hybrid Projection", line=dict(color=BLUE, width=2.0, dash="dashdot"), mode="lines+markers"))

        view_start = pd.Timestamp(chosen_start_date) - pd.DateOffset(months=2)
        view_end   = pd.Timestamp(f_dates[-1]) + pd.DateOffset(months=2)

        hist_in_view = df.loc[(df.index >= view_start) & (df.index <= view_end), "Adj Close"].dropna()
        all_visible_prices = list(hist_in_view.values) + list(f_prices)
        if b_prices is not None and len(b_prices) > 0:
            all_visible_prices += list(b_prices)
        all_visible_prices += list(f_lower) + list(f_upper)
        if b_lower is not None and len(b_lower) > 0:
            all_visible_prices += list(b_lower) + list(b_upper)

        all_visible_prices = [v for v in all_visible_prices if np.isfinite(v) and v > 0]
        if all_visible_prices:
            y_min   = min(all_visible_prices)
            y_max   = max(all_visible_prices)
            y_pad   = (y_max - y_min) * 0.08
            y_range = [y_min - y_pad, y_max + y_pad]
        else:
            y_range = None

        base_ly_params = base_layout(
            440,
            "Neural Engine Simulation Space Continuum",
            override_yaxis=dict(tickprefix="$", range=y_range) if y_range else dict(tickprefix="$"),
        )
        base_ly_params["xaxis"].update(dict(range=[view_start, view_end]))

        fig_fc.update_layout(**base_ly_params)
        fig_fc.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5
            )
        )

        # Re-divide bottom row of Tab 2 to support a dynamic StackedCards scenario panel (Component 17)
        left_f, right_f = st.columns([2.5, 1], gap="large")
        with left_f:
            st.markdown(clean_html('<div class="chart-wrap" style="position: relative; overflow: visible;">'), unsafe_allow_html=True)
            st.markdown(clean_html("""
            <div class="rb-tech-bracket rb-bracket-tl" style="border-width: 2.5px 0 0 2.5px; border-color: #3b82f6;"></div>
            <div class="rb-tech-bracket rb-bracket-tr" style="border-width: 2.5px 2.5px 0 0; border-color: #3b82f6;"></div>
            <div class="rb-tech-bracket rb-bracket-bl" style="border-width: 0 0 2.5px 2.5px; border-color: #3b82f6;"></div>
            <div class="rb-tech-bracket rb-bracket-br" style="border-width: 0 2.5px 2.5px 0; border-color: #3b82f6;"></div>
            """), unsafe_allow_html=True)
            st.plotly_chart(fig_fc, use_container_width=True)
            st.markdown(clean_html('</div>'), unsafe_allow_html=True)
            
        with right_f:
            st.markdown(clean_html('<p class="section-header" style="margin-top:0;"><span class="rb-gradienttext">Projected Scenario Stack</span></p>'), unsafe_allow_html=True)
            
            # StackedCards (Component 17) layout representing Bull, Median Expected, and Bear projections!
            st.markdown(clean_html(f"""
            <div class="rb-spotlightcard rb-stack-hover-trigger" style="padding: 16px; min-height: 380px; position: relative; background: #0c101c !important;">
                <!-- Component 4: Orb Light background glow behind scenario profiles stack -->
                <div class="rb-orb-viewport" style="position: absolute; inset: 0; background: transparent; pointer-events: none; height: 100%; width: 100%;">
                    <div class="rb-orb-light" style="opacity: 0.12; width: 120px; height: 120px; background: radial-gradient(circle, #a855f7 0%, transparent 70%);"></div>
                </div>
                
                <p style="color: #94a3b8; font-size: 0.72rem; line-height: 1.5; margin: 0 0 16px 0; font-family: 'Inter', sans-serif;">
                    Hover cursor over primary stack layers below to expand and separate scenario model values:
                </p>
                
                <div class="rb-stack-container" style="margin-top: 10px;">
                    <!-- Component 17 Stack Card 3: Bear Case (Red border) -->
                    <div class="rb-stack-card rb-stack-c3" style="border: 1px solid rgba(244,63,94,0.35); background: #0c080d; padding: 14px; display: flex; flex-direction: column; justify-content: space-between;">
                        <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem; color: #f43f5e; font-weight: 700; letter-spacing: 0.08em;">CASE 03: BEAR FLOOR</span>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight:700; color: #fff;">${f_prices.min() * 0.90:.2f}</div>
                        <p style="font-size: 0.58rem; color: #a1b0cb; line-height: 1.3; margin:0;">Negative drift cap correction: -10% volatility damping.</p>
                    </div>
                    <!-- Component 17 Stack Card 2: Median Case (Blue border) -->
                    <div class="rb-stack-card rb-stack-c2" style="border: 1px solid rgba(59,130,246,0.35); background: #070912; padding: 14px; display: flex; flex-direction: column; justify-content: space-between;">
                        <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem; color: #60a5fa; font-weight: 700; letter-spacing: 0.08em;">CASE 02: MEDIAN COMPLY</span>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight:700; color: #fff;">${f_prices[-1]:.2f}</div>
                        <p style="font-size: 0.58rem; color: #a1b0cb; line-height: 1.3; margin:0;">Expected neural output synced directly on Hurst parameters.</p>
                    </div>
                    <!-- Component 17 Stack Card 1: Bull Case (Gold border) -->
                    <div class="rb-stack-card rb-stack-c1" style="border: 1px solid rgba(255,204,0,0.45); background: #0c0a06; padding: 14px; display: flex; flex-direction: column; justify-content: space-between;">
                        <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem; color: #ffcc00; font-weight: 700; letter-spacing: 0.08em;">CASE 01: BULL ORBITAL</span>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight:700; color: #ffcc00;">${f_prices.max() * 1.10:.2f}</div>
                        <p style="font-size: 0.58rem; color: #ffeb99; line-height: 1.3; margin:0;">Optimistic model boundary with extra drift scaling factor.</p>
                    </div>
                </div>
                
                <p style="color: #64748b; font-size: 0.62rem; text-align: center; margin-top: 14px; font-family: 'JetBrains Mono', monospace; margin-bottom: 0;">
                    SCENARIO LAYER SEPARATION ACTIVE
                </p>
            </div>
            """), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 3 — TELEMETRY CONTROL MATRIX
# ════════════════════════════════════════════════════════════
with tab3:
    df_vis = df.copy()
    
    if b_dates is not None and len(b_dates) > 0:
        df_br = pd.DataFrame(index=b_dates)
        for c in ["Close", "Open", "High", "Low"]: df_br[c] = b_prices
        df_br["Volume"] = df["Volume"].iloc[-1]
        df_vis = pd.concat([df_vis, df_br])
        
    if f_prices is not None:
        df_f = pd.DataFrame(index=f_dates)
        df_f["Close"] = f_prices; df_f["Open"] = f_prices
        df_f["High"] = f_upper; df_f["Low"] = f_lower
        df_f["Volume"] = df["Volume"].iloc[-1]
        df_vis = pd.concat([df_vis, df_f])
        
    df_vis = df_vis[~df_vis.index.duplicated(keep='first')].sort_index()
    
    df_vis["Spread"] = df_vis["High"] - df_vis["Low"]
    df_vis["MA30"]   = df_vis["Close"].rolling(30).mean()
    df_vis["MA90"]   = df_vis["Close"].rolling(90).mean()
    df_vis["BB_Mid"] = df_vis["Close"].rolling(20).mean()
    v_std            = df_vis["Close"].rolling(20).std()
    df_vis["BB_Upper"] = df_vis["BB_Mid"] + 2 * v_std
    df_vis["BB_Lower"] = df_vis["BB_Mid"] - 2 * v_std
    
    df_vis["EMA12"] = df_vis["Close"].ewm(span=12, adjust=False).mean()
    df_vis["EMA26"] = df_vis["Close"].ewm(span=26, adjust=False).mean()
    df_vis["MACD"]  = df_vis["EMA12"] - df_vis["EMA26"]
    df_vis["MACDSig"] = df_vis["MACD"].ewm(span=9, adjust=False).mean()
    df_vis["MACDHist"] = df_vis["MACD"] - df_vis["MACDSig"]
    
    d_v = df_vis["Close"].diff()
    g_v = d_v.clip(lower=0).rolling(14).mean()
    l_v = (-d_v.clip(upper=0)).rolling(14).mean()
    df_vis["RSI"] = 100 - (100 / (1 + (g_v / l_v.replace(0, np.nan))))

    fa, fb = st.columns(2)
    with fa: viz_start = st.date_input("Vector Frame Start", value=df.index[-90].date(), min_value=df.index[0].date(), max_value=df_vis.index[-1].date(), key="v_start")
    with fb: viz_end   = st.date_input("Vector Frame Stop", value=df_vis.index[-1].date(), min_value=df.index[0].date(), max_value=df_vis.index[-1].date(), key="v_end")

    if viz_start >= viz_end:
        st.warning("⚠️ Frame Index Exception. Correct inputs.")
        st.stop()

    dv = df_vis.loc[str(viz_start):str(viz_end)].copy()

    # Calculate figures inside the local frame buffer
    # Fig 1: Continuous Close Pricing Sequence
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dv.index, y=dv["Close"], name="Unified Close Connection", line=dict(color=ACCENT, width=1.5)))
    for c, col_color in [("MA30", BLUE), ("MA90", PURPLE)]:
        if c in dv.columns and dv[c].notna().any():
            fig1.add_trace(go.Scatter(x=dv.index, y=dv[c], name=c, line=dict(color=col_color, width=1, dash="dot")))
    fig1.update_layout(**base_layout(480, ""))

    # Fig 2: Segmented Bar Volume Distribution
    v_cols = [GREEN if i==0 else (GREEN if dv["Close"].iloc[i]>=dv["Close"].iloc[i-1] else RED) for i in range(len(dv))]
    fig2 = go.Figure(go.Bar(x=dv.index, y=dv["Volume"], marker_color=v_cols, name="Volume Stream"))
    fig2.update_layout(**base_layout(480, ""))

    # Fig 3: Holographic Structural Candlestick Envelope
    fig3 = go.Figure(go.Candlestick(x=dv.index, open=dv["Open"], high=dv["High"], low=dv["Low"], close=dv["Close"], increasing_line_color=GREEN, decreasing_line_color=RED, name="OHLC Candlestick"))
    if dv["BB_Upper"].notna().any():
        fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Upper"], name="Volatility Cell Upper", line=dict(color=MUTED, width=0.8, dash="dash")))
        fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Lower"], fill="tonexty", fillcolor="rgba(100,116,139,0.02)", name="Volatility Cell Lower", line=dict(color=MUTED, width=0.8, dash="dash")))
    fig3.update_layout(**base_layout(480, ""))
    fig3.update_layout(xaxis_rangeslider_visible=False)

    # Fig 4: Intraday Dispersion Bounds
    fig4 = go.Figure(go.Scatter(x=dv.index, y=dv["Spread"], fill="tozeroy", fillcolor="rgba(255,204,0,0.06)", line=dict(color=ACCENT, width=1.0)))
    fig4.update_layout(**base_layout(480, ""))

    # Fig 5: Momentum Convergence/Divergence Oscillator (MACD)
    fig5 = None
    if dv["MACD"].notna().any():
        fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig5.add_trace(go.Scatter(x=dv.index, y=dv["MACD"], name="MACD Vector", line=dict(color=BLUE, width=1.2)), row=1, col=1)
        fig5.add_trace(go.Scatter(x=dv.index, y=dv["MACDSig"], name="MACD Signal", line=dict(color=ACCENT, width=1.2)), row=1, col=1)
        h_colors = [GREEN if val >= 0 else RED for val in dv["MACDHist"].fillna(0)]
        fig5.add_trace(go.Bar(x=dv.index, y=dv["MACDHist"], name="Histogram Matrix", marker_color=h_colors), row=2, col=1)
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,16,28,0.40)", font_color=FONT_COL, height=480, margin=dict(l=40,r=20,t=10,b=20), showlegend=False)
        fig5.update_xaxes(gridcolor=GRID_COL); fig5.update_yaxes(gridcolor=GRID_COL)

    # Fig 6: RSI Engine
    fig6 = None
    if dv["RSI"].notna().any():
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=dv.index, y=dv["RSI"], name="RSI Engine", line=dict(color=PURPLE, width=1.2)))
        fig6.add_hrect(y0=70, y1=100, fillcolor="rgba(244,63,94,0.03)", line_width=0)
        fig6.add_hrect(y0=0,  y1=30,  fillcolor="rgba(16,185,129,0.03)", line_width=0)
        fig6.add_hline(y=70, line_color=RED, line_dash="dash", line_width=0.8)
        fig6.add_hline(y=30, line_color=GREEN, line_dash="dash", line_width=0.8)
        fig6.update_layout(**base_layout(480, "", override_yaxis=dict(range=[0, 100])))

    # Fig 7: Macro Annualized Core Price Assets
    yearly = dv.groupby(dv.index.year)["Close"].mean().reset_index()
    fig7 = go.Figure(go.Bar(x=yearly.iloc[:, 0].astype(str), y=yearly["Close"], marker_color=ACCENT))
    fig7.update_layout(**base_layout(480, ""))

    # Fig 8: Seasonality Distribution
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig8 = go.Figure()
    for m_idx, m_name in enumerate(months, 1):
        sub = dv[dv.index.month == m_idx]["Close"].dropna()
        if not sub.empty: fig8.add_trace(go.Box(y=sub, name=m_name, marker_color=BLUE, line_color=BLUE, fillcolor="rgba(59,130,246,0.12)"))
    fig8.update_layout(**base_layout(480, ""), showlegend=False)

    # Fig 11: Correlation matrix
    fig11 = None
    corr_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Spread"] if c in dv.columns]
    corr_data = dv[corr_cols].dropna()
    if len(corr_data) >= 5 and len(corr_cols) >= 2:
        c_mat = corr_data.corr().round(3)
        fig11 = go.Figure(go.Heatmap(z=c_mat.values, x=corr_cols, y=corr_cols, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, text=c_mat.values, texttemplate="%{text:.2f}", showscale=True))
        fig11.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,16,28,0.40)", font_color=FONT_COL, height=480, margin=dict(l=40, r=20, t=10, b=40))

    # Output the dynamic horizontal scroll sequence frame (Flying Posters - reactbits)
    st.markdown('<div class="flying-posters-viewport">', unsafe_allow_html=True)

    # Poster 1: Continuous Close Pricing Sequence
    st.markdown('''
        <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                📈 CONTINUOUS CLOSE PRICING SEQUENCE
            </div>
    ''', unsafe_allow_html=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Poster 2: Segmented Bar Volume Distribution
    st.markdown('''
        <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                📊 SEGMENTED BAR VOLUME DISTRIBUTION
            </div>
    ''', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Poster 3: Holographic Candlestick Envelope
    st.markdown('''
        <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                🕯️ HOLOGRAPHIC CANDLESTICK ENVELOPE
            </div>
    ''', unsafe_allow_html=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Poster 4: Intraday Dispersion Bounds Variance
    st.markdown('''
        <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                📈 INTRADAY DISPERSION BOUNDS VARIANCE
            </div>
    ''', unsafe_allow_html=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Poster 5: Momentum Convergence Contour
    if fig5 is not None:
        st.markdown('''
            <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                    📉 MOMENTUM CONVERGENCE CONTOUR
                </div>
        ''', unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Poster 6: Strength Velocity RSI Oscillator
    if fig6 is not None:
        st.markdown('''
            <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                    📶 STRENGTH VELOCITY RSI OSCILLATOR
                </div>
        ''', unsafe_allow_html=True)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Poster 7: Macro Annualized Price Assets
    st.markdown('''
        <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                📅 MACRO ANNUALIZED PRICE ASSETS
            </div>
    ''', unsafe_allow_html=True)
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Poster 8: Seasonality Distribution Matrices
    st.markdown('''
        <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                ❄️ SEASONALITY DISTRIBUTION MATRICES
            </div>
    ''', unsafe_allow_html=True)
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Poster 9: Attribute Correlation Matrix
    if fig11 is not None:
        st.markdown('''
            <div class="flying-poster-card" style="display: flex; flex-direction: column; align-items: center; justify-content: flex-start;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #ffcc00; letter-spacing: 0.08em; margin: 14px 0 6px 0; text-transform: uppercase;">
                    🧬 ATTRIBUTE CORRELATION MATRIX
                </div>
        ''', unsafe_allow_html=True)
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # End of viewport container

# ════════════════════════════════════════════════════════════
#  TAB 4 — REACTBITS SHOWROOM (DEPRECATED — INTEGRATED DIRECTLY INTO CORES)
# ════════════════════════════════════════════════════════════
if False:
    st.markdown("""
    <div style="background: rgba(168,85,247,0.06); border: 1px solid rgba(168,85,247,0.22); border-radius: 12px; padding: 24px; margin-bottom: 24px;">
        <h2 style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #ffcc00; margin: 0 0 8px 0; font-size: 1.6rem;"><span class="rb-gradienttext">REACTBITS LABORATORY</span></h2>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0; line-height: 1.6;">Welcome to the Unified Shadow Component Registry. Here, <strong>exactly 27 different visual components</strong> sourced directly from the <span style="color: #ffcc00; font-weight: 600;">reactbits.dev</span> library are custom-engineered and fully executed directly in Python using optimized CSS3 acceleration, fluid transitions, and vector rendering. Hover, click, and interact with each element below.</p>
    </div>
    """, unsafe_allow_html=True)

    # Category 1: Atmospheric Framework
    st.markdown('<p class="section-header">1. Atmospheric & Spatial Frameworks</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        # component 1: Aurora
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #a855f7; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 8px;">REACTBITS: Aurora</div>
            <div class="rb-aurora-viewport">
                <div class="rb-aurora-blend"></div>
                <div class="rb-aurora-blob"></div>
                <div style="position: absolute; bottom: 12px; left: 12px; color: #fff; font-family: 'Space Grotesk', sans-serif; font-size: 0.8rem; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">Plasma Aurora Drift</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Slow-orbiting multi-spectral radial gradients that morph and blend dynamically inside a soft mask container.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        # component 2: DotGrid
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #ff9900; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 8px;">REACTBITS: DotGrid</div>
            <div class="rb-dotgrid">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; color: #ffaa11; text-align: center;">Active Grid Portal</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.61rem; color: rgba(255,255,255,0.4); text-align:center; margin-top:6px;">SPACING: 14PX</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Procedural dotted coordinate matrix backplane representing an infinity-grid for modern dark industrial HUD workspaces.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        # component 3: Noise Overlay
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #06b6d4; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 8px;">REACTBITS: Noise Overlay</div>
            <div class="rb-noise-bg">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 600; color: #e2e8f0; text-align:center;">Grain Texture Module</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #10b981; text-align: center; margin-top: 8px;">65% GAUSS</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Micro-grain fractal noise bitmap layer acting as a physical cinematic texture filter across standard HTML card layers.</p>
        </div>
        """, unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    with c4:
        # component 4: Orb Light
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #ec4899; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 8px;">REACTBITS: Orb</div>
            <div class="rb-orb-viewport">
                <div class="rb-orb-light"></div>
                <div style="position: absolute; top: 12px; right: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: rgba(255,255,255,0.4);">PROP: BLUR 40PX</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Floating blurred circular energy nucleus that shifts coordinates back-and-forth along a dynamic trigonometric pathway.</p>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        # component 5: GridMotion
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #6366f1; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 8px;">REACTBITS: GridMotion</div>
            <div class="rb-gridmotion">
                <div class="rb-grid-lines"></div>
                <div style="position: absolute; bottom: 8px; right: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #06b6d4; font-weight:600;">SPEED: 12S/LOOP</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Perspective grid lines animating horizontally to resemble endless motion telemetry grids, running smoothly on hardware GPU threads.</p>
        </div>
        """, unsafe_allow_html=True)

    # Category 2: Chromatic Typography
    st.markdown('<p class="section-header">2. Chromatic & Kinematic Typography</p>', unsafe_allow_html=True)
    ct1, ct2, ct3 = st.columns(3)
    with ct1:
        # component 6: ShinyText
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.65rem; color: #ffaa11; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: ShinyText</div>
            <div style="margin: 12px 0;"><span class="rb-shinytext" style="font-size: 1.25rem;">GLIDE VELOCITY ALPHA</span></div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Animates an angled, semi-transparent white-gold reflection beam sliding across text spans recursively.</p>
        </div>
        """, unsafe_allow_html=True)
    with ct2:
        # component 7: GradientText
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.65rem; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: GradientText</div>
            <div style="margin: 12px 0;"><span class="rb-gradienttext" style="font-size: 1.25rem;">SPECTRAL WAVEFORM</span></div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Continuously cycles text fill along an 8-color gradient spectrum utilizing high-performance background-clip shifts.</p>
        </div>
        """, unsafe_allow_html=True)
    with ct3:
        # component 8: BlurText
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.65rem; color: #a855f7; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: BlurText</div>
            <div style="margin: 12px 0;"><span class="rb-blurtext" style="font-size: 1.25rem; font-weight:700; color: #ffcc00; text-shadow:0 0 8px rgba(255,204,0,0.2);">FOCUS RESOLVING</span></div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Triggers a visual focus entrance state where text shifts from maximum gaussian blur to crystal sharpness on rendering.</p>
        </div>
        """, unsafe_allow_html=True)

    ct4, ct5, ct6 = st.columns(3)
    with ct4:
        # component 9: SplitText
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.65rem; color: #f43f5e; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: SplitText</div>
            <div style="margin: 12px 0;"><span class="rb-splittext" style="font-size: 1.3rem; font-weight: 700; color: #fff;">[EXPAND]</span></div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Letter-spacing transition array that reacts as you hover over characters, expanding spatial intervals with fluid feedback.</p>
        </div>
        """, unsafe_allow_html=True)
    with ct5:
        # component 10: ScrambleText
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.65rem; color: #06b6d4; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: ScrambleText</div>
            <div style="margin: 12px 0;"><span class="rb-scrambletext">Ø9_NΞURÆL_CØRΞ</span></div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Simulates military decryption logs by rapidly cycling through random cyber characters until final stabilization.</p>
        </div>
        """, unsafe_allow_html=True)
    with ct6:
        # component 11: TrueFocus
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.65rem; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 11px;">REACTBITS: TrueFocus</div>
            <div style="margin: 8px 0; font-family:'Space Grotesk',sans-serif; text-align: center;">
                <span class="rb-truefocus-item">TSLA</span>
                <span class="rb-truefocus-item active" style="border: 1px solid rgba(255,204,0,0.5); padding: 2px 6px; border-radius:4px;">CORE</span>
                <span class="rb-truefocus-item">UNIT</span>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4; text-align: left;">Isolates targeted vocabulary blocks inside absolute mechanical brackets while blurring out surrounding word blocks.</p>
        </div>
        """, unsafe_allow_html=True)

    # Category 3: Card & Layout Architecture
    st.markdown('<p class="section-header">3. Precision Card & Layout Architecture</p>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        # component 12: SpotlightCard
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #06b6d4; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 10px;">REACTBITS: SpotlightCard</div>
            <div style="border: 1px dashed rgba(255,255,255,0.08); border-radius: 8px; padding: 14px; text-align:center;">
                <div style="font-family:'Space Grotesk',sans-serif; font-size: 0.9rem; font-weight:600; color:#fff;">SPOTLIGHT ACTIVE</div>
                <div style="font-size:0.55rem; color:rgba(255,255,255,0.4); margin-top:4px;">HOVER CURSOR TO REVEAL HALO</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Tracks cursor positions relative to boundaries to draw a smooth radial lighting gradient dynamically.</p>
        </div>
        """, unsafe_allow_html=True)
    with cc2:
        # component 13: TiltedCard
        st.markdown("""
        <div class="rb-tiltedcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #a855f7; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 10px;">REACTBITS: TiltedCard</div>
            <div style="border: 1px solid rgba(168,85,247,0.15); background:rgba(168,85,247,0.02); border-radius: 8px; padding: 14px; text-align:center;">
                <div style="font-family:'Space Grotesk',sans-serif; font-size: 0.9rem; font-weight:600; color:#fff;">3D DEPTH CARD</div>
                <div style="font-size:0.55rem; color:rgba(255,255,255,0.4); margin-top:4px;">HOVER FOR 3D PERSPECTIVE</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Uses CSS matrix transforms to tilt, scale, and adjust perspective shadows cleanly as the pointer moves across bounds.</p>
        </div>
        """, unsafe_allow_html=True)
    with cc3:
        # component 14: StarBorder Card
        st.markdown("""
        <div class="rb-starborder-container">
            <div class="rb-starborder-anim"></div>
            <div class="rb-starborder-content">
                <div style="font-size: 0.65rem; color: #ffaa11; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 8px;">REACTBITS: StarBorder</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size: 0.95rem; font-weight:600; color:#fff;">Cosmic Star Frame</div>
                <p style="color: #64748b; font-size: 0.72rem; margin: 6px 0 0 0; line-height: 1.4;">Continuous revolving conic gradient that runs inside a thin frame mask, generating an elegant edge-glow ribbon structure.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    cc4, cc5, cc6 = st.columns(3)
    with cc4:
        # component 15: TechCardDecorator
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; position:relative; overflow:visible;">
            <div class="rb-tech-bracket rb-bracket-tl"></div>
            <div class="rb-tech-bracket rb-bracket-tr"></div>
            <div class="rb-tech-bracket rb-bracket-bl"></div>
            <div class="rb-tech-bracket rb-bracket-br"></div>
            <div style="font-size: 0.65rem; color: #ffcc00; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 10px;">REACTBITS: TechCardDecorator</div>
            <div style="text-align: center; margin: 14px 0 8px 0;">
                <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#ffcc00; background:rgba(255,204,0,0.08); padding:4px 10px; border-radius:4px; font-weight:600; border:1px solid rgba(255,204,0,0.15)">IND_MODULE_01</span>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Places absolute retro-futuristic geometric gold L-shaped metal brackets on structural board corners to enhance mechanical telemetry feel.</p>
        </div>
        """, unsafe_allow_html=True)
    with cc5:
        # component 16: BounceCard
        st.markdown("""
        <div class="rb-bounce-card" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 10px;">REACTBITS: BounceCard</div>
            <div style="border: 1px solid rgba(16,185,129,0.15); background:rgba(16,185,129,0.02); border-radius: 8px; padding: 14px; text-align:center;">
                <div style="font-family:'Space Grotesk',sans-serif; font-size: 0.9rem; font-weight:600; color:#10b981;">ELASTIC SPRING HOVER</div>
                <div style="font-size:0.55rem; color:rgba(255,255,255,0.4); margin-top:4px;">REBOUNDS ON ENTRANCE</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Applies an elastic cubic-bezier timing function (`0.175, 0.885, 0.45, 1.4`) that causes cards to rebound and spring upon cursor entry.</p>
        </div>
        """, unsafe_allow_html=True)
    with cc6:
        # component 17: StackedCards
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #ff3366; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 10px;">REACTBITS: StackedCard</div>
            <div class="rb-stack-container">
                <div class="rb-stack-card rb-stack-c3" style="text-align: center; color:rgba(255,255,255,0.1); border-color:#ff3366;">LAYER 03</div>
                <div class="rb-stack-card rb-stack-c2" style="text-align: center; color:rgba(255,255,255,0.4); border-color:#06b6d4;">LAYER 02</div>
                <div class="rb-stack-card rb-stack-c1" style="text-align: center; color:#fff; display:flex; flex-direction:column; justify-content:center; align-items:center; border-color:#ffaa11;">
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:0.75rem; font-weight:700;">ACTIVE METRIC SHEET</div>
                </div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Creates an overlay layered stack representing cascading profiles which separate and highlight beautifully when hovered.</p>
        </div>
        """, unsafe_allow_html=True)

    # Category 4: Interaction Elements
    st.markdown('<p class="section-header">4. Dynamic Feedback & Interaction Elements</p>', unsafe_allow_html=True)
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        # component 18: Ripple Button
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center;">
            <div style="font-size: 0.65rem; color: #3b82f6; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: Ripple</div>
            <div class="rb-ripple-btn">TRIGGER RADIAL RIPPLE</div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Uses absolute pseudo-elements expanding in radius within a clipped container to emulate active circular liquid ripples on click.</p>
        </div>
        """, unsafe_allow_html=True)
    with ci2:
        # component 19: Magnet Pill
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center;">
            <div style="font-size: 0.65rem; color: #ffaa11; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: Magnetic</div>
            <div class="rb-magnet-pill">MAGNETIC FIELD PILL</div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Slightly pulls, translates and scales navigation indicators closer to the cursor coordinates when entering the trigger threshold.</p>
        </div>
        """, unsafe_allow_html=True)
    with ci3:
        # component 20: ShimmerButton
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center;">
            <div style="font-size: 0.65rem; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: ShimmerButton</div>
            <div class="rb-shimmer-btn">ESTABLISH CHANNELS</div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Fires a beautiful, continuous high-contrast glossy sliding horizontal linear gradient across primary control action paths.</p>
        </div>
        """, unsafe_allow_html=True)

    ci4, ci5, ci6 = st.columns(3)
    with ci4:
        # component 21: DigitalPulse
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center;">
            <div style="font-size: 0.65rem; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: DigitalPulse</div>
            <div style="margin: 18px auto 14px auto;">
                <span class="rb-pulse"></span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700; color:#10b981; margin-left:8px;">HEARTBEAT ONLINE</span>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 14px 0 0 0; line-height: 1.4; text-align: left;">Creates a steady state diagnostic blinking ring around heart indicators mimicking operational real-time node pings.</p>
        </div>
        """, unsafe_allow_html=True)
    with ci5:
        # component 22: GlowingRing
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center;">
            <div style="font-size: 0.65rem; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: GlowingRing</div>
            <div style="margin: 10px 0;">
                <div class="rb-glowingring-box">
                    <span style="font-size:1.1rem;">⚡</span>
                </div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4; text-align: left;">Emits cyclic outward circular waves using timed expanding glowing box shadows to reflect critical triggers.</p>
        </div>
        """, unsafe_allow_html=True)
    with ci6:
        # component 23: LiquidProgress
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center;">
            <div style="font-size: 0.65rem; color: #ffcc00; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: LiquidProgress</div>
            <div class="rb-liquid-ball">
                <div class="rb-liquid-wave"></div>
                <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:0.9rem; color:#fff; text-shadow:0 1px 3px rgba(0,0,0,0.6); z-index:4;">60%</div>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4; text-align: left;">Renders a clipping container with wavy absolute spinning overlay boundaries representing dynamic reservoir height loaders.</p>
        </div>
        """, unsafe_allow_html=True)

    # Category 5: Data & Telemetry Components
    st.markdown('<p class="section-header">5. Live Data & Telemetry Matrix</p>', unsafe_allow_html=True)
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        # component 24: CountUp
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.65rem; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: CountUp</div>
            <div style="margin: 10px 0;">
                <span class="rb-odometer">
                    <span>$224.58</span>
                </span>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Runs high-efficiency CSS transitions on layout lines to scramble digits upwards and settle smoothly upon metric reloads.</p>
        </div>
        """, unsafe_allow_html=True)
    with cd2:
        # component 25: RollingCharacters
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px; text-align: center; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.65rem; color: #60a5fa; font-family: 'JetBrains Mono', monospace; font-weight:700; text-align: left; margin-bottom: 12px;">REACTBITS: RollingCharacters</div>
            <div style="margin: 10px 0; font-family:'JetBrains Mono', monospace; font-size: 1.15rem; font-weight:700; color:#fff;">
                TSLA STATE: 
                <span class="rb-roller">
                    <span class="rb-roller-list" style="display: flex; flex-direction: column; height: 4.84em; line-height:1.21em;">
                        <span style="color:#10b981;">BULL</span>
                        <span style="color:#f43f5e;">BEAR</span>
                        <span style="color:#ffcc00;">NEUT</span>
                        <span style="color:#60a5fa;">HALT</span>
                    </span>
                </span>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 12px 0 0 0; line-height: 1.4; text-align: left;">Executes infinitely repeating structural vertical translations of ticker character indices inside structural masked views.</p>
        </div>
        """, unsafe_allow_html=True)
    with cd3:
        # component 26: DiagnosticConsole
        st.markdown("""
        <div class="rb-spotlightcard" style="padding: 16px;">
            <div style="font-size: 0.65rem; color: #10b981; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 10px;">REACTBITS: DiagnosticConsole</div>
            <div class="rb-console">
                <span style="color:rgba(16,185,129,0.5)">[system]</span> core init...<br>
                <span style="color:#ffcc00">TSLA Price:</span> $220.00<br>
                <span style="color:#06b6d4">Hurst index:</span> 0.548<br>
                <span class="rb-pulse" style="width:6px; height:6px; vertical-align:middle; margin-left:4px;"></span>
            </div>
            <p style="color: #64748b; font-size: 0.72rem; margin: 10px 0 0 0; line-height: 1.4;">Styled green-on-black CRT terminal shell mimicking dynamic scrolling log arrays with static scans overlays.</p>
        </div>
        """, unsafe_allow_html=True)

    # Wrap up: Component 27 - ParticlesBg (Background Particles simulation in laboratory)
    st.markdown('<p class="section-header">6. Particle Universe backplanes</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="rb-starborder-container" style="margin-bottom: 24px;">
        <div class="rb-starborder-anim"></div>
        <div class="rb-starborder-content" style="background:#04060c; display:flex; align-items:center; flex-direction:column; padding:24px; text-align:center;">
            <div style="font-size: 0.65rem; color: #ff3366; font-family: 'JetBrains Mono', monospace; font-weight:700; margin-bottom: 8px;">REACTBITS: ParticlesBg (Component 27)</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:700; color:#fff; margin-bottom:12px;">Active Floating Cosmic Backplane</div>
            <p style="color:#94a3b8; font-size:0.8rem; max-width:650px; margin:0 auto 16px auto; line-height:1.6;">Renders dozens of lightweight, responsive, hardware-accelerated vectors drifting within the container space. This serves as a magnificent live backplane to display critical information under modern, high-contrast dark visual interfaces in absolute spatial alignment.</p>
            <div style="display:flex; gap:12px; justify-content:center; align-items:center;">
                <div style="background:rgba(255,170,17,0.1); border:1px solid rgba(255,170,17,0.3); padding:4px 14px; border-radius:12px; font-size:0.7rem; font-family:'JetBrains Mono',monospace; color:#ffaa11; font-weight:600;">ACTIVE COSMOS VECTOR LAYER</div>
                <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); padding:4px 14px; border-radius:12px; font-size:0.7rem; font-family:'JetBrains Mono',monospace; color:#10b981; font-weight:600;">30FPS RAF RUNTIME</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
