import { useState, useRef, useCallback } from "react";

// ── CONFIG — change this to your backend URL ──────────────────────────────
const API_BASE = "http://localhost:5000";
// In production: const API_BASE = "https://your-domain.com";
// ──────────────────────────────────────────────────────────────────────────

const steps = ["Upload", "Scoring", "Recommendations", "Enhanced Poster"];

const ScoreRing = ({ score, label, color, animate }) => {
  const radius = 36;
  const circ   = 2 * Math.PI * radius;
  const offset = circ - (score / 100) * circ;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <svg width="90" height="90" viewBox="0 0 90 90">
        <circle cx="45" cy="45" r={radius} fill="none" stroke="#1a1a2e" strokeWidth="7" />
        <circle
          cx="45" cy="45" r={radius} fill="none"
          stroke={color} strokeWidth="7"
          strokeDasharray={circ}
          strokeDashoffset={animate ? offset : circ}
          strokeLinecap="round"
          transform="rotate(-90 45 45)"
          style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)" }}
        />
        <text x="45" y="50" textAnchor="middle" fill={color} fontSize="16"
          fontFamily="'Bebas Neue', sans-serif" fontWeight="700">{score}</text>
      </svg>
      <span style={{ fontSize: 11, color: "#8a8aaa", fontFamily: "'DM Sans', sans-serif",
        letterSpacing: "0.08em", textTransform: "uppercase" }}>{label}</span>
    </div>
  );
};

const Tag = ({ text, type }) => {
  const colors = {
    critical: { bg: "rgba(255,80,80,0.12)",   border: "rgba(255,80,80,0.3)",   text: "#ff6060" },
    improve:  { bg: "rgba(255,180,0,0.12)",   border: "rgba(255,180,0,0.3)",   text: "#ffb800" },
    good:     { bg: "rgba(0,210,130,0.12)",   border: "rgba(0,210,130,0.3)",   text: "#00d282" },
  };
  const c = colors[type] || colors.improve;
  return (
    <span style={{
      background: c.bg, border: `1px solid ${c.border}`, color: c.text,
      borderRadius: 6, padding: "3px 10px", fontSize: 11,
      fontFamily: "'DM Sans', sans-serif", fontWeight: 600, letterSpacing: "0.05em",
    }}>{text}</span>
  );
};

const ErrorBanner = ({ message, onDismiss }) => (
  <div style={{
    margin: "16px 0", padding: "14px 18px", borderRadius: 12,
    background: "rgba(255,80,80,0.1)", border: "1px solid rgba(255,80,80,0.3)",
    display: "flex", alignItems: "center", justifyContent: "space-between",
  }}>
    <span style={{ color: "#ff8080", fontSize: 13 }}>⚠ {message}</span>
    <button onClick={onDismiss} style={{ background: "none", border: "none", color: "#ff6060", cursor: "pointer", fontSize: 16 }}>✕</button>
  </div>
);

export default function App() {
  const [step, setStep]               = useState(0);
  const [dragging, setDragging]       = useState(false);
  const [file, setFile]               = useState(null);
  const [preview, setPreview]         = useState(null);
  const [loading, setLoading]         = useState(false);
  const [loadingMsg, setLoadingMsg]   = useState("");
  const [loadingPct, setLoadingPct]   = useState(0);
  const [scores, setScores]           = useState(null);
  const [recommendations, setRecs]   = useState(null);
  const [features, setFeatures]       = useState(null);
  const [enhancedPoster, setEnhanced] = useState(null);
  const [changesApplied, setChanges]  = useState([]);
  const [error, setError]             = useState(null);
  const [ringsAnimated, setRingsAnim] = useState(false);
  const fileRef = useRef();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please upload a valid image file (PNG, JPG, WEBP).");
      return;
    }
    setFile(f);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  // ── Step 1 → 2: Call /api/score ──────────────────────────────────────────
  const runScoring = async () => {
    if (!file) return;
    setError(null);
    setStep(1);
    setLoading(true);
    setLoadingPct(10);

    const msgSteps = [
      [10,  "Extracting visual features…"],
      [30,  "Running ResNet50 CNN encoder…"],
      [55,  "Applying PCA dimensionality reduction…"],
      [75,  "Computing TOPSIS scores…"],
      [90,  "Detecting design issues…"],
    ];
    let msgIdx = 0;
    const ticker = setInterval(() => {
      if (msgIdx < msgSteps.length) {
        const [pct, msg] = msgSteps[msgIdx++];
        setLoadingPct(pct);
        setLoadingMsg(msg);
      }
    }, 600);

    try {
      const formData = new FormData();
      formData.append("poster", file);

      const res = await fetch(`${API_BASE}/api/score`, {
        method: "POST",
        body: formData,
      });

      clearInterval(ticker);
      setLoadingPct(100);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        throw new Error(err.error || `Server error ${res.status}`);
      }

      const data = await res.json();
      // data = { score, scores{overall,visual,typography,layout,branding,contrast},
      //          recommendations[], features{} }

      setScores(data.scores);
      setRecs(data.recommendations);
      setFeatures(data.features);

      await new Promise(r => setTimeout(r, 400));   // brief pause for UX
      setLoading(false);
      setStep(2);
      setTimeout(() => setRingsAnim(true), 200);    // trigger ring animation

    } catch (err) {
      clearInterval(ticker);
      setLoading(false);
      setStep(0);
      setError(err.message);
    }
  };

  // ── Step 2 → 3: Call /api/enhance ────────────────────────────────────────
  const runEnhancement = async () => {
    if (!file || !recommendations) return;
    setError(null);
    setLoading(true);
    setLoadingMsg("Applying AI enhancements to your poster…");
    setLoadingPct(0);

    const enhMsgs = [
      [15,  "Analysing layout structure…"],
      [35,  "Adjusting contrast and brightness…"],
      [55,  "Rebalancing color palette…"],
      [75,  "Optimising typography spacing…"],
      [90,  "Rendering enhanced poster…"],
    ];
    let idx = 0;
    const ticker = setInterval(() => {
      if (idx < enhMsgs.length) {
        const [pct, msg] = enhMsgs[idx++];
        setLoadingPct(pct);
        setLoadingMsg(msg);
      }
    }, 700);

    try {
      const formData = new FormData();
      formData.append("poster", file);
      formData.append("recommendations", JSON.stringify(recommendations));

      const res = await fetch(`${API_BASE}/api/enhance`, {
        method: "POST",
        body: formData,
      });

      clearInterval(ticker);
      setLoadingPct(100);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        throw new Error(err.error || `Server error ${res.status}`);
      }

      const data = await res.json();
      // data = { enhanced_image_base64, image_format, changes_applied[] }

      const mime = data.image_format === "png" ? "image/png" : "image/jpeg";
      setEnhanced(`data:${mime};base64,${data.enhanced_image_base64}`);
      setChanges(data.changes_applied || []);

      await new Promise(r => setTimeout(r, 300));
      setLoading(false);
      setStep(3);

    } catch (err) {
      clearInterval(ticker);
      setLoading(false);
      setError(err.message);
    }
  };

  const reset = () => {
    setStep(0); setFile(null); setPreview(null);
    setScores(null); setRecs(null); setFeatures(null);
    setEnhanced(null); setChanges([]); setError(null); setRingsAnim(false);
  };

  const priorityGroups = recommendations
    ? {
        critical: recommendations.filter(r => r.priority === "critical"),
        improve:  recommendations.filter(r => r.priority === "improve"),
        good:     recommendations.filter(r => r.priority === "good"),
      }
    : {};

  // ── Feature table for the detail panel ─────────────────────────────────
  const featureLabels = {
    brightness:   "Brightness",
    contrast:     "Contrast",
    entropy:      "Entropy",
    edge_density: "Edge Density",
    colorfulness: "Colorfulness",
    lr_balance:   "L/R Balance",
    tb_balance:   "T/B Balance",
    whitespace:   "Whitespace",
    saliency:     "Saliency",
    text_density: "Text Density",
    word_count:   "Word Count",
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#0b0b18",
      fontFamily: "'DM Sans', sans-serif",
      backgroundImage: "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(91,50,255,0.18) 0%, transparent 70%)",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0b0b18; }
        ::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
        @keyframes spin  { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
        @keyframes fadeUp{ from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
        .rec-card:hover  { border-color:rgba(91,50,255,0.4)!important; transform:translateY(-2px); background:rgba(255,255,255,0.04)!important; }
        .action-btn:hover{ transform:translateY(-2px); box-shadow:0 12px 40px rgba(91,50,255,0.4)!important; }
        .upload-zone:hover{ border-color:rgba(91,50,255,0.6)!important; background:rgba(91,50,255,0.06)!important; }
        .feat-row:nth-child(odd){ background:rgba(255,255,255,0.015); }
      `}</style>

      {/* ── Header ── */}
      <header style={{
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        padding: "0 40px", height: 64,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        backdropFilter: "blur(10px)", position: "sticky", top: 0, zIndex: 100,
        background: "rgba(11,11,24,0.8)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 8,
            background: "linear-gradient(135deg, #5b32ff, #a855f7)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <span style={{ color: "#fff", fontSize: 16 }}>✦</span>
          </div>
          <span style={{ color: "#fff", fontFamily: "'Bebas Neue', sans-serif", fontSize: 22, letterSpacing: "0.1em" }}>POSTERIQ</span>
          <span style={{
            background: "rgba(91,50,255,0.2)", border: "1px solid rgba(91,50,255,0.4)",
            color: "#9b7aff", borderRadius: 20, padding: "2px 10px", fontSize: 11, fontWeight: 600, letterSpacing: "0.06em",
          }}>BETA</span>
        </div>
        <nav style={{ display: "flex", gap: 8 }}>
          {steps.map((s, i) => (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "5px 14px", borderRadius: 20,
              background: i === step ? "rgba(91,50,255,0.2)" : "transparent",
              border: `1px solid ${i === step ? "rgba(91,50,255,0.5)" : "transparent"}`,
              transition: "all 0.3s",
            }}>
              <span style={{
                width: 18, height: 18, borderRadius: "50%", fontSize: 10, fontWeight: 700,
                display: "flex", alignItems: "center", justifyContent: "center",
                background: i < step ? "#5b32ff" : i === step ? "rgba(91,50,255,0.5)" : "rgba(255,255,255,0.1)",
                color: i <= step ? "#fff" : "#666",
              }}>{i < step ? "✓" : i + 1}</span>
              <span style={{ fontSize: 12, fontWeight: 500, color: i === step ? "#c4adff" : i < step ? "#9b7aff" : "#555" }}>{s}</span>
            </div>
          ))}
        </nav>
      </header>

      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "48px 24px" }}>

        {/* ── STEP 0: Upload ── */}
        {step === 0 && (
          <div style={{ animation: "fadeUp 0.5s ease" }}>
            <div style={{ textAlign: "center", marginBottom: 48 }}>
              <h1 style={{
                fontFamily: "'Bebas Neue', sans-serif", fontSize: 64,
                letterSpacing: "0.05em", lineHeight: 1.1,
                background: "linear-gradient(135deg, #fff 40%, #9b7aff)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>ANALYZE YOUR<br />UNIVERSITY POSTER</h1>
              <p style={{ color: "#666", marginTop: 16, fontSize: 16, maxWidth: 480, margin: "16px auto 0" }}>
                Upload your promotional poster — our TOPSIS AI model scores it across 11 design dimensions, surfaces issues, and generates an enhanced version.
              </p>
            </div>

            {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

            <div
              className="upload-zone"
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => !preview && fileRef.current.click()}
              style={{
                border: `2px dashed ${dragging ? "rgba(91,50,255,0.8)" : preview ? "rgba(91,50,255,0.5)" : "rgba(255,255,255,0.12)"}`,
                borderRadius: 20, padding: preview ? 0 : "80px 40px",
                textAlign: "center", cursor: preview ? "default" : "pointer",
                background: dragging ? "rgba(91,50,255,0.08)" : "rgba(255,255,255,0.02)",
                transition: "all 0.3s", overflow: "hidden",
                minHeight: preview ? "auto" : 320,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              {preview ? (
                <div style={{ position: "relative", width: "100%" }}>
                  <img src={preview} alt="Poster preview" style={{ width: "100%", maxHeight: 480, objectFit: "contain", borderRadius: 18, display: "block" }} />
                  <div style={{
                    position: "absolute", inset: 0,
                    background: "linear-gradient(to top, rgba(11,11,24,0.92) 0%, transparent 45%)",
                    borderRadius: 18, display: "flex", flexDirection: "column",
                    justifyContent: "flex-end", padding: 24,
                  }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <div>
                        <p style={{ color: "#fff", fontWeight: 600, fontSize: 15 }}>{file?.name}</p>
                        <p style={{ color: "#555", fontSize: 12, marginTop: 2 }}>
                          <span onClick={() => fileRef.current.click()} style={{ cursor: "pointer", color: "#7a5cff", textDecoration: "underline" }}>Replace file</span>
                          {" "}· {(file?.size / 1024).toFixed(0)} KB
                        </p>
                      </div>
                      <button onClick={runScoring} className="action-btn" style={{
                        background: "linear-gradient(135deg, #5b32ff, #a855f7)",
                        border: "none", borderRadius: 12, padding: "14px 32px",
                        color: "#fff", fontSize: 14, fontWeight: 700, cursor: "pointer",
                        letterSpacing: "0.04em", boxShadow: "0 8px 32px rgba(91,50,255,0.3)",
                        transition: "all 0.3s",
                      }}>ANALYZE POSTER →</button>
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{
                    width: 72, height: 72, borderRadius: 20,
                    background: "rgba(91,50,255,0.15)", border: "1px solid rgba(91,50,255,0.3)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    margin: "0 auto 24px", fontSize: 28,
                  }}>📄</div>
                  <p style={{ color: "#fff", fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Drop your poster here</p>
                  <p style={{ color: "#555", fontSize: 14 }}>or click to browse — PNG, JPG, WEBP supported</p>
                </div>
              )}
            </div>
            <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
          </div>
        )}

        {/* ── STEP 1: Loading ── */}
        {step === 1 && loading && (
          <div style={{ animation: "fadeUp 0.5s ease", textAlign: "center", padding: "80px 0" }}>
            <div style={{
              width: 80, height: 80, border: "3px solid rgba(91,50,255,0.2)",
              borderTop: "3px solid #5b32ff", borderRadius: "50%",
              margin: "0 auto 32px", animation: "spin 1s linear infinite",
            }} />
            <h2 style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 36, color: "#fff", letterSpacing: "0.08em" }}>
              {loadingMsg || "Initialising model…"}
            </h2>
            <p style={{ color: "#555", marginTop: 12, fontSize: 14 }}>TOPSIS model is examining every dimension of your design</p>

            {/* Progress bar */}
            <div style={{ width: 320, margin: "28px auto 0", background: "rgba(255,255,255,0.06)", borderRadius: 8, height: 4 }}>
              <div style={{
                width: `${loadingPct}%`, height: "100%",
                background: "linear-gradient(90deg, #5b32ff, #a855f7)",
                borderRadius: 8, transition: "width 0.5s ease",
              }} />
            </div>
            <p style={{ color: "#444", marginTop: 10, fontSize: 12, fontFamily: "'Space Mono', monospace" }}>{loadingPct}%</p>

            <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 32, flexWrap: "wrap" }}>
              {["Brightness", "Contrast", "Entropy", "Edge Density", "Colorfulness", "TOPSIS"].map((l, i) => (
                <span key={i} style={{
                  background: "rgba(91,50,255,0.1)", border: "1px solid rgba(91,50,255,0.2)",
                  color: "#7a5cff", borderRadius: 20, padding: "5px 14px", fontSize: 12,
                  animation: `pulse 1.5s ease ${i * 0.2}s infinite`,
                }}>{l}</span>
              ))}
            </div>
          </div>
        )}

        {/* ── STEP 2: Recommendations ── */}
        {step === 2 && scores && recommendations && (
          <div style={{ animation: "fadeUp 0.5s ease" }}>
            {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>

              {/* Left: poster + score rings */}
              <div>
                <h2 style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 28, color: "#fff", letterSpacing: "0.06em", marginBottom: 20 }}>ORIGINAL POSTER</h2>
                <div style={{ borderRadius: 16, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)" }}>
                  <img src={preview} alt="Original" style={{ width: "100%", display: "block", maxHeight: 300, objectFit: "contain", background: "#111" }} />
                </div>

                {/* Overall score */}
                <div style={{
                  marginTop: 20, padding: 24, borderRadius: 16,
                  background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
                    <div>
                      <p style={{ color: "#666", fontSize: 12, letterSpacing: "0.08em", textTransform: "uppercase" }}>Overall TOPSIS Score</p>
                      <p style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 56, color: "#fff", lineHeight: 1, marginTop: 4 }}>
                        {scores.overall}<span style={{ fontSize: 24, color: "#555" }}>/100</span>
                      </p>
                    </div>
                    <div style={{
                      padding: "10px 20px", borderRadius: 12, fontWeight: 700, fontSize: 14,
                      background: scores.overall >= 75 ? "rgba(0,210,130,0.15)" : scores.overall >= 50 ? "rgba(255,180,0,0.15)" : "rgba(255,80,80,0.15)",
                      border: `1px solid ${scores.overall >= 75 ? "rgba(0,210,130,0.3)" : scores.overall >= 50 ? "rgba(255,180,0,0.3)" : "rgba(255,80,80,0.3)"}`,
                      color: scores.overall >= 75 ? "#00d282" : scores.overall >= 50 ? "#ffb800" : "#ff6060",
                    }}>
                      {scores.overall >= 75 ? "GOOD" : scores.overall >= 50 ? "NEEDS WORK" : "POOR"}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 12, justifyContent: "space-between", flexWrap: "wrap" }}>
                    <ScoreRing score={scores.visual}     label="Visual"   color="#a855f7" animate={ringsAnimated} />
                    <ScoreRing score={scores.typography} label="Type"     color="#3b82f6" animate={ringsAnimated} />
                    <ScoreRing score={scores.layout}     label="Layout"   color="#00d282" animate={ringsAnimated} />
                    <ScoreRing score={scores.branding}   label="Color"    color="#ff6060" animate={ringsAnimated} />
                    <ScoreRing score={scores.contrast}   label="Entropy"  color="#ffb800" animate={ringsAnimated} />
                  </div>
                </div>

                {/* Feature values table */}
                {features && (
                  <div style={{ marginTop: 16, borderRadius: 14, overflow: "hidden", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <div style={{ padding: "10px 16px", background: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                      <span style={{ color: "#555", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase" }}>Raw Feature Values</span>
                    </div>
                    {Object.entries(featureLabels).map(([key, label]) => (
                      <div key={key} className="feat-row" style={{ display: "flex", justifyContent: "space-between", padding: "7px 16px" }}>
                        <span style={{ color: "#666", fontSize: 12 }}>{label}</span>
                        <span style={{ color: "#aaa", fontSize: 12, fontFamily: "'Space Mono', monospace" }}>
                          {typeof features[key] === "number" ? features[key].toFixed(4) : features[key]}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Right: recommendations */}
              <div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
                  <h2 style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 28, color: "#fff", letterSpacing: "0.06em" }}>RECOMMENDATIONS</h2>
                  <div style={{ display: "flex", gap: 8 }}>
                    {priorityGroups.critical?.length > 0 && <Tag text={`${priorityGroups.critical.length} Critical`} type="critical" />}
                    {priorityGroups.improve?.length  > 0 && <Tag text={`${priorityGroups.improve.length} Improve`}  type="improve" />}
                    {priorityGroups.good?.length     > 0 && <Tag text={`${priorityGroups.good.length} Good`}        type="good" />}
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 12, maxHeight: 480, overflowY: "auto", paddingRight: 4 }}>
                  {recommendations.map((rec) => (
                    <div key={rec.id} className="rec-card" style={{
                      padding: "16px 18px", borderRadius: 14,
                      background: "rgba(255,255,255,0.02)",
                      border: `1px solid ${rec.priority === "critical" ? "rgba(255,80,80,0.2)" : rec.priority === "improve" ? "rgba(255,180,0,0.2)" : "rgba(0,210,130,0.2)"}`,
                      transition: "all 0.25s", cursor: "default",
                    }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                        <Tag text={rec.priority === "critical" ? "⚠ Critical" : rec.priority === "improve" ? "↑ Improve" : "✓ Good"} type={rec.priority} />
                        <span style={{ color: "#444", fontSize: 12, fontFamily: "'Space Mono', monospace" }}>{rec.category}</span>
                      </div>
                      <p style={{ color: "#ccc", fontSize: 13, lineHeight: 1.6 }}>{rec.text}</p>
                    </div>
                  ))}
                </div>

                <button onClick={runEnhancement} className="action-btn" style={{
                  width: "100%", marginTop: 24,
                  background: "linear-gradient(135deg, #5b32ff, #a855f7)",
                  border: "none", borderRadius: 14, padding: "18px",
                  color: "#fff", fontSize: 15, fontWeight: 700, cursor: "pointer",
                  letterSpacing: "0.06em", boxShadow: "0 8px 32px rgba(91,50,255,0.3)",
                  transition: "all 0.3s",
                }}>✦ GENERATE ENHANCED POSTER</button>
              </div>
            </div>
          </div>
        )}

        {/* Enhancement loading overlay (stays on step 2 visually) */}
        {loading && step === 2 && (
          <div style={{
            position: "fixed", inset: 0, background: "rgba(11,11,24,0.9)",
            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
            zIndex: 200, backdropFilter: "blur(6px)",
          }}>
            <div style={{
              width: 80, height: 80, border: "3px solid rgba(91,50,255,0.2)",
              borderTop: "3px solid #5b32ff", borderRadius: "50%",
              animation: "spin 1s linear infinite", marginBottom: 28,
            }} />
            <h2 style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 32, color: "#fff", letterSpacing: "0.08em" }}>
              {loadingMsg}
            </h2>
            <div style={{ width: 300, margin: "20px auto 0", background: "rgba(255,255,255,0.06)", borderRadius: 8, height: 4 }}>
              <div style={{
                width: `${loadingPct}%`, height: "100%",
                background: "linear-gradient(90deg, #5b32ff, #a855f7)",
                borderRadius: 8, transition: "width 0.5s ease",
              }} />
            </div>
          </div>
        )}

        {/* ── STEP 3: Enhanced Poster ── */}
        {step === 3 && enhancedPoster && (
          <div style={{ animation: "fadeUp 0.5s ease" }}>
            {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
            <div style={{ textAlign: "center", marginBottom: 40 }}>
              <div style={{
                display: "inline-flex", alignItems: "center", gap: 8, padding: "8px 20px",
                background: "rgba(0,210,130,0.12)", border: "1px solid rgba(0,210,130,0.3)",
                borderRadius: 20, marginBottom: 16,
              }}>
                <span style={{ color: "#00d282", fontSize: 14 }}>✓</span>
                <span style={{ color: "#00d282", fontSize: 13, fontWeight: 600, letterSpacing: "0.06em" }}>Enhancement Complete</span>
              </div>
              <h1 style={{
                fontFamily: "'Bebas Neue', sans-serif", fontSize: 52, color: "#fff", letterSpacing: "0.05em",
                background: "linear-gradient(135deg, #fff 40%, #9b7aff)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>YOUR POSTER HAS BEEN UPGRADED</h1>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              <div>
                <p style={{ color: "#555", fontSize: 12, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>Before</p>
                <div style={{ borderRadius: 16, overflow: "hidden", border: "1px solid rgba(255,80,80,0.2)" }}>
                  <img src={preview} alt="Original" style={{ width: "100%", display: "block", maxHeight: 420, objectFit: "contain", background: "#111" }} />
                </div>
                <div style={{ marginTop: 12, padding: "10px 16px", background: "rgba(255,80,80,0.08)", borderRadius: 10, border: "1px solid rgba(255,80,80,0.2)" }}>
                  <span style={{ color: "#ff6060", fontFamily: "'Bebas Neue', sans-serif", fontSize: 28 }}>{scores?.overall}</span>
                  <span style={{ color: "#555", fontSize: 13 }}>/100 original score</span>
                </div>
              </div>
              <div>
                <p style={{ color: "#9b7aff", fontSize: 12, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>After — AI Enhanced</p>
                <div style={{ borderRadius: 16, overflow: "hidden", border: "1px solid rgba(91,50,255,0.4)", boxShadow: "0 0 40px rgba(91,50,255,0.15)" }}>
                  <img src={enhancedPoster} alt="Enhanced" style={{ width: "100%", display: "block", maxHeight: 420, objectFit: "contain", background: "#111" }} />
                </div>
                <div style={{ marginTop: 12, padding: "10px 16px", background: "rgba(0,210,130,0.08)", borderRadius: 10, border: "1px solid rgba(0,210,130,0.2)" }}>
                  <span style={{ color: "#00d282", fontSize: 13 }}>Enhancement model applied {changesApplied.length} improvements</span>
                </div>
              </div>
            </div>

            {changesApplied.length > 0 && (
              <div style={{ marginTop: 28, padding: 24, borderRadius: 16, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}>
                <h3 style={{ color: "#fff", fontWeight: 600, marginBottom: 16, fontSize: 14, letterSpacing: "0.06em" }}>CHANGES APPLIED</h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {changesApplied.map((c, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "10px 14px", background: "rgba(0,210,130,0.06)", borderRadius: 10, border: "1px solid rgba(0,210,130,0.15)" }}>
                      <span style={{ color: "#00d282", fontSize: 12, marginTop: 1 }}>✓</span>
                      <span style={{ color: "#aaa", fontSize: 12 }}>{c}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: "flex", gap: 12, marginTop: 28 }}>
              <a href={enhancedPoster} download="enhanced-poster.png" style={{ textDecoration: "none" }}>
                <button className="action-btn" style={{
                  background: "linear-gradient(135deg, #5b32ff, #a855f7)",
                  border: "none", borderRadius: 12, padding: "14px 32px",
                  color: "#fff", fontSize: 14, fontWeight: 700, cursor: "pointer",
                  letterSpacing: "0.04em", boxShadow: "0 8px 32px rgba(91,50,255,0.3)",
                  transition: "all 0.3s",
                }}>↓ Download Enhanced Poster</button>
              </a>
              <button onClick={reset} style={{
                background: "transparent", border: "1px solid rgba(255,255,255,0.15)",
                borderRadius: 12, padding: "14px 28px", color: "#888",
                fontSize: 14, cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
              }}>← Analyze Another Poster</button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
