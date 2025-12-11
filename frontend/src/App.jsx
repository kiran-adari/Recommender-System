import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

// Small re-usable movie card component
function MovieCard({ rec, index }) {
  return (
    <div className="movie-card">
      <div className="poster-wrapper">
        {rec.poster_url ? (
          <img src={rec.poster_url} alt={rec.title} className="poster-img" />
        ) : (
          <div className="poster-placeholder">
            <span>No poster</span>
          </div>
        )}
        <span className="card-rank">#{index + 1}</span>
      </div>

      <div className="card-body">
        <h3 className="card-title">{rec.title || "Unknown title"}</h3>

        <div className="card-meta">
          <span className="card-pill">ID: {rec.item_id}</span>
          <span className="card-pill score-pill">
            ⭐ {rec.score.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [scenario, setScenario] = useState("baseline");
  const [userId, setUserId] = useState(1);

  // mode: "single" = old behavior, "compare" = new 3-column view
  const [mode, setMode] = useState("single");

  const [recs, setRecs] = useState([]);
  const [compareData, setCompareData] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleFetch(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setRecs([]);
    setCompareData(null);

    try {
      if (mode === "single") {
        // Call /api/recommend
        const res = await axios.post(`${API_BASE}/api/recommend`, {
          scenario,
          user_id: Number(userId),
          top_k: 12,
        });
        setRecs(res.data.recommendations);
      } else {
        // mode === "compare" -> call /api/compare
        const res = await axios.post(`${API_BASE}/api/compare`, {
          user_id: Number(userId),
          top_k: 12,
        });
        setCompareData(res.data);
      }
    } catch (err) {
      console.error(err);
      setError("Could not get recommendations.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-root">
      {/* HEADER */}
      <header className="app-header">
        <h1 className="app-title">Movie Recommender</h1>
        <p className="app-subtitle">
          Python + FastAPI · React + Axios <br />
          Baseline · Attack · Defense
        </p>
      </header>

      {/* CONTROL PANEL */}
      <section className="control-panel">
        <form onSubmit={handleFetch} className="control-form">
          {/* first row: user + scenario */}
          <div className="form-row">
            <label className="form-label">
              User ID
              <input
                type="number"
                className="form-input"
                value={userId}
                min={1}
                onChange={(e) => setUserId(e.target.value)}
              />
            </label>

            <label className="form-label">
              Scenario (for single mode)
              <select
                className="form-select"
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                disabled={mode === "compare"}
              >
                <option value="baseline">Baseline</option>
                <option value="attack">Attack</option>
                <option value="defense">Defense</option>
              </select>
            </label>
          </div>

          {/* second row: mode toggle */}
          <div className="mode-toggle-row">
            <span className="mode-label">View mode:</span>
            <div className="mode-toggle">
              <button
                type="button"
                className={
                  "mode-pill" + (mode === "single" ? " mode-pill-active" : "")
                }
                onClick={() => setMode("single")}
              >
                Single scenario
              </button>
              <button
                type="button"
                className={
                  "mode-pill" + (mode === "compare" ? " mode-pill-active" : "")
                }
                onClick={() => setMode("compare")}
              >
                Compare all (baseline · attack · defense)
              </button>
            </div>
          </div>

          <button className="primary-btn" type="submit" disabled={loading}>
            {loading ? "Loading..." : "Get Recommendations"}
          </button>
        </form>

        {error && <p className="error-text">{error}</p>}
      </section>

      {/* SINGLE MODE RESULTS */}
      {mode === "single" && recs.length > 0 && (
        <section className="results-section">
          <div className="results-header">
            <h2 className="results-title">Recommendations</h2>
            <span className="results-meta">
              User {userId} · Scenario:{" "}
              <strong>
                {scenario.charAt(0).toUpperCase() + scenario.slice(1)}
              </strong>
            </span>
          </div>

          <div className="card-grid">
            {recs.map((r, idx) => (
              <MovieCard key={r.item_id} rec={r} index={idx} />
            ))}
          </div>
        </section>
      )}

      {/* COMPARE MODE RESULTS */}
      {mode === "compare" && compareData && (
        <section className="results-section">
          <div className="results-header">
            <h2 className="results-title">Baseline vs Attack vs Defense</h2>
            <span className="results-meta">
              User {userId} · Top {compareData.top_k}
            </span>
          </div>

          <div className="compare-grid">
            <div className="compare-column">
              <div className="compare-column-header baseline-header">
                <h3>Baseline</h3>
                <p>Original model (no attack)</p>
              </div>
              <div className="card-grid vertical-grid">
                {compareData.baseline.map((r, idx) => (
                  <MovieCard key={"b-" + r.item_id} rec={r} index={idx} />
                ))}
              </div>
            </div>

            <div className="compare-column">
              <div className="compare-column-header attack-header">
                <h3>Attack</h3>
                <p>With injected fake user</p>
              </div>
              <div className="card-grid vertical-grid">
                {compareData.attack.map((r, idx) => (
                  <MovieCard key={"a-" + r.item_id} rec={r} index={idx} />
                ))}
              </div>
            </div>

            <div className="compare-column">
              <div className="compare-column-header defense-header">
                <h3>Defense</h3>
                <p>After clipping suspicious ratings</p>
              </div>
              <div className="card-grid vertical-grid">
                {compareData.defense.map((r, idx) => (
                  <MovieCard key={"d-" + r.item_id} rec={r} index={idx} />
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* HINT WHEN NOTHING YET */}
      {recs.length === 0 &&
        !compareData &&
        !loading &&
        !error && (
          <p className="hint-text">
            Choose a user, select a mode, then click{" "}
            <strong>Get Recommendations</strong> to see movies.
          </p>
        )}
    </div>
  );
}

export default App;
