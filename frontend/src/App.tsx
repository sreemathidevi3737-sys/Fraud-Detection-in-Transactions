import React, { useState, useEffect } from 'react';

const API_BASE = 'http://127.0.0.1:5000/api';

interface SampleItem {
  id: number;
  Amount: number;
  label: string;
  [key: string]: any;
}

interface PredictResult {
  id: number;
  amount: number;
  is_fraud: boolean;
  fraud_probability: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  flags: string[];
}

export default function App() {
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [amount, setAmount] = useState<number>(145.5);
  const [features, setFeatures] = useState<{ [key: string]: number }>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);

  useEffect(() => {
    fetchSamples();
  }, []);

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/samples`);
      if (res.ok) {
        const data = await res.json();
        if (data?.samples) setSamples(data.samples);
      }
    } catch {
      console.warn("Backend on port 5000 not reachable.");
    }
  };

  const loadPreset = (sample: SampleItem) => {
    setAmount(sample.Amount || 0);
    const feats: { [key: string]: number } = {};
    for (let i = 1; i <= 28; i++) {
      feats[`V${i}`] = sample[`V${i}`] ?? 0;
    }
    setFeatures(feats);
    setResult(null);
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: 1,
          Amount: amount,
          ...features
        })
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
      } else {
        alert("Prediction server returned an error.");
      }
    } catch {
      alert("Error contacting Flask backend. Ensure 'python3 app.py' is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1 style={{ margin: 0, fontSize: '26px' }}>
            Fraud Sentinel <span className="badge">v1.0 ML</span>
          </h1>
          <p style={{ margin: '4px 0 0 0', color: '#94a3b8', fontSize: '13px' }}>
            Real-time Credit Card Transaction Anomaly Detection
          </p>
        </div>
        <button className="btn-preset" onClick={fetchSamples}>
          ↻ Refresh Presets
        </button>
      </header>

      {samples.length > 0 && (
        <div className="presets-card">
          <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Quick-Test Benchmark Presets
          </span>
          <div className="preset-buttons">
            {samples.map((s, idx) => (
              <button
                key={idx}
                className={`btn-preset ${s.label.includes('Fraud') ? 'fraud' : 'normal'}`}
                onClick={() => loadPreset(s)}
              >
                {s.label} (${s.Amount.toFixed(2)})
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid-layout">
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Transaction Inspector</h3>
          <form onSubmit={handlePredict}>
            <div className="form-group">
              <label>TRANSACTION AMOUNT ($)</label>
              <input
                type="number"
                step="any"
                value={amount}
                onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
                required
              />
            </div>

            <div className="form-group">
              <label>KEY PCA ANOMALY VECTORS</label>
              <div className="features-grid">
                {['V4', 'V11', 'V12', 'V14', 'V17'].map((v) => (
                  <div key={v}>
                    <label style={{ fontSize: '10px' }}>{v}</label>
                    <input
                      type="number"
                      step="any"
                      value={features[v] ?? 0}
                      onChange={(e) =>
                        setFeatures({ ...features, [v]: parseFloat(e.target.value) || 0 })
                      }
                    />
                  </div>
                ))}
              </div>
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? 'Evaluating...' : 'Run Real-Time Prediction'}
            </button>
          </form>
        </div>

        <div className="card">
          <h3 style={{ marginTop: 0 }}>Evaluation Verdict</h3>
          {result ? (
            <div>
              <div className={`verdict-box ${result.is_fraud ? 'fraud' : 'normal'}`}>
                <h4 style={{ margin: 0, fontSize: '17px' }}>
                  {result.is_fraud ? '🚨 High-Risk Fraud Flagged' : '✅ Legitimate Transaction'}
                </h4>
                <p style={{ margin: '6px 0 0 0', fontSize: '12px' }}>
                  Evaluated Amount: ${result.amount.toFixed(2)}
                </p>
              </div>

              <div style={{ marginTop: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                  <span>Fraud Risk Probability</span>
                  <strong>{(result.fraud_probability * 100).toFixed(1)}%</strong>
                </div>

                <div className="progress-bar">
                  <div
                    className={`progress-fill ${result.risk_level === 'HIGH' ? 'high' : ''}`}
                    style={{ width: `${Math.max(result.fraud_probability * 100, 5)}%` }}
                  />
                </div>
              </div>

              {result.flags?.length > 0 && (
                <div style={{ marginTop: '16px' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase' }}>
                    Detected Indicators:
                  </span>
                  {result.flags.map((flag, idx) => (
                    <div key={idx} className="flag-item">
                      ⚠️ {flag}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '40px 10px', color: '#64748b' }}>
              <p style={{ fontSize: '13px' }}>
                Select a preset at the top or enter values and click <b>Run Real-Time Prediction</b>.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}