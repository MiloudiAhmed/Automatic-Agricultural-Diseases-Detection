"use client";

import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ImageIcon,
  Leaf,
  Loader2,
  RefreshCcw,
  ScanSearch,
  Sparkles,
  UploadCloud
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";

type ApiStatus = "checking" | "online" | "offline";
type ViewMode = "preview" | "overlay" | "heatmap";

type TopPrediction = {
  class_name: string;
  label: string;
  confidence: number;
};

type PredictionResult = {
  filename: string;
  model: string;
  class_name: string;
  label: string;
  confidence: number;
  is_healthy: boolean;
  recommendation: string;
  preview: string;
  saliency: {
    heatmap: string;
    overlay: string;
  };
  top_predictions: TopPrediction[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

function formatPercent(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

export default function Home() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [file, setFile] = useState<File | null>(null);
  const [localPreview, setLocalPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("preview");
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_URL}/api/health`, { signal: controller.signal })
      .then((response) => setApiStatus(response.ok ? "online" : "offline"))
      .catch(() => setApiStatus("offline"));

    return () => controller.abort();
  }, []);

  useEffect(() => {
    return () => {
      if (localPreview?.startsWith("blob:")) {
        URL.revokeObjectURL(localPreview);
      }
    };
  }, [localPreview]);

  const selectedImage = useMemo(() => {
    if (!result) {
      return localPreview;
    }

    if (viewMode === "overlay") {
      return result.saliency.overlay;
    }

    if (viewMode === "heatmap") {
      return result.saliency.heatmap;
    }

    return result.preview;
  }, [localPreview, result, viewMode]);

  function selectFile(nextFile: File | null) {
    setError(null);
    setResult(null);
    setViewMode("preview");
    setFile(nextFile);

    if (!nextFile) {
      setLocalPreview(null);
      return;
    }

    if (!nextFile.type.startsWith("image/")) {
      setError("Selectionnez une image valide.");
      setLocalPreview(null);
      setFile(null);
      return;
    }

    setLocalPreview(URL.createObjectURL(nextFile));
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function analyze() {
    if (!file) {
      setError("Ajoutez une image de feuille.");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Prediction impossible.");
      }

      setResult(payload);
      setViewMode("overlay");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Prediction impossible.");
    } finally {
      setIsLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setLocalPreview(null);
    setResult(null);
    setViewMode("preview");
    setError(null);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Leaf size={22} aria-hidden="true" />
          </div>
          <div>
            <p className="eyebrow">Plant Disease AI</p>
            <h1>Diagnostic visuel des feuilles</h1>
          </div>
        </div>

        <div className={`api-pill ${apiStatus}`}>
          <Activity size={16} aria-hidden="true" />
          <span>{apiStatus === "online" ? "API active" : apiStatus === "offline" ? "API hors ligne" : "Connexion"}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Image</p>
              <h2>Feuille a analyser</h2>
            </div>
            <button className="icon-button" type="button" onClick={reset} aria-label="Reinitialiser">
              <RefreshCcw size={18} aria-hidden="true" />
            </button>
          </div>

          <label
            className={`dropzone ${isDragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
          >
            <input type="file" accept="image/*" onChange={onFileChange} />
            <UploadCloud size={34} aria-hidden="true" />
            <strong>{file ? file.name : "Deposer une feuille"}</strong>
            <span>JPG, PNG, WEBP</span>
          </label>

          <button className="primary-button" type="button" onClick={analyze} disabled={!file || isLoading}>
            {isLoading ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <ScanSearch size={18} aria-hidden="true" />}
            <span>{isLoading ? "Analyse en cours" : "Analyser"}</span>
          </button>

          {error && (
            <div className="alert">
              <AlertTriangle size={17} aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <div className="model-strip">
            <BrainCircuit size={18} aria-hidden="true" />
            <div>
              <span>Modele</span>
              <strong>{result?.model ?? "EfficientNet"}</strong>
            </div>
          </div>
        </aside>

        <section className="panel viewer-panel">
          <div className="viewer-toolbar">
            <div className="panel-heading compact">
              <div>
                <p className="eyebrow">Visualisation</p>
                <h2>{result ? result.label : "Apercu"}</h2>
              </div>
            </div>

            <div className="segmented" aria-label="Mode visualisation">
              <button type="button" className={viewMode === "preview" ? "active" : ""} onClick={() => setViewMode("preview")}>
                Photo
              </button>
              <button type="button" className={viewMode === "overlay" ? "active" : ""} onClick={() => setViewMode("overlay")} disabled={!result}>
                Saliency
              </button>
              <button type="button" className={viewMode === "heatmap" ? "active" : ""} onClick={() => setViewMode("heatmap")} disabled={!result}>
                Heatmap
              </button>
            </div>
          </div>

          <div className="image-stage">
            {selectedImage ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={selectedImage} alt="Feuille analysee" />
            ) : (
              <div className="empty-state">
                <ImageIcon size={42} aria-hidden="true" />
                <span>Aucune image</span>
              </div>
            )}
          </div>
        </section>

        <aside className="panel result-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Prediction</p>
              <h2>Resultat</h2>
            </div>
            {result?.is_healthy ? (
              <CheckCircle2 className="healthy" size={24} aria-hidden="true" />
            ) : (
              <Sparkles className="attention" size={24} aria-hidden="true" />
            )}
          </div>

          {result ? (
            <>
              <div className={`diagnosis ${result.is_healthy ? "ok" : "risk"}`}>
                <span>{result.is_healthy ? "Saine" : "Maladie probable"}</span>
                <strong>{result.label}</strong>
                <div className="confidence-row">
                  <span>Confiance</span>
                  <strong>{formatPercent(result.confidence)}</strong>
                </div>
                <div className="confidence-meter">
                  <span style={{ width: `${Math.max(3, result.confidence * 100)}%` }} />
                </div>
              </div>

              <div className="recommendation">{result.recommendation}</div>

              <div className="scores">
                {result.top_predictions.map((prediction) => (
                  <div className="score-row" key={prediction.class_name}>
                    <div className="score-label">
                      <span>{prediction.label}</span>
                      <strong>{formatPercent(prediction.confidence)}</strong>
                    </div>
                    <div className="score-track">
                      <span style={{ width: `${Math.max(2, prediction.confidence * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="result-empty">
              <BrainCircuit size={34} aria-hidden="true" />
              <span>En attente</span>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}
