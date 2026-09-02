import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/landing.css";
import "./styles/widget.css";

const container = document.getElementById("botox-widget-root");
if (!container) {
  throw new Error("botox-widget-root mount node not found");
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
