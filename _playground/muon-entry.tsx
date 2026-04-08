import React from "react";
import ReactDOM from "react-dom/client";
import App from "./muon-advantage-demo-2";

const el = document.getElementById("playground-muon-root");
if (el) {
  ReactDOM.createRoot(el).render(React.createElement(App));
}
