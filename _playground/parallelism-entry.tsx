import React from "react";
import ReactDOM from "react-dom/client";
import App from "../parallelism-explorer";

const el = document.getElementById("parallelism-root");
if (el) {
  ReactDOM.createRoot(el).render(React.createElement(App));
}
