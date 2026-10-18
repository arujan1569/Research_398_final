# OryzaAI React frontend

React 19 + TypeScript + Tailwind CSS 3 + Vite. The visual design follows both supplied OryzaAI reference screens.

```powershell
npm ci
npm run dev
```

Open http://127.0.0.1:5173 with FastAPI running on port 8001. Vite proxies `/api`, `/docs`, and `/openapi.json`. For a separately hosted API, copy `.env.example` to `.env`, set `VITE_API_URL`, and allow the frontend origin in the backend's CORS configuration.

```powershell
npm run build
```

Build performs strict TypeScript checking then writes `dist/`. Start FastAPI after building to serve the UI and API together at http://127.0.0.1:8001. Pages use hash navigation, so reloads work with static serving. `npm run preview` previews the static build only; use FastAPI for the integrated production preview.

Implemented: upload/drop, client and server validation, object-URL preview cleanup, model selection, live readiness, cancellation/timeout, real ranked results, measured timings, original/letterbox preview, enlarge image, model comparison, JSON report, print/PDF layout, searchable disease library, exported evaluation metrics, and API documentation.

PDF export uses the browser print dialog's Save as PDF option. JSON exports contain predictions and comparison results; images are not embedded. Comparison calls the other real model on the same upload. No placeholder sample photo or fabricated prediction is used. Processing stages deliberately show an indeterminate state because the API does not stream stage progress.

Botanical reference: [IRRI Rice Diseases Online Resource](https://rice-diseases.irri.org/contents). This is reference context, not an IRRI endorsement or a claim that the training data were independently validated. Google Fonts are optional network assets; local system font fallbacks are provided. No uploaded image is sent to the font provider.
