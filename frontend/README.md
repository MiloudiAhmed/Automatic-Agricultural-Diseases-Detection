# Next.js Frontend

Next.js interface for uploading a plant leaf image to the FastAPI backend and displaying the prediction result with saliency map visualization.

## Run

```powershell
cd frontend
npm install
npm run dev
```

In local development on port `3000`, the frontend calls this API URL by default:

```text
http://127.0.0.1:8000
```

To override the API URL:

```powershell
$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev
```

For deployment, leave `NEXT_PUBLIC_API_URL` empty. The exported frontend will call `/api/*` on the same public domain.
