"""Run with python -m tools.gcp_annotator.app (loopback only by default)."""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core import Catalog, Conflict, DEFAULT_ROOT, Store

STATIC = Path(__file__).with_name('static')


def create_app(root=DEFAULT_ROOT, output=None, manifest=None, sample_size=20, seed=42):
    catalog = Catalog(root, manifest)
    store = Store(catalog, output, sample_size, seed)
    app = FastAPI(title='MHINet GoogleEarth Ground GCP', docs_url='/api/docs')
    app.state.store = store

    @app.middleware('http')
    async def local_mutations(request: Request, call_next):
        # Browsers on unrelated sites cannot mutate a loopback/forwarded server.
        if request.method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            from urllib.parse import urlsplit
            origin = request.headers.get('origin')
            if origin and urlsplit(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail': 'Cross-origin writes are disabled'}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=409 if isinstance(exc, Conflict) else 422)

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse({'detail': 'Unknown pair or missing required field: ' + str(exc)}, status_code=404)

    @app.get('/api/catalog')
    def overview():
        return store.overview()

    @app.get('/api/pairs/{pair_id}')
    def load_pair(pair_id: str):
        return dict(annotation=store.load(pair_id), metadata=store.metadata(pair_id))

    @app.put('/api/pairs/{pair_id}')
    def save_pair(pair_id: str, body: dict = Body(...)):
        return store.save(pair_id, body)

    @app.get('/api/pairs/{pair_id}/image/{side}/{mode}')
    def image(pair_id: str, side: str, mode: str):
        path = catalog.image_path(catalog.row(pair_id), side, mode)
        return FileResponse(path, headers={'Cache-Control': 'private, max-age=3600'})

    @app.post('/api/export')
    def export():
        return store.export()

    @app.get('/api/exports/{format}')
    def download(format: str):
        if format not in ('csv', 'jsonl'):
            raise HTTPException(404)
        path = store.output / 'exports' / ('ground_gcps.' + format)
        if not path.exists():
            raise HTTPException(404, 'Export first')
        return FileResponse(path, filename=path.name, media_type='text/csv' if format == 'csv' else 'application/x-ndjson')

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html', headers={'Cache-Control': 'no-cache'})

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--output', type=Path, help='Default: DATA_ROOT/ground_gcp_annotations')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--sample-size', type=int, default=20, help='Used only on first run')
    p.add_argument('--seed', type=int, default=42, help='Used only on first run')
    p.add_argument('--extend-sample', type=int, help='Extend frozen selection, keep original prefix, then exit')
    args = p.parse_args()
    app = create_app(args.data_root, args.output, args.manifest, args.sample_size, args.seed)
    if args.extend_sample:
        plan = app.state.store.extend(args.extend_sample)
        print(f'Frozen sample now contains {len(plan["pair_ids"])} pairs')
        return
    import uvicorn
    print('Annotations:', app.state.store.output, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()
