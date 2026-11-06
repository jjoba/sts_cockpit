'''
STS Cockpit UI server. Serves the web app in ui/web and a small JSON API on top of the real models. Run from the project root:

    python ui/server/app.py            (then open http://127.0.0.1:8765)
    python ui/server/app.py port=9000

Only the standard library plus the packages the cockpit already needs.

    GET  /api/catalog   cards, relics, starting states and the model registry (with held-out metrics and the default model)
    POST /api/predict   {model, character, states: [{floor, cards: {column: count}, relics: [column], hp, max_hp, gold}]} -> {heads, probs: [[...]]}
                        hp, max_hp and gold are used by the models that take run state (v24 and later), the others ignore them
'''

import json
import mimetypes
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(HERE), 'web')
sys.path.insert(0, HERE)
import engine

def load(name):
    path = f'{HERE}/data/{name}.json'
    if not os.path.exists(path):
        raise SystemExit(f'{path} is missing. Run `python ui/server/build_catalog.py` (and build_metrics.py) from the project root first.')
    return json.load(open(path))

CARDS, RELICS = load('cards'), load('relics')
METRICS_FILE = json.load(open(f'{HERE}/data/metrics.json')) if os.path.exists(f'{HERE}/data/metrics.json') else {'test_set': None, 'models': {}}
METRICS = METRICS_FILE.get('models', METRICS_FILE if 'test_set' not in METRICS_FILE else {}) # an older metrics.json is a plain {model: metrics}
START_RUN = json.load(open(f'{HERE}/data/start_run_state.json')) if os.path.exists(f'{HERE}/data/start_run_state.json') else {}

# The recipes that are results in their own right (the tables in outstanding_experiments.md). Every other model is an experiment variant (batch size, learning rate, flip
# sampling, ...) and the model directory hides those until asked.
FEATURED = {
    'fnn_v25_alpha_beta_mt_hp_gold_flip_synthetic', 'fnn_v25_ft_mt_hp_gold_flip_synthetic', 'fnn_v25_ft_mt_flip_synthetic',
    'fnn_v24_alpha_beta_mt_hp_gold', 'fnn_v24_ft_mt_hp_gold', 'fnn_v24_ft_mt_hp', 'fnn_v24_ft_mt_gold', 'fnn_v24_ft_mt_hp_gold_noflip',
    'fnn_v22_alpha_beta_mt_cum', 'fnn_v22_beta_mt_cum', 'fnn_v22_beta_act3_mt_cum', 'fnn_v22_ft_mt_cum', 'fnn_v22_ft_st_cum', 'fnn_v22_ft_mt_cum_flip1',
    'fnn_v21_alpha_beta_mt', 'fnn_v21_beta_mt', 'fnn_v21_beta_act3_mt', 'fnn_v21_ft_mt', 'fnn_v21_ft_st',
}
ENGINE = engine.Engine()

def starting_states():
    '''Per character: the starting deck, relic and floor, straight from starting_decks.csv (the rows the models were built around).'''
    df = pd.read_csv(f'{engine.ROOT}/starting_decks.csv').fillna(0)
    relic_cols = {r['column'] for r in RELICS}
    out = {}
    for ch in engine.CHARACTERS:
        row = df[df[ch] == 1].iloc[0]
        cols = [c for c in df.columns if row[c] > 0 and c != 'floor' and c not in engine.CHARACTERS]
        out[ch] = {'cards': {c: int(row[c]) for c in cols if c not in relic_cols}, 'relics': [c for c in cols if c in relic_cols], 'floor': 1}
    return out

def default_model(models):
    '''Highest version number, then best AUC, among runnable models with held-out metrics (so v22 first, then its best variant).'''
    ok = [m for m in models if m['supported'] and m.get('metrics')]
    return max(ok, key = lambda m: (m['version'], m['metrics']['auc']))['id'] if ok else None

def catalog():
    models = []
    for name, info in sorted(ENGINE.registry.items(), key = lambda kv: (-kv[1]['version'], kv[0])):
        models.append({**info, 'metrics': METRICS.get(name), 'featured': name in FEATURED or info['version'] < 21})
    return {'cards': CARDS, 'relics': RELICS, 'start': starting_states(), 'characters': engine.CHARACTERS, 'models': models,
            'defaultModel': default_model(models), 'device': engine.DEVICE, 'heads': engine.HEADS, 'startRun': START_RUN, 'testSet': METRICS_FILE.get('test_set')}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/api/catalog':
            return self.send_json(catalog())
        rel = 'index.html' if path in ('/', '') else path.lstrip('/')
        full = os.path.realpath(os.path.join(WEB, rel))
        if not full.startswith(os.path.realpath(WEB)) or not os.path.isfile(full):
            return self.send_json({'error': 'not found'}, 404)
        data = open(full, 'rb').read()
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(full)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != '/api/predict':
            return self.send_json({'error': 'not found'}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            probs = ENGINE.predict(req['model'], req['character'], req['states'])
            info = ENGINE.registry[req['model']]
            self.send_json({'heads': engine.HEADS if probs.shape[1] == 5 else ['heart'], 'probs': probs.round(6).tolist(), 'valid': info['heads']})
        except (ValueError, KeyError) as e:
            self.send_json({'error': str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self.send_json({'error': f'{type(e).__name__}: {e}'}, 500)

def main():
    port = 8765
    for a in sys.argv[1:]:
        if a.startswith('port='):
            port = int(a[5:])
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'STS Cockpit UI on http://127.0.0.1:{port}  (models run on {engine.DEVICE})')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
