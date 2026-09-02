# 13F Holdings Tracker

Pipeline de datos que se conecta directamente a **SEC EDGAR**, descarga los formularios **13F-HR** de un universo de managers institucionales, los parsea, clasifica cada posición (tipo de activo, sector, subyacente) y construye un **tracker de cambios en tenencia y exposición** por tipo de activo, por fondo y por tipo de manager, con un reporte analítico y un dashboard interactivo.

```
SEC EDGAR ──► ingest ──► parse ──► classify ──► track ──► analyze ──► dashboard.html + report.md + CSV
 (submissions API,        XML        reglas +      diff QoQ,    insights
  Archives XML)           13F        Naive Bayes   flujo/precio
```

## Inicio rápido

```bash
pip install -r requirements.txt

# 1) Muestra offline (no requiere red): filings sintéticos con el esquema exacto de EDGAR
python -m sec13f.cli all --source sample --clean

# 2) Datos reales de la SEC (requiere internet). La SEC exige identificarse en el User-Agent:
export SEC_USER_AGENT="Tu Nombre tu@email.com"
python -m sec13f.cli verify                      # comprueba los CIK del universo contra EDGAR
python -m sec13f.cli all --source sec --quarters 4 --clean

# Re-generar solo los productos (sin volver a descargar)
python -m sec13f.cli build
```

Salidas en `output/`:

| Archivo | Contenido |
|---|---|
| `dashboard.html` | Dashboard interactivo (selector de trimestre, exposición por activo/sector, rotación por tipo de manager, movimientos, consenso, puts/calls, detalle por manager) |
| `report.md` | Reporte trimestral en Markdown con la lectura del trimestre, tablas y metodología |
| `insights.json` | Insights estructurados por trimestre (para alimentar otros sistemas) |
| `holdings.csv` | Posiciones clasificadas (una fila por manager × trimestre × CUSIP × put/call) con peso y precio implícito |
| `changes.csv` | Diff trimestre a trimestre por posición: acción (NEW/EXIT/ADD/TRIM/HOLD), Δ títulos, efecto flujo y efecto precio |
| `manager_summary.csv` | Por manager y trimestre: valor, posiciones, concentración (top-10, HHI), share de opciones/ETF/crédito, rotación, tipo inferido |
| `exposure_*.csv`, `sector_rotation.csv` | Exposición por tipo de activo y sector (equal-weight y por valor), y flujo neto por tipo de manager × sector |
| `consensus.csv`, `put_call.csv` | Crowding (tenedores, compradores/vendedores netos) y nocional de puts vs calls por subyacente |

## Universo de managers

`config/managers.json` define el universo: CIK, nombre, tipo declarado y estilo. La muestra incluye 15 filers (Berkshire, Bridgewater, Renaissance, Citadel, Pershing Square, Elliott, Tiger Global, Baupost, Soros, Scion, Duquesne, Vanguard, BlackRock, Two Sigma, Appaloosa). Para expandir el universo basta añadir filas; `python -m sec13f.cli verify` confirma cada CIK contra EDGAR y `EdgarClient.lookup_cik("nombre")` ayuda a resolverlos.

## Cómo funciona

**Ingesta (`sec13f/ingest.py`, `sec13f/edgar_client.py`).** Usa la API `data.sec.gov/submissions/CIK##########.json` para listar filings 13F-HR/13F-HR/A, el índice JSON de cada accession para localizar el XML del *information table* y la portada, y guarda todo en `data/raw/<cik>/<accession>/`. El cliente respeta la política de acceso de la SEC (User-Agent descriptivo, ≤10 req/s, backoff en 403/429/5xx) y cachea en disco para que las re-ejecuciones no descarguen de nuevo.

**Parser (`sec13f/parser.py`).** Lectura del XML independiente del namespace (EDGAR lo ha cambiado varias veces). Normaliza fechas y unidades: los filings presentados antes del 3-ene-2023 reportan valores en miles de dólares, los posteriores en dólares. Agrega renglones duplicados de sub-managers, conserva puts/calls como posiciones separadas y cuadra el total contra la portada (`reconciliation_gap_pct`). Si hay enmiendas, prevalece el filing más reciente por trimestre.

**Clasificación (`sec13f/classify.py`).** Tres capas con score de confianza:
1. Maestro de valores por CUSIP (`config/issuers.json`; 220+ emisores con ticker, sector, industria y país; se amplía con el tiempo o se puede sustituir por un proveedor de datos).
2. Reglas sobre `titleOfClass`, `putCall`, `sshPrnamtType` y el nombre del emisor: común, ADR, ETF (y su bucket: renta variable amplia/internacional/sectorial/renta fija/commodity/temático), preferente, deuda/convertible, warrant, right, unit/SPAC, REIT, put, call.
3. Palabras clave sectoriales + un clasificador Naive Bayes sobre los tokens del nombre del emisor, entrenado con el maestro, para emisores desconocidos.

Además, un modelo de *huella de cartera* infiere el tipo de manager (índice, quant, multi-estrategia/opciones, macro/allocator, concentrado/activista, valor concentrado, crédito) a partir del número de posiciones, concentración, share de opciones/ETF/crédito y rotación, y lo compara con el tipo declarado.

**Tracker (`sec13f/tracker.py`).** Diff de cada libro entre trimestres consecutivos. El Δ de valor se descompone en **efecto flujo** (Δ títulos × precio implícito actual) y **efecto precio**. Calcula rotación (Σ|flujo| / valor promedio), exposición por tipo de activo y sector (ponderada por valor y equal-weight entre managers), rotación sectorial por tipo de manager, consenso (tenedores, compradores/vendedores netos, nuevos, salidas) y señal put/call.

**Análisis y salida (`sec13f/analysis.py`, `report.py`, `dashboard.py`).** Genera bullets de lectura del trimestre (flujos, mezcla de activos, rotación sectorial, consenso, mayores movimientos, rotación por manager, discrepancias de perfil, opciones), el reporte Markdown y el dashboard HTML (Plotly.js desde CDN, datos embebidos, tema claro/oscuro).

## Tests

```bash
python -m pytest -q
```

## Limitaciones del 13F que conviene recordar

Solo posiciones largas en valores 13(f) de EE.UU.; no hay cortos, bonos soberanos, derivados OTC ni acciones extranjeras sin ADR. El rezago es de hasta 45 días tras el cierre del trimestre. Las opciones se reportan por nocional del subyacente y sin distinguir compradas de suscritas. Los precios implícitos (valor/títulos) son aproximaciones.

## Roadmap

- Expandir el universo (top-100 hedge funds por AUM) y el histórico (8-12 trimestres); soporte para el *Form 13F data set* trimestral de la SEC como fuente alternativa de carga masiva.
- Enriquecer el maestro de valores con un proveedor externo (OpenFIGI / SEC `company_tickers.json`) para cubrir CUSIPs no vistos.
- Clasificación asistida por LLM para emisores ambiguos y para resumir la narrativa del trimestre.
- Alertas: nuevas posiciones de consenso, cambios de exposición sectorial mayores a N pp, picos de puts.
