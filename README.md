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

# 1) Muestra offline (no requiere red): 50 managers x 60 trimestres (2011Q3-2026Q2) con el esquema exacto de EDGAR
python -m sec13f.cli all --source sample --clean            # ~2 min; --quarters 8 para una corrida rápida

# 2) Datos reales de la SEC (requiere internet). La SEC exige identificarse en el User-Agent:
export SEC_USER_AGENT="Tu Nombre tu@email.com"
python -m sec13f.cli verify                      # comprueba los CIK del universo contra EDGAR
python -m sec13f.cli all --source sec --quarters 60 --clean  # ~3,000 filings; ~20-30 min a 8 req/s, cacheado en disco

# Re-generar solo los productos (sin volver a descargar). --detail-quarters controla cuántos
# trimestres llevan detalle por posición en el dashboard (las series agregadas siempre cubren todo).
python -m sec13f.cli build --detail-quarters 12
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

`config/managers.json` define el universo: CIK, nombre, tipo declarado, estilo, `since` (primer trimestre que reporta) y `previous_ciks` (CIKs anteriores cuyos filings se fusionan, p. ej. Elliott pasó de 1048445 a 1791786 en 2020). El universo actual tiene 50 filers: hedge funds (Berkshire, Bridgewater, Renaissance, Citadel, Millennium, D.E. Shaw, Two Sigma, AQR, Point72, Pershing Square, Elliott, Third Point, Icahn, ValueAct, Starboard, Trian, Jana, Tiger Global, Lone Pine, Viking, Coatue, Altimeter, Maverick, Baupost, Greenlight, Appaloosa, Glenview, Paulson, Tudor, Adage, Marshall Wace, Balyasny, Farallon, Scion), market makers (Jane Street, Susquehanna), family offices y fundaciones (Soros, Duquesne, Gates Foundation), fondos soberanos y de pensiones (Norges Bank, CalPERS) y asset managers (Vanguard, BlackRock, State Street, Fidelity, T. Rowe Price, Wellington, Dodge & Cox, Harris, Baillie Gifford). Para expandir basta añadir filas; `python -m sec13f.cli verify` confirma cada CIK contra EDGAR y `EdgarClient.lookup_cik("nombre")` ayuda a resolverlos.

## Cómo funciona

**Ingesta (`sec13f/ingest.py`, `sec13f/edgar_client.py`).** Usa la API `data.sec.gov/submissions/CIK##########.json` para listar filings 13F-HR/13F-HR/A, el índice JSON de cada accession para localizar el XML del *information table* y la portada, y guarda todo en `data/raw/<cik>/<accession>/`. El cliente respeta la política de acceso de la SEC (User-Agent descriptivo, ≤10 req/s, backoff en 403/429/5xx) y cachea en disco para que las re-ejecuciones no descarguen de nuevo.

**Parser (`sec13f/parser.py`).** Lectura del XML independiente del namespace (EDGAR lo ha cambiado varias veces). Normaliza fechas y unidades: los filings presentados antes del 3-ene-2023 reportan valores en miles de dólares, los posteriores en dólares. Agrega renglones duplicados de sub-managers, conserva puts/calls como posiciones separadas y cuadra el total contra la portada (`reconciliation_gap_pct`). Si hay enmiendas, prevalece el filing más reciente por trimestre. Para el histórico anterior a mediados de 2013 (cuando EDGAR aún no exigía XML) la ingesta guarda el *complete submission* de texto y un parser best-effort anclado en el CUSIP de 9 caracteres extrae emisor, clase, valor, títulos, put/call, discreción y voto; esas filas quedan marcadas con `text_format` en `filings.csv` para poder auditarlas.

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

## Datos de muestra

El generador (`sec13f/sample.py`) simula 60 trimestres con un factor de mercado que sigue los retornos trimestrales aproximados del S&P 500 (2011Q4-2025Q1; los posteriores son inventados), betas sectoriales, ruido idiosincrático, fechas de IPO por valor (`ipo` en `issuers.json`), fecha de primer reporte por manager (`since`) y algunos eventos guionizados (p. ej. la construcción y reducción de la posición de Berkshire en Apple). Los filings anteriores a 2023 se escriben en miles de dólares, como en EDGAR, para ejercitar la normalización. Ningún dato de tenencias es real.

## Roadmap

- Validar la ingesta contra SEC en vivo (este entorno no tenía salida a sec.gov); soporte para el *Form 13F data set* trimestral de la SEC como fuente alternativa de carga masiva.
- Enriquecer el maestro de valores con un proveedor externo (OpenFIGI / SEC `company_tickers.json`) para cubrir CUSIPs no vistos.
- Clasificación asistida por LLM para emisores ambiguos y para resumir la narrativa del trimestre.
- Alertas: nuevas posiciones de consenso, cambios de exposición sectorial mayores a N pp, picos de puts.
