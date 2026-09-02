# 13F Holdings Tracker

**Dashboard en vivo:** https://alanvaa06.github.io/13F_fillings/ (se publica automáticamente desde `output/dashboard.html` en cada push a `main`; el reporte queda en `/report.md` y los CSV en la raíz del sitio).

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
python -m sec13f.cli fetch --quarters 40                  # ~3,600 filings para 92 managers; ~1 filing/s, cacheado en disco
python -m sec13f.cli build --history-quarters 40         # parsea, clasifica y genera dashboard + reporte (10 anos)
python -m sec13f.cli enrich-issuers --top 2500           # opcional: amplia el maestro de emisores y vuelve a correr build

# Re-generar solo los productos (sin volver a descargar). --detail-quarters controla cuántos
# trimestres llevan detalle por posición en el dashboard (las series agregadas siempre cubren todo).
python -m sec13f.cli build --detail-quarters 12
```

Salidas en `output/`:

| Archivo | Contenido |
|---|---|
| `dashboard.html` | Dashboard interactivo: navegación por secciones, modo claro/oscuro, selector de trimestre, exposición por activo y sector con conmutador equal-weight / por valor (market weight), rotación por tipo de manager, tablas compactas ordenables por cualquier columna (managers, movimientos con peso previo y actual, consenso), scatter de concentración vs. amplitud con etiquetas activables, puts/calls y detalle por manager |
| `report.md` | Reporte trimestral en Markdown con la lectura del trimestre, tablas y metodología |
| `insights.json` | Insights estructurados por trimestre (para alimentar otros sistemas) |
| `holdings.csv` | Posiciones clasificadas (una fila por manager × trimestre × CUSIP × put/call) con peso y precio implícito |
| `changes.csv` | Diff trimestre a trimestre por posición: acción (NEW/EXIT/ADD/TRIM/HOLD), Δ títulos, efecto flujo y efecto precio |
| `manager_summary.csv` | Por manager y trimestre: valor, posiciones, concentración (top-10, HHI), share de opciones/ETF/crédito, rotación, tipo inferido |
| `exposure_*.csv`, `sector_rotation.csv` | Exposición por tipo de activo y sector (equal-weight y por valor), y flujo neto por tipo de manager × sector |
| `sector_positioning.csv` | Por sector y trimestre (acciones directas): peso EW de managers activos, Δ QoQ/YoY, gap vs promedio 8T, percentil histórico, benchmark implícito (managers índice) y externo si está configurado, peso activo, % de managers sobreponderados, flujo neto y efecto precio |
| `company_moves.csv` | Por empresa (CUSIP) y trimestre: tenedores antes/después, compradores/vendedores, Δ títulos agregados, flujo neto/bruto, efecto precio y clasificación del movimiento (MAJOR / MINOR / NONE) |
| `consensus.csv`, `put_call.csv` | Crowding (tenedores, compradores/vendedores netos) y nocional de puts vs calls por subyacente |

Con datos reales, `holdings.csv`, `changes.csv`, `consensus.csv`, `company_moves.csv` y `put_call.csv` pesan decenas de MB o varios GB, así que no se versionan (están en `.gitignore`) y no se publican en el sitio. `holdings.csv` y `changes.csv` solo se escriben con `build --position-csv`; sus equivalentes Parquet (`data/processed/`) se escriben siempre. El dashboard embebe solo las emisoras y subyacentes con más movimiento por trimestre (los KPIs sí usan el universo completo).

## Datos publicados

El sitio y `output/` se generan con datos reales de EDGAR: los últimos **40 trimestres** (2016Q3–2026Q2) de los **92 managers**, ~4,200 filings (originales y enmiendas) y ~16 millones de renglones de tenencias. Cosas que conviene saber al leer el corte actual:

- **Managers sin filing en el último trimestre** se listan en la lectura del trimestre (bullet *Cobertura*) y no entran en los agregados de ese periodo; el cambio del titular se calcula solo sobre los managers presentes en ambos trimestres. En 2026Q2 faltan Vanguard (último 13F público bajo su CIK: 2025Q4), Scion (dejó de reportar tras 2025Q3), Pershing Square (2026Q2 aún no presentado) y Amundi US (último 2021Q1; se conserva por su historia).
- **Norges Bank** presenta en Q1 y Q3 un 13F-HR *placeholder* (una fila `NA` / CUSIP `000000000` / $0) con todas las posiciones bajo tratamiento confidencial y publica la tabla real como 13F-HR/A cerca de un año después. El parser descarta esas filas, así que esos trimestres cuentan como sin datos hasta que llegue la enmienda.
- **Enmiendas**: ~500 de los libros trimestrales incorporan una 13F-HR/A (restatement o posiciones añadidas). `filings.csv` marca con `used` qué filings forman cada libro.
- **Unidades**: ~100 libros llegaron en miles cuando ya tocaba dólares (o al revés) y se reescalaron; el factor queda en `unit_factor`. Renaissance, Baupost, Flow Traders o T. Rowe siguieron reportando en miles varios trimestres después del cambio de la SEC.
- **Cambios de entidad filer**: BlackRock (1364742 → 2012383 desde 2024Q3; su 13F de 2016 cubría solo parte del grupo, de ahí el salto de 2017Q1), Jana (1159159 → 1998597), Greenlight, hoy DME Capital Management (1079114 → 1489933), y ARK (1579984 → 1697748) se fusionan vía `previous_ciks`.
- **Sector**: con el maestro enriquecido (~3,200 emisores) el valor *Unclassified* de la vista sectorial baja de ~28 % a ~1 %. Los sectores de los emisores enriquecidos derivan del código SIC de la SEC con una tabla de equivalencias estilo GICS y anulaciones puntuales; no son clasificaciones GICS oficiales.
- Para regenerar: `SEC_USER_AGENT="Tu Nombre tu@email.com" python -m sec13f.cli fetch --quarters 40 && python -m sec13f.cli build --history-quarters 40`, commit de `output/` y el workflow de Pages publica solo. La primera corrida parsea todo el XML (~1 h); las siguientes leen la caché por filing (`parsed.parquet` junto a cada filing).

## Universo de managers

`config/managers.json` define el universo: CIK, nombre, tipo declarado, estilo, `since` (primer trimestre que reporta) y `previous_ciks` (CIKs anteriores cuyos filings se fusionan, p. ej. Elliott pasó de 1048445 a 1791786 en 2020; BlackRock, Jana y Greenlight/DME también cambiaron de entidad filer). El universo actual tiene **92 filers**:

- **Hedge funds** (activistas, valor, long/short, event driven, macro, quant, multi-estrategia): Berkshire, Bridgewater, Renaissance, Citadel, Millennium, D.E. Shaw, Two Sigma, AQR, Point72, Pershing Square, Elliott, Third Point, Icahn, ValueAct, Starboard, Trian, Jana, TCI, Tiger Global, Lone Pine, Viking, Coatue, Altimeter, Maverick, D1, Whale Rock, Light Street, Dragoneer, Baupost, Greenlight/DME, Appaloosa, Glenview, Paulson, Tudor, Adage, Marshall Wace, Man Group, Balyasny, ExodusPoint, Hudson Bay, Farallon, Scion, Voloridge, PDT, Qube.
- **Market makers**: Jane Street, Susquehanna, Tower Research, Flow Traders.
- **Asset managers índice y activos**: Vanguard, BlackRock, State Street, Fidelity (FMR), T. Rowe Price, Wellington, Capital Research, Capital World, Geode, Northern Trust, Invesco, Franklin Templeton, Dimensional, Schwab, Amundi US, Dodge & Cox, Harris (Oakmark), Baillie Gifford, Polen, ARK, Akre; y los brazos de inversión de los bancos: Goldman Sachs, JPMorgan, Bank of America, Morgan Stanley, UBS, Wells Fargo.
- **Alternativos y conglomerados**: Apollo, Blackstone, Brookfield, Fairfax.
- **Soberanos, pensiones, fundaciones y family offices**: Norges Bank, Temasek, CPP Investments, Ontario Teachers, CalPERS, CalSTRS, NY State Common, SWIB, Gates Foundation, Harvard, Soros, Duquesne.

Para expandir basta añadir filas; `python -m sec13f.cli verify --cik <cik...>` confirma cada CIK contra EDGAR (la búsqueda por nombre de EDGAR, `browse-edgar?company=...&type=13F-HR&output=atom`, sirve para resolverlos). El `manager_type` declarado debe contener una de las palabras clave que entiende el comparador de perfil (index, quant, multi, activist, macro, value, opportunistic, growth, conglomerate, family office).

## Maestro de emisores y enriquecimiento

`config/issuers.json` arranca con ~220 emisores curados (con `ref_price`, que usa el generador de muestra) y crece con `python -m sec13f.cli enrich-issuers --top 2500`, que toma los CUSIP no clasificados con más valor en cartera y los resuelve en cuatro pasos: CUSIP → ticker/nombre con la API de OpenFIGI (coincidencia exacta; sin API key admite 25 peticiones/min, `OPENFIGI_API_KEY` la acelera), ticker → CIK con `company_tickers_exchange.json` de la SEC (con respaldo por nombre normalizado), CIK → código SIC vía la API de submissions, y SIC → sector estilo GICS con una tabla de rangos y anulaciones por descripción. Las entradas añadidas llevan `source`, `sic` y `cik` para auditarlas o sustituirlas por datos de un proveedor; los resultados se cachean en `data/cache/enrich_cache.json`. Tras enriquecer, vuelve a correr `build`.

## Cómo funciona

**Ingesta (`sec13f/ingest.py`, `sec13f/edgar_client.py`).** Usa la API `data.sec.gov/submissions/CIK##########.json` para listar filings 13F-HR/13F-HR/A, el índice JSON de cada accession para localizar el XML del *information table* y la portada, y guarda todo en `data/raw/<cik>/<accession>/`. El cliente respeta la política de acceso de la SEC (User-Agent descriptivo, ≤10 req/s, backoff en 403/429/5xx) y cachea en disco para que las re-ejecuciones no descarguen de nuevo.

**Parser (`sec13f/parser.py`).** Lectura del XML independiente del namespace (EDGAR lo ha cambiado varias veces). Normaliza fechas y unidades: los filings presentados antes del 3-ene-2023 reportan valores en miles de dólares, los posteriores en dólares. Agrega renglones duplicados de sub-managers, conserva puts/calls como posiciones separadas y cuadra el total contra la portada (`reconciliation_gap_pct`). Si hay enmiendas, prevalece el filing más reciente por trimestre. Para el histórico anterior a mediados de 2013 (cuando EDGAR aún no exigía XML) la ingesta guarda el *complete submission* de texto y un parser best-effort anclado en el CUSIP de 9 caracteres extrae emisor, clase, valor, títulos, put/call, discreción y voto; esas filas quedan marcadas con `text_format` en `filings.csv` para poder auditarlas.

**Enmiendas y unidades.** La ingesta baja *todos* los filings de cada trimestre (original y 13F-HR/A) y el parser los combina: una enmienda `RESTATEMENT` sustituye el libro, una `NEW HOLDINGS` añade las posiciones omitidas en el original (sin el tipo en la portada, una enmienda con al menos la mitad de filas que el libro se trata como restatement). Antes, una enmienda aditiva de una fila borraba el libro completo de ese trimestre. Los valores se normalizan a dólares con la regla de la SEC (miles antes del 3-ene-2023, dólares después), pero como muchos filers tardaron trimestres en cambiar de unidad, cada libro se compara con la mediana de sus trimestres vecinos: si difiere en más de 200x y reescalarlo por 1000 lo devuelve a rango, se corrige y el factor queda en `filings.csv` (`unit_factor`), junto con `used` (si el filing forma parte del libro final).

**Clasificación (`sec13f/classify.py`).** Tres capas con score de confianza:
1. Maestro de valores por CUSIP (`config/issuers.json`; 220+ emisores con ticker, sector, industria y país; se amplía con el tiempo o se puede sustituir por un proveedor de datos).
2. Reglas sobre `titleOfClass`, `putCall`, `sshPrnamtType` y el nombre del emisor: común, ADR, ETF (y su bucket: renta variable amplia/internacional/sectorial/renta fija/commodity/temático), preferente, deuda/convertible, warrant, right, unit/SPAC, REIT, put, call.
3. Palabras clave sectoriales + un clasificador Naive Bayes sobre los tokens del nombre del emisor, entrenado con el maestro, para emisores desconocidos.

Además, un modelo de *huella de cartera* infiere el tipo de manager (índice, quant, multi-estrategia/opciones, macro/allocator, concentrado/activista, valor concentrado, crédito) a partir del número de posiciones, concentración, share de opciones/ETF/crédito y rotación, y lo compara con el tipo declarado.

**Tracker (`sec13f/tracker.py`).** Diff de cada libro entre trimestres consecutivos. El Δ de valor se descompone en **efecto flujo** (Δ títulos × precio implícito actual) y **efecto precio**. Calcula rotación (Σ|flujo| / valor promedio), exposición por tipo de activo y sector (ponderada por valor y equal-weight entre managers), rotación sectorial por tipo de manager, consenso (tenedores, compradores/vendedores netos, nuevos, salidas) y señal put/call.

**Posicionamiento sectorial (`sec13f/sectors.py`).** Sobre el libro de acciones directas de cada manager (sin ETFs, opciones, deuda ni preferentes) calcula el peso sectorial equal-weight de los managers activos, su cambio QoQ/YoY, el gap contra el promedio de los últimos 8 trimestres, el percentil dentro de su historia y la descomposición flujo/precio del trimestre. El **benchmark implícito** es la mezcla sectorial ponderada por valor de los managers índice del universo (Vanguard, BlackRock, State Street), cuyos 13F replican de cerca el mercado estadounidense; el peso activo (EW − benchmark) y el % de managers sobreponderados miden dirección y amplitud del posicionamiento. Opcionalmente, `config/benchmarks.json` (ver `benchmarks.example.json`) añade índices externos como el S&P 500: cada corte se aplica *as-of* y el reporte muestra la fecha usada.

**Movimientos por empresa (`sec13f/movers.py`).** Agrega los diffs de todos los managers por CUSIP (solo acciones directas: común, ADR, REIT) y mide cada movimiento de dos formas: intensidad (Δ % de los títulos agregados en manos del universo, independiente del precio) y materialidad (flujo bruto en dólares). Clasifica cada empresa en cambio **mayor** (|Δ títulos| ≥ 10%, o ≥ 3 compradores/vendedores netos con |Δ| ≥ 3%), **menor** (|Δ| ≥ 1%) o **sin cambio** (por debajo de 1%); las entradas nuevas al universo cuentan como mayor. Los umbrales viven en `MoverThresholds` y se imprimen en el reporte.

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
