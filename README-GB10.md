# DS4 GB10 + DeepSeek V4 Flash + DSpark

Questa guida descrive la configurazione corrente del fork GB10 per eseguire
DeepSeek V4 Flash con il sidecar DSpark su una singola NVIDIA GB10. La priorità
è mantenere invariata la qualità del modello target, aumentare il prefill e
conservare il throughput DSpark sui contesti lunghi.

La sezione **Avvio rapido** è la procedura operativa aggiornata. Le sezioni
successive conservano la cronologia tecnica, comprese prove scartate e rollback.

## Esperimento Adaptive IndexCache scartato - 27 luglio 2026

Adaptive IndexCache e' stato implementato e misurato come tentativo di ridurre
il costo del selettore CSA nei 21 layer ratio-4. Il prototipo calcolava un
Top-512 exact in cinque layer Full (`2,12,22,32,42`) e permetteva ai sedici
layer Shared intermedi di riusarne gli indici. Query dell'attenzione, KV,
compressor e stati del layer target restavano locali; soltanto il set di righe
sparse veniva condiviso.

L'implementazione comprendeva tre modalita':

- `off`, identica al percorso exact stabile;
- `shadow`, che calcolava comunque il Top-512 exact e misurava su GPU
  l'intersezione con il set riusato, senza modificare l'output;
- `reuse`, zero-copy e senza memoria permanente aggiuntiva, che saltava
  selector-Q, scorer e Top-K nei layer Shared.

Il lifecycle invalidava la cache a ogni cambio di chunk, restore o reset; le
KV disk cache sperimentali erano separate per maschera. Un controllo `reuse`
con tutti i 21 layer Full ha conservato gli hash greedy del baseline, quindi
planner, allocazioni e invalidazioni non alteravano da soli il risultato.

### Profilo e limite teorico

Il profilo Nsight su Athena, chunk 8192 e contesto fisico 262144, ha misurato:

| `pos0` | Chunk GPU ms | Selector-Q ms | Score ms | Top-K ms | Quota selettore | Ceiling ideale 5F |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 8423,725 | 155,566 | 91,185 | 132,514 | 4,50% | 1,036x |
| 32768 | 8555,033 | 154,753 | 186,607 | 66,939 | 4,77% | 1,038x |
| 65536 | 8818,657 | 155,610 | 317,908 | 79,202 | 6,27% | 1,050x |
| 98304 | 9199,001 | 155,818 | 566,291 | 93,130 | 8,86% | 1,072x |
| 131072 | 9395,929 | 156,938 | 585,943 | 111,192 | 9,09% | 1,074x |

Il gate Amdahl originale richiedeva una quota del 26,25% per ottenere `1,25x`.
Nessuna profondita' lo ha superato: il massimo realistico misurato era circa
`+7,4%`, non `+25%`.

Il benchmark end-to-end ha confermato che il prototipo eliminava il lavoro
previsto:

| Frontiera | Exact `off` | 5 Full `reuse` | Delta | Hash greedy |
|---:|---:|---:|---:|---|
| 65536 | 989,59 t/s | 1018,81 t/s | +2,95% | diverso |
| 98304 | 944,90 t/s | 992,52 t/s | +5,04% | diverso |
| 131072 | 910,96 t/s | 972,15 t/s | +6,72% | diverso |

Rispetto al controllo all-F eseguito nello stesso ciclo, il guadagno a 131K
era `+7,58%`. Il risultato prestazionale era quindi reale e vicino al ceiling,
ma ottenuto cambiando il Top-512 accessibile al modello.

### Gate di qualita'

Lo shadow ha mostrato overlap medio fra circa 13% e 49% a seconda della coppia
source/target, con minimi per-token vicini a zero. Il riuso uniforme non era
quindi una cache exact: poteva escludere righe che il selettore del layer
target avrebbe scelto.

Il test qualitativo usato per la decisione e' stato
`tool-eval-bench --hardmode --seed 42 --temperature 0.95 --parallel 2`.
Il confronto pulito, senza pressione artificiale sul contesto, ha prodotto:

| Versione | Risultato | Tempo | Mediana turn | Errori HTTP |
|---|---:|---:|---:|---:|
| Fork stabile, IndexCache `off` | 77/100 | 399,2 s | 6,3 s | 0 |
| Antirez `80ebbc396aee40eedc1d829222f3362d10fa4c6c` | 77/100 | 803,1 s | 11,2 s | 0 |
| Prototipo 5 Full `reuse` | 60/100 | 640,9 s | 8,4 s | 0 |

Fork stabile e Antirez hanno avuto lo stesso esito in tutti i 15 scenari:
nove `PASS`, cinque `PARTIAL` e un `FAIL`. Il fork ha completato lo stesso test
in circa meta' tempo; il log server ha misurato 23,432 t/s di decode pesato su
8248 token generati.

Il candidato `reuse` ha invece degradato casi gia' superati, in particolare
TC73 e TC82 nel run completo. Le ripetizioni mirate hanno confermato che la
traiettoria e' sensibile al contesto, ma anche che la perdita non era soltanto
rumore: senza pressione sul contesto, `off` ha superato entrambi i casi mentre
il candidato ha ancora fallito TC73. Un precedente risultato `off` da 70/100
non viene usato come baseline, perche' includeva circa 15107 token filler per
scenario e quindi misurava anche la pressione del contesto.

### Decisione e archivio

Adaptive IndexCache offre un guadagno crescente sui contesti lunghi, ma la
maschera uniforme a cinque layer Full non preserva la selezione sparse e non ha
superato il gate qualitativo. Tutto il runtime, le API, la telemetria e i test
IndexCache sono stati rimossi dalla versione di produzione; non restano flag o
allocazioni inattive.

Il prototipo completo e' conservato localmente fuori dal repository in
`../ds4-gb10-indexcache-prototype-20260727/`. L'archivio contiene la patch
binaria ripristinabile, lo snapshot dei file modificati, i due tool non
tracciati, lo stato Git e il commit base. Una futura ripresa deve partire da
una maschera calibrata con NLL/top-1 agreement per layer, non dal solo overlap
Top-K o dal throughput.

## Esperimento HybridLC-2.1 scartato - 26 luglio 2026

Dopo `3ef444b` e' stata provata una revisione piu' complessa di HybridLC:
suffix a ordine variabile `16/12/8/6/4`, splice sulle frontier K/K-1,
distribuzione `q` sparse lossless, scratch host pinned, una sola readback del
verifier, restore D2D asincrono e dispatch BlockV differenziato per larghezza.
Erano inoltre disponibili, ma disattivati per default, semantic retrieval e un
local head causale.

La regressione CUDA e i self-check sono passati senza errori o fallback, ma il
test end-to-end non ha prodotto un miglioramento stabile:

- 1.476 cicli DSpark, acceptance verifier del 70,85% e 163,04 ms per ciclo;
- solo 21 cicli retrieval-hit; 185 token di coda proposti, 55 raggiunti e 40
  accettati, quindi coverage del 29,73% e acceptance condizionale del 72,73%;
- `N=8` ha raggiunto 27,39 t/s, mentre il costo del verifier largo ha ridotto
  `N=12` a 18,18 t/s e `N=16` a 13,48 t/s;
- il prefill e' rimasto forte, con 936 t/s sul segmento da 49K token e circa
  866 t/s a 147K, ma la risposta finale lunga ha chiuso a 15,69 t/s.

La conclusione e' che l'hardening era corretto, ma coverage rara e verifier
`N=12/16` troppo costoso annullavano il beneficio end-to-end. L'esperimento e'
stato archiviato integralmente fuori dal worktree e il codice e' tornato a
`3ef444b`. Eventuali sviluppi futuri devono preferire una strategia piu'
semplice, con beneficio misurabile prima di introdurre nuove larghezze target.

## HybridLC predefinito: DSpark + retrieval + BlockV - 26 luglio 2026

La versione stabile descritta in questa sezione è il commit `98c71c0`
(`Add lossless HybridLC speculative decoding`). I risultati riportati sotto
sono la caratterizzazione di rilascio eseguita su Athena; non includono le
successive prove variable-order/MoE-aware, archiviate e poi rimosse perché
riducevano l'acceptance del retriever.

Il launcher abilita per default HybridLC, che estende la catena neurale DSpark
con una coda recuperata dal contesto già verificato. Il percorso resta lossless:
il target verifica ogni proposta e BlockV applica la correzione dalla
distribuzione residua esatta. Il rollback operativo è
`DS4_DSPARK_HYBRID_LC=0`.

Prima della verifica target si può ancora misurare il retriever in shadow mode,
senza modificare token, `q`, CUDA Graph o output. Se production non è impostato
esplicitamente, il launcher lo disattiva automaticamente durante lo shadow:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_HYBRID_LC_SHADOW=1 DS4_TELEMETRY=1 ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-hybrid-shadow.log
```

L'avvio normale usa HybridLC senza richiedere flag. Per raccogliere la
telemetria completa:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_TELEMETRY=1 ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-hybrid-lc.log
```

L'implementazione completa mantiene una sola catena lineare e lossless:

- DSpark produce sempre il blocco neurale ufficiale da cinque slot; lo
  scheduler esistente sceglie il prefisso `K=1..5`;
- un indice hash incrementale cerca in `O(1)` un suffix esatto di otto token
  nelle ultime 131.072 posizioni. La sequenza viene sempre ricontrollata
  token-per-token, quindi una collisione hash non può produrre un match falso;
- l'indice conserva le due occorrenze più recenti e misura un match esteso fino
  a 64 token. Restore, rewind e canonicalizzazione invalidano l'oracle;
- quando il suffix esatto termina, una tabella bounded top-8 propone la
  transizione successiva. La sua `q` è la distribuzione normalizzata realmente
  campionata, non una confidence implicita;
- ogni draft conserva token, sorgente (`neural`, `suffix`, `transition`),
  supporto sparse di `q` e probabilità del token sul cammino;
- suffix e transizioni vengono materializzati nella matrice `q` con un kernel
  sparse; il suffix esatto è una riga one-hot;
- il target verifica una sola catena fino a 15 draft. Le sole nuove famiglie
  CUDA Graph sono le larghezze complessive `N=8`, `N=12` e `N=16`; `N=3` e
  `N=6` riusano l'oracle già misurato dal verifier DSpark;
- lo scheduler misura costo target e throughput per bucket di contesto e per
  larghezza, esegue due probe iniziali per N e poi sceglie il rate migliore
  rispetto al prefisso neurale. Un probe raro evita che il vincitore diventi
  permanente quando cambia il tipo di testo;
- `DS4_DSPARK_HYBRID_WIDTH=8|12|16` richiede una larghezza per benchmark
  controllati quando il retriever ha prodotto abbastanza righe; in caso
  contrario resta il prefisso neurale, senza padding fittizio;
- con HybridLC attivo, ogni ciclo DSpark full-vocabulary usa Block Verification,
  anche quando il retriever non aggiunge una coda
  (`arXiv:2403.10444`): prefisso accettato congiuntamente e correzione dalla
  distribuzione residua esatta `max(a_i p_i - q_i, 0)`;
- a temperatura zero resta il confronto greedy esistente;
- un riferimento CPU ricontrolla il prefisso BlockV al primo ciclo e poi ogni
  1.024 cicli. Una divergenza ripristina la snapshot e disabilita HybridLC per
  la sola sessione. L'intervallo si cambia con
  `DS4_DSPARK_HYBRID_SELF_CHECK_INTERVAL`;
- lo shadow oracle conserva una sola previsione pendente fino alla sua
  risoluzione, senza sostituirla al ciclo successivo. La coverage `>=8`/`>=16`
  usa come denominatore tutti i cicli shadow che interrogano il retriever;
  accuratezza token e match completi usano invece soltanto le code che diventano
  osservabili dopo l'accettazione integrale del prefisso neurale. Sono contati
  anche i rebuild causati da rewind/canonicalizzazione;
- acceptance della coda sotto il 20% attiva un cooldown di 64 cicli.

Con il flag spento il percorso corrente rimane invariato: cinque snapshot di
prefisso, sei righe di backup DSpark e gli stessi buffer `q`/verifier. Con il
flag acceso vengono allocati i buffer fino a 15 draft e le snapshot aggiuntive.
L'oracle host usa circa 10,5 MiB costanti: 262.144 slot suffix e una tabella
top-8 per il vocabolario. Non cresce con il contesto fisico, quindi il profilo
da 1M non crea una seconda struttura da un milione di elementi. Lo shadow mode
alloca soltanto l'oracle host, non i buffer target da 16 righe.

La regressione CUDA copre `q` one-hot e sparse, acceptance totale, rifiuto con
campionamento residuo e restore delle frontier:

```text
cuda-regression: HybridLC sparse q + lossless BlockV (rows=1..15) OK
cuda long-context regression: OK
```

Il report unificato separa suffix e transizioni, unisce la curva esistente
`N=3/6` a quella HybridLC `N=8/12/16`, controlla shadow e self-check e può
confrontare il decode con un log baseline:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./tools/hybrid_lc_report.py /tmp/ds4-hybrid-lc.log --baseline /tmp/ds4-baseline.log --require-all-widths --json /tmp/ds4-hybrid-lc-report.json
```

Il gate shadow segue gli oracle del paper: almeno il 20% dei cicli interrogati
con exact match `>=8`, oppure almeno il 10% con exact match `>=16`. Le soglie
sono modificabili con `--min-shadow-match8-coverage` e
`--min-shadow-match16-coverage`; `--min-shadow-accuracy` controlla
separatamente l'accuratezza token-per-token della coda proposta.

Nei log, `hybrid_enabled=1` indica tutti i cicli eseguiti con la feature di
produzione, `hybrid=1` soltanto quelli che hanno realmente aggiunto una coda
retrieval e `blockv=1` quelli verificati con BlockV. Questa separazione evita di
confondere un miss del retriever con un fallback o con una feature disattivata.

La validazione GB10 del commit `98c71c0`, eseguita il 25-26 luglio, ha superato
il gate del report:

- 1.406 cicli production, di cui 1.315 verificati con BlockV;
- 183 cicli retrieval, con 1.022 token accettati su 1.446 (`70,68%`);
- acceptance suffix `70,96%`, match esatto medio di 34,66 token;
- N=8 a 26,89 t/s, N=12 a 34,15 t/s e N=16 a 34,97 t/s;
- due self-check CPU/GPU superati, nessun errore, OOM o swap del processo;
- throughput interno dei cicli verifier 20,52 t/s su 1.406 cicli;
- decode telemetrico 20,48 t/s contro 20,06 t/s baseline; il run silenzioso ha
  misurato 21,30 t/s. Tra 25K e 100K i turni operativi sono rimasti fra 25,84
  e 26,73 t/s, con picchi superiori a 30 t/s;
- i prefill append sostanziali sono rimasti fra circa 851 e 907 t/s.

Il vantaggio dipende dalla copertura: nella risposta finale libera da 2.195
token il retrieval è intervenuto soltanto nel 6,1% dei cicli e il decode è
sceso a 16,37 t/s. Quando il suffix era disponibile, N=12/N=16 ha mantenuto
31,29/35,32 t/s. Il risultato promuove quindi HybridLC perché accelera in modo
marcato le regioni ripetitive senza cambiare la distribuzione target, non perché
elimini il costo della generazione neurale non coperta.

Production e shadow mode restano mutuamente esclusivi. Se entrambe vengono
impostate esplicitamente a `1`, il launcher rifiuta la configurazione.

## Frontier GPU per tool call - 24 luglio 2026

Le richieste con tool conservano ora fino a quattro snapshot della frontier del
prompt prima del decode. Il ring resta sul device e ogni slot contiene soltanto
lo stato mutabile necessario a ripartire esattamente da quel punto:

- le ultime 128 righe raw SWA dei 43 layer;
- le frontiere attention/indexer e i relativi contatori;
- i tre ring main-KV DSpark;
- token e logits gia' disponibili sul lato host.

La compressed KV non viene duplicata: e' append-only, quindi il restore
riposiziona i contatori e sovrascrive soltanto la nuova coda. Ogni slot usa una
singola arena CUDA bounded da circa 23,14 MiB; il massimo osservato per quattro
slot e' 92,56 MiB e non introduce copie device-to-host nel percorso prima del
decode. Gli slot vengono identificati esplicitamente e sostituiti in ordine
LRU, così una tool call può tornare a una frontier precedente senza perdere le
altre ramificazioni ancora valide.

Quando la canonicalizzazione di una tool call sostituisce la coda generata, il
server prova prima `source=gpu-frontier` e processa soltanto la coda canonica.
Se token o layout non coincidono, rimangono invariati i fallback su checkpoint
disco e replay completo. I log utili per il gate Athena sono:

```text
ds4: CUDA prompt frontier ring active (slots=4 ... MiB/slot max=... MiB device, append-only compressed KV)
ds4-server: GPU prompt frontier captured id=... tokens=... ...ms
ds4-server: GPU prompt frontier restored id=... cached=... prompt=... replay=...
```

Il gate controlla regressione CUDA completa, hash/qualita' invariati, decode
DSpark non inferiore al baseline e riduzione del replay dopo una tool call.
Su Athena e' stato superato:

```text
cuda-regression: frontier ring + batched async D2D restore OK
cuda long-context regression: OK
ds4: CUDA prompt frontier ring active (slots=4 23.14 MiB/slot max=92.56 MiB device, append-only compressed KV)
ds4-server: GPU prompt frontier captured id=1 tokens=345 2.388ms
ds4-server: GPU prompt frontier restored id=1 cached=345 prompt=408 replay=63
ds4-server: chat ctx=345..408:63 TOOLS prompt done 0.512s
```

Nel test controllato la coda tool e' stata resa deliberatamente diversa
(`Rome` contro `Milan`) per impedire il normale live-prefix hit. Il server ha
ripristinato 345 token dalla frontier GPU e ne ha riprocessati soltanto 63.
La risposta deterministica a `temperature=0` e' risultata identica al cold
prefill dello stesso prompt, incluso l'hash SHA-256 normalizzato:

```text
10f62d5605334a5f239c563ba80d0b2a26257b39d482beef8dfe5f24a3bd8f27
```

La cattura avviene fuori dal loop di decode, soltanto sulle richieste con tool.
Nel test ha richiesto 1-2,4 ms; il fallback disco/full replay resta attivo se il
prefisso testuale, l'identificatore o il layout della snapshot non coincidono.

Il successivo test end-to-end con Claude Code ha validato quattro restore
consecutivi fino a 126.876 token:

| Frontier | Coda nuova | Prefill coda | Tempo |
| ---: | ---: | ---: | ---: |
| 24.979 -> 28.545 | 3.566 token | 903,2 t/s | 3,95 s |
| 28.545 -> 88.550 | 60.005 token | 920,6 t/s | 65,18 s |
| 88.550 -> 112.476 | 23.926 token | 868,4 t/s | 27,55 s |
| 112.476 -> 126.876 | 14.400 token | 849,6 t/s | 16,95 s |

Le cinque catture sono costate complessivamente circa 6,9 ms e hanno occupato
23,14 MiB. I quattro restore hanno evitato di riprocessare 254.550 token di
prefisso. Stimando il vecchio full replay alle velocita' osservate nello stesso
run, il tempo cumulativo dei quattro prefill sarebbe stato circa 407 secondi;
il percorso incrementale ne ha richiesti 114, con circa 293 secondi risparmiati
e una riduzione stimata del TTFT cumulativo del 72%.

Nello stesso run il decode DSpark ha misurato 26,4 t/s a 25K, 26,1 t/s a 28K,
22,4 t/s a 88K e 21,2 t/s a 112K. La generazione finale di 3.035 token a 126K
ha mantenuto 16,2 t/s. La memoria di sistema e' rimasta intorno a 115 GiB usati
e il processo DS4 non ha usato swap.

## Scheduler DSpark sensibile al contesto - 24 luglio 2026

La telemetria del verifier distingue ora fasce hardware da 32K token e conserva
per ogni `K=1..5` costo del target, throughput recente e acceptance
condizionale. Lo scheduler usa questi dati locali insieme alla confidence head:
può ridurre K nella prosa lunga a bassa acceptance senza restare bloccato sul
campione storico raccolto durante tool call più prevedibili. Un probe raro
mantiene osservabili le altre ampiezze e permette loro di tornare candidate.

Con `DS4_DSPARK_ALWAYS_DRAFT=1`, il draft e' gia' stato calcolato quando viene
scelto K. In questo caso il minimo è K=1: scegliere K=0 dopo aver pagato il
drafter ha prodotto una regressione misurata ed e' stato escluso. Rejection
sampling, logits target, RNG e criterio di acceptance non sono cambiati.

Confronto controllato sullo stesso prompt, sidecar Q2, frontiera 98.304, chunk
8192 e 1.024 token generati:

| Build | Prefill | Decode | Cicli | Token/ciclo | Picco RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline frontier ring | 922,19 t/s | 10,96 t/s | 671 | 1,5261 | 2.484,8 MB |
| scheduler context-aware | 920,00 t/s | 11,07 t/s | 660 | 1,5515 | 2.473,9 MB |

Il guadagno decode di questo carico severo e' circa 1%; il prefill differisce
dello 0,24%, entro il rumore del run singolo. La telemetria si abilita con
`DS4_TELEMETRY=1`; le righe `dspark scheduler` includono `pos`, bucket,
confidence, acceptance recente, costo e rate per K.

E' stato anche provato il warm-start GVR exact del Top-512 seriale in broadcast
sulle 2-6 righe del verifier. La parita' esatta e il fallback Radix hanno
superato la regressione, ma K=1 e' passato da 122,5 a 125,3 ms nel confronto
end-to-end. L'estensione multi-riga e' stata quindi rimossa; resta soltanto il
GVR exact a token singolo gia' validato.

## Scheduler hardware-aware DSpark - 27-29 luglio 2026

Il canary implementa ora direttamente l'Algoritmo 1 del paper
[DSpark](https://arxiv.org/html/2607.05147v1), separato dal runtime CUDA in un
modulo C puro e testabile. L'interfaccia accetta gia' `R` richieste identificate
stabilmente, confidence condizionali diverse e una curva hardware `SPS(B)`
comune. L'executor CUDA multi-sessione e' attivo nel backend fino a `R=3`. Il
server HTTP dispone ora di due lane persistenti e puo' riunire due cicli DSpark
indipendenti in una coorte fisica `R=2`.

Per ogni richiesta vengono calcolate le probabilita' di sopravvivenza cumulative
`a[r,j] = product(c[r,1..j])`. Tutte le estensioni `(r,j)` vengono ordinate
globalmente, con tie-break deterministico che preserva la chiusura del prefisso.
Partendo dalle `R` righe target obbligatorie, ogni estensione aumenta di una
riga il batch fisico e di `a[r,j]` i token attesi. Il criterio massimizzato e':

```text
Theta(B) = expected_tokens(B) * SPS(B)
```

La variante teorica si arresta al primo candidato che non migliora `Theta`, come
richiesto dal vincolo causale del paper. `K=0` rimane una decisione valida. Sul
GB10 a singola GPU la curva usa il rate effettivo del ciclo serializzato
`1 / (draft + target)`: in questo modo comprende il costo fisso del sidecar che
nel deployment disaggregato del paper puo' essere sovrapposto ad altro lavoro.

Il canary implementa anche l'adattamento production della sezione 5.2. La
capacita' fisica viene scelta con una ricerca globale, senza early-stop, usando
esclusivamente le confidence di due cicli prima; quella capacita' ormai fissa
viene poi riempita con i candidati correnti a maggiore sopravvivenza. La
barriera temporale consente di attraversare i gradini non monotoni di
`SPS(B)` senza far dipendere l'ammissione di un token da una sua realizzazione
futura. La storia e' indicizzata per identita' e numero di step globale: viene
usata soltanto una misura del preciso step `t-2`, non semplicemente la seconda
osservazione precedente della richiesta. Startup, join o un buco nella
pianificazione fanno ricadere l'intera coorte sull'Algoritmo 1 causale per
quello step. Ogni reset della sessione azzera la storia ritardata.

Il profilo conserva per ogni `K=1..5` una finestra robusta di nove misure,
scarta i primi due warm-up e usa mediana e MAD. I costi sono locali a fasce di
contesto da 32K token, con fallback globale. Un probe periodico aggiorna la
curva scegliendo soltanto da contatori e misure storiche: non consulta token o
confidence future del ciclo corrente e quindi non introduce selection bias.
Durante il bootstrap il probe usa una cadenza piu' breve e resta attivo anche
prima che tutte le forme siano pronte; questo evita il lock-in in cui una forma
mai scelta non poteva raccogliere i campioni necessari ad abilitare il canary.

E' definito anche il contratto fisico flattened della sezione 5.3. Per ogni
richiesta contiene identita', offset, numero di righe e posizione assoluta; per
ogni riga contiene marker di richiesta, posizione e indice del prefisso
(`0` per il pending target, `1..K` per il draft). Il piano e' request-major,
senza padding, e verifica che la somma delle righe coincida con il batch scelto.
Il contratto include ora anche token fisici contigui, stato RNG indipendente per
richiesta e scatter del prefisso accettato verso la corretta continuation row.
I test coprono coorti eterogenee (`K=2,0,1`), assenza di padding e rifiuto di un
commit che attraversi il confine fra richieste.

Il percorso CUDA reale condivide modello e workspace transiente, mantiene KV,
compressor, ring DSpark e distribuzioni `q` privati per sessione, appiattisce le
righe request-major senza padding e lancia una sola pipeline target. La
transazione conserva le frontier pre-verifica, applica rejection sampling p/q
direttamente alle slice di logits sul device e committa un prefisso indipendente
per lane senza rieseguire il verifier. Su GB10, sidecar Q2 e una coorte
eterogenea `R=3` da `2+3+3=8` righe, il confronto con tre verifiche sequenziali
ha dato parita' numerica e decisionale esatta (`rel-rmse=0`) con contesto fisico
allocato a 262K e frontiera di prova a 4.096 token. Il verifier e' passato da
`367,751 ms` a `177,127 ms`, cioe' `2,076x` e `45,17` righe target/s aggregate.

La prima implementazione R=3 aveva esposto una discontinuita' nascosta: otto
righe uscivano dai fast path Q8 e MoE limitati a `N<=6`, producendo sia
divergenza numerica sia una perdita di throughput. Il percorso Q8 weight-reuse
e il MoE aligned GB10 coprono ora esattamente `N<=8`; per `N<=6` resta una
specializzazione separata, cosi' R=2 non paga registri o shared memory
aggiuntivi. Le due lane oltre la principale usano `4.493,71 MiB` complessivi di
stato privato; non aumentano la memoria permanente del normale serving `R=1`.
La preparazione valida prima l'intera coorte e, in caso di errore CUDA,
ripristina gli RNG di tutte le lane. `R=3` e' il limite operativo scelto sul
profilo da 128 GiB; `R=4` non lascia un margine RAM accettabile.

Il coordinatore conserva inoltre lo stesso contratto HybridLC del percorso
`R=1`: ogni lane puo' estendere il prefisso neurale con righe retrieval sparse,
usa BlockV lossless e mantiene il self-check CPU/GPU periodico sulle proprie
slice di logits. Un fallimento prima del commit ripristina anche gli uniformi
RNG aggiuntivi della coda; nel fallback seriale, una lane gia' conclusa non viene
eseguita una seconda volta. Restore KV, rewind, invalidazione e un ritorno
temporaneo a `R=1` svuotano le confidence ritardate ma preservano il contatore
globale di maturita'. I due successivi passi `R>1` restano quindi causali, senza
riportare ogni volta il gate nel warm-up fisico. Un fallback del verifier
ripristina inoltre gli RNG target e HybridLC soltanto per le lane non
committate. Queste correzioni non cambiano logits target, regola p/q, Top-K
accessibile o distribuzione campionata.

Un canary `R=2` separato, sulla stessa allocazione 262K e frontiera 4.096, ha
confermato logits, Top-1, rejection p/q e commit indipendente esatti
(`rel-rmse=0`). Il verifier e' passato da `249,518 ms` sequenziali a
`212,170 ms` fisici, cioe' `1,176x` e `28,28` righe target/s aggregate; la
seconda lane ha richiesto `2.246,36 MiB` di stato privato.

### Nightjar: scelta pre-draft della profondita' - 31 luglio 2026

Il canary Nightjar completa la parte adattiva prima del drafter. L'unita' di
scelta e' il budget aggregato di proposte della coorte: `B=0` esegue il
target-only nel singleton, mentre `B=1..R*5` sceglie la capacita' fisica totale.
In `R>1`, `B=0` resta escluso finche' il target-only non dispone di un vero
executor fisico: far competere R decode seriali falserebbe la curva hardware.
Il segnale storico non congela pero' l'assegnazione alle lane. Il drafter
produce il blocco corrente completo e, una volta disponibili le confidence
aggiornate, il budget viene redistribuito globalmente; sono quindi possibili
profondita' come `[0,4,2]` senza imporre lo stesso K a tutte o sprecare una
proposta su ogni sessione. Questa separazione
e' essenziale: `t-2` determina soltanto la capacita', mentre il passo corrente
conserva rigorosamente il rank dei candidati piu' sicuri come richiesto dalla
Sezione 5.2 del paper.

Ogni contesto e' separato per numero di richieste residenti, vettore ordinato
delle fasce di contesto e sorgente del draft. Quando disponibile, il budget
usa la confidence causale esatta di `t-2`; startup e buchi di scheduling
ricadono sul prior di acceptance recente. Per assorbire differenze fra la STS
held-out e il target quantizzato in esecuzione, la confidence ritardata pesa il
60% e l'acceptance recente della lane il 40%. I bracci usano ADA-BINGREEDY con
il reward realmente osservato, calcolato come latenza totale divisa per token
emessi. Solo il migliore secondo il prior SPS riceve il bootstrap iniziale;
gli altri budget entrano in probation se diventano competitivi o durante una
esplorazione locale entro l'8% del migliore, limitata al 10% dei round
eleggibili. L'esplorazione prova il budget sicuro piu' vicino, preferendo il
meno campionato, e non puo' quindi saltare casualmente da una forma corta a
una forma estrema. Un
braccio richiede due osservazioni prima di poter bloccare un bin. Alla media
storica si affianca una EWMA recente; dopo quattro campioni il reward misurato
pesa il 90% e il prior soltanto il 10%. Un braccio bloccato che degrada viene
revocato subito, e le prove esplorative non occupano un intero bin quadratico.
Il costo di riattivazione viene addebitato soltanto nel passaggio
`B=0 -> B>0`.

Nel singleton ogni braccio resta il solo budget `B`. In `R>1` il braccio e'
invece la coppia `(executor, B)`: physical e serial/lane-partitioned mantengono
prior, campioni, reward e persistenza indipendenti. Nightjar confronta quindi
la latenza prevista della forma che verra' davvero eseguita, invece di usare il
minimo fra due executor e attribuire poi il risultato all'altro. Le curve SPS
offline forniscono il fallback. Gli arm sono interleaved per budget, quindi
physical e serial dello stesso `B` sono vicini anche per l'esplorazione locale;
dopo otto campioni l'EWMA della forma esatta ha priorita'. HybridLC continua a
essere lossless e indipendente; per non
attribuire al drafter neurale token prodotti dal retrieval, Nightjar aggiorna
il proprio reward soltanto sulle coorti pure-neural.

Il percorso storico resta identico quando il canary e' disattivato: draft fisso
da cinque slot e scheduler hardware-aware esistente. Per osservare le decisioni
senza cambiare l'esecuzione:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_NIGHTJAR_SHADOW=1 DS4_TELEMETRY=1 ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-nightjar-shadow.log
```

Per il canary attivo isolato da HybridLC:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_NIGHTJAR=1 DS4_DSPARK_HYBRID_LC=0 DS4_TELEMETRY=1 ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-nightjar.log
```

Nightjar e' ora promosso fino a `R=3`, dopo gate controllati separati per il
singleton e per le coorti multi-sessione. Per isolarlo da HybridLC durante una
misurazione:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_NIGHTJAR=1 DS4_DSPARK_NIGHTJAR_MAX_R=3 DS4_DSPARK_HYBRID_LC=0 DS4_TELEMETRY=1 ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-nightjar-r3.log
```

Il test operativo Q2 a `R=1`, dopo la correzione del reward e del bootstrap,
ha prodotto `19,41 t/s` su 512 token, con soltanto due cicli `B=0`; i bracci
dominanti sono stati `B=2` (`20,50 t/s`, acceptance `84,66%`) e `B=5`
(`19,11 t/s`). Due richieste simultanee hanno formato 245 coorti fisiche su
318 e misurato `19,82 t/s` aggregate; la forma fisica `R=2` ha raggiunto
`20,20 t/s` contro `17,07 t/s` del fallback seriale nello stesso run. Questi
risultati sono canary operativi, non sostituiscono ancora il baseline
controllato gia' registrato per tutti i bucket di contesto.

La correzione causale e' stata verificata anche con A/B puliti `R=3`, cache KV
separate, prompt tecnico, temperatura `0,95` e 256 token per lane. La baseline
hardware-aware ha ottenuto `14,86 t/s` aggregate. La prima implementazione,
che usava erroneamente `t-2` anche per scegliere la lane, scendeva a
`14,47 t/s` e al `37,24%` di acceptance. Dopo la separazione fra capacita'
storica e rank corrente, due run intermedi hanno ottenuto `14,99` e
`15,04 t/s`. Il canary completo, con budget `B=1..R*5` ed esplorazione locale,
ha chiuso a `14,88 t/s`: neutro sul wall-clock rispetto alla baseline, ma con
throughput del solo ciclo verifier salito da `12,446` a `13,361 t/s`
(`+7,35%`). Il throughput delle coorti e' passato da `15,400` a
`15,425 t/s`. Il beneficio interno viene ancora assorbito da divergenza delle
lane, tail e probe fisici `R=3`; questo risultato ha mantenuto Nightjar opt-in
per le coorti multi-sessione. Il successivo gate R1 ne ha invece consentito la
promozione nel singleton, senza estendere automaticamente la policy a R2/R3.

Topologia target, logits, KV, RNG e rejection sampling p/q non cambiano. Il
numero di proposte puo' cambiare e quindi una risposta campionata puo' divergere
bit per bit, ma ogni token resta accettato o corretto dal modello target con la
stessa distribuzione lossless.

#### Persistenza bounded e guard anti-regressione - 1 agosto 2026

Il canary ora conserva nel medesimo profilo hardware sia le forme dell'executor
(`P`) sia i reward Nightjar maturi (`N`). Il file resta legato al fingerprint
esatto di binario, modello target e sidecar: una build o un GGUF differente
viene ignorato, senza riutilizzare misure incompatibili. Un braccio entra nel
file soltanto dopo quattro osservazioni. Al riavvio vengono ripristinati al
massimo otto campioni storici e quattro recenti; lock, round e revoche non sono
mai persistiti, perche' contesa e request mix possono essere cambiati.

Il percorso attivo applica inoltre due guard lossless. Prima del draft rifiuta
una stima peggiore del miglior braccio noto oltre la soglia configurata; dopo il
draft confronta il budget bloccato con i candidati physical e serial calcolati
sulla forma realmente eseguibile. Il default ammette al massimo il 3% di
regressione prevista:

```bash
DS4_DSPARK_NIGHTJAR_MAX_PREDICTED_REGRESSION=0.03
DS4_DSPARK_NIGHTJAR_MAX_R=3
```

Quando il guard interviene, viene eseguito lo scheduler hardware-aware stabile.
Il reward viene ora attribuito al budget effettivamente eseguito, non al budget
Nightjar rifiutato: il canary impara quindi anche dalle proprie correzioni. Nel
primo test freddo questa chiusura del feedback ha registrato 148 coorti protette
su 152 reward valide; alla chiusura sono stati salvati 10 bracci. Il secondo
avvio dello stesso binario ha caricato correttamente `nightjar_arms=10`,
verificando il warm-start end-to-end.

Per evitare i test operativi da circa 40 minuti, la matrice breve esegue
`R=1,2,3` nello stesso comando e salva un JSON riproducibile:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && python3 tools/benchmark_dspark_concurrency.py --concurrency 1 2 3 --runs 2 --max-tokens 256 --temperature 0.95 --label nightjar-short --json-output /tmp/ds4-nightjar-short.json
```

Un confronto rapido a parita' di binario, profilo caldo e 128 token per lane ha
misurato:

| Coorte | Hardware-aware stabile | Nightjar warm | Delta |
| --- | ---: | ---: | ---: |
| `R=1` | 11,71 t/s | 14,84 t/s | +26,7% |
| `R=2` | 14,01 t/s | 13,06 t/s | -6,8% |
| `R=3` | 13,32 t/s | 13,14 t/s | -1,4% |

E' un test diagnostico a un solo run, quindi non sostituisce la matrice lunga.
Questo primo test spiegava perche' Nightjar non era ancora promosso: il
target-only adattivo appariva promettente nel singleton, mentre il budget
adattivo multi-sessione non mostrava ancora un vantaggio stabile. Il successivo
gate executor-paired, riportato sotto, ha superato questa limitazione per R1.

La successiva correzione executor-paired e' stata verificata sullo stesso
binario, con 128 token per lane, temperatura `0,95`, profilo hardware condiviso
e un run per punto. I numeri sono diagnostici, ma confermano che reward e costo
non vengono piu' mescolati fra executor:

| Coorte | Nightjar off | Paired cold | Paired warm |
| --- | ---: | ---: | ---: |
| `R=1` | 12,64 t/s | 14,34 t/s (+13,4%) | 14,14 t/s (+11,9%) |
| `R=2` | 12,92 t/s | 14,86 t/s (+15,0%) | 13,35 t/s (+3,3%) |
| `R=3` | 11,82 t/s | 11,98 t/s (+1,4%) | 12,42 t/s (+5,1%) |

Nel passaggio warm il guard ha ancora protetto il 49,5% delle decisioni. Gli
arm Nightjar hanno preferito il physical executor; le misure effettive hanno
confermato che il seriale era piu' lento nelle coorti osservate. Per questo il
supporto `R=2/3` era rimasto un canary esplicito e
`DS4_DSPARK_NIGHTJAR_MAX_R=1` il limite predefinito. Il gate dedicato seguente
ha poi verificato l'estensione sulle sessioni multiple con un campione piu'
lungo e il guard anti-regressione attivo.

#### Gate di promozione R1

Il gate definitivo ha confrontato lo stesso binario e profilo hardware con
HybridLC attivo, cinque run da 256 token, prompt e seed invariati:

| Metrica | Nightjar off | Nightjar R1 | Delta |
| --- | ---: | ---: | ---: |
| Mediana decode | 13,28 t/s | 15,83 t/s | +19,2% |
| Intervallo dei cinque run | 13,17..14,28 | 15,39..15,90 | separato |
| Acceptance verifier | 43,22% | 48,39% | +5,17 punti |
| Swap processo | 0 | 0 | invariato |

Il fixture greedy ha poi avviato baseline e candidato come processi separati:
5/5 prompt hanno prodotto contenuto, reasoning e finish reason identici. Il
candidato ha committato gli stessi 200 token draft del baseline. Nightjar viene
quindi promosso per il singleton. Il rollback completo resta
`DS4_DSPARK_NIGHTJAR=0`.

#### Gate di promozione R2/R3

Il gate multi-sessione ha confrontato lo stesso binario, lo stesso profilo
hardware privo di reward Nightjar preesistenti e tre run da 256 token per
coorte, con HybridLC attivo, temperatura `0,95` e seed invariato:

| Coorte | Hardware-aware stabile | Nightjar | Delta |
| --- | ---: | ---: | ---: |
| `R=2` mediana | 14,70 t/s | 15,28 t/s | +3,9% |
| `R=2` intervallo | 14,20..14,77 | 14,24..15,77 | - |
| `R=3` mediana | 13,89 t/s | 14,74 t/s | +6,1% |
| `R=3` intervallo | 13,83..14,42 | 14,13..14,74 | - |
| Acceptance verifier | 43,91% | 44,80% | +0,89 punti |
| Swap processo | 0 | 0 | invariato |

Il guard ha respinto l'84,73% delle decisioni Nightjar non sufficientemente
convincenti e ha eseguito al loro posto il piano hardware-aware. Il vantaggio
misurato deriva quindi da un numero limitato di scelte adattive protette dal
fallback, non da esplorazione indiscriminata. Dopo questo gate,
`DS4_DSPARK_NIGHTJAR=1` e `DS4_DSPARK_NIGHTJAR_MAX_R=3` sono i default per
`R=1..3`; la soglia anti-regressione resta al 3%.

Prima del coordinatore, un test operativo con due client HTTP indipendenti ha
mantenuto due socket contemporanei ma ha eseguito le richieste in alternanza
strettamente seriale. Fra `chatcmpl-9` e `chatcmpl-26` sono stati generati 7.953
token in 368,2 secondi di decode (`21,60 t/s` ponderati); includendo prefill e
attese in coda, il throughput wall-clock e' stato circa `17,36 t/s`. Le tool
call si sono mosse prevalentemente fra 21 e 30 t/s, mentre le due risposte
finali lunghe hanno chiuso a `17,17 t/s` (2.200 token) e `19,86 t/s` (940
token). Gli append prefill sostanziali sono rimasti circa fra 949 e 987 t/s.

### Coordinatore HTTP R=2 e canary R=3

Il coordinatore mantiene da due a tre sessioni con KV, compressor, RNG e stato
DSpark indipendenti. I pesi e lo scratch CUDA transiente restano condivisi. Il
dispatcher considera vincolante il `call_id` di una tool continuation, evitando
che una risposta torni sulla lane sbagliata. Per le richieste non vincolate
sceglie prima la lane con meno job assegnati e usa il prefisso piu' lungo come
tie-breaker fra lane con lo stesso carico. Questo evita che una lane con una
buona affinita' storica monopolizzi anche client indipendenti, costringendola a
scaricare e ricaricare checkpoint KV mentre la seconda lane resta vuota. Due
o tre worker si incontrano ai confini dei cicli speculativi per un massimo
predefinito di 500 microsecondi. Il rendezvous mira a tutte le lane attive,
evitando che la prima coppia parta prima dell'arrivo della terza. Con
`DS4_TELEMETRY=1`, la riga
`coordinator dispatch` espone lane, carico, prefisso comune e lunghezza prompt.

Il costo minimo del microbatch GB10 rende `K=1` equivalente, come forma
hardware, al vecchio `K=0` con una riga ombra. Lo scheduler include quindi quel
primo token nel baseline: la stessa riga pagata diventa lavoro utile e resta
soggetta alla normale verifica target lossless.

Il runtime misura separatamente il rate delle forme fisiche e quello delle
verifiche seriali che riusano il draft gia' calcolato. Dopo 8 cicli iniziali
sceglie la forma fisica soltanto quando supera la previsione seriale di almeno
l'1%; le forme sconosciute vengono sondate ogni 64 cicli. In caso contrario
esegue i verifier per lane senza rigenerare il draft. Questa selezione non
modifica logits, p/q rejection, RNG, KV o distribuzione del modello target.

La stabilizzazione successiva (`cdfaaa3`) rende il profilo sensibile alla forma
esatta: executor fisico o seriale, percorso neural o HybridLC, bucket di
contesto, numero di richieste, righe e vettore dei prefissi. Ogni richiesta
decode riceve inoltre un'identita' monotona valida per tutto il proprio
lifecycle; al termine viene rimossa dalla storia dello scheduler. Questo evita
che join, leave o il riuso di una lane contaminino le confidence ritardate di
un'altra richiesta. Il percorso fisico supporta sia greedy sia rejection p/q
stocastica con uniformi e RNG privati per lane; un errore prima del commit
ripristina gli RNG interessati.

Anche un cohort temporaneamente ridotto a `R=1` resta ora nel physical
executor. Questo evita la cattura a freddo della famiglia CUDA Graph legacy
quando una delle due richieste termina prima dell'altra e, soprattutto, non
azzera la storia hardware-aware a ogni rendezvous mancato. Un server avviato
con una sola lane continua a usare il percorso storico; il rollback esplicito
del singleton coordinato e' `DS4_DSPARK_COORDINATOR_MODE=serial`.

Con due richieste fisse da 512 token sullo stesso server:

| Modalita' | Tempo wall | Throughput aggregate |
| --- | ---: | ---: |
| baseline, una lane seriale | 63,701 s | 16,075 t/s |
| `R=2` fisico sempre attivo | 81,140 s | 12,620 t/s |
| `R=2` con minimo hardware `K=1` | 66,940 s | 15,297 t/s |
| coordinatore adattivo | 64,019 s | 15,995 t/s |
| HC/QKV fisico + singleton legacy | 61,953 s | 16,529 t/s |
| HC/QKV fisico + singleton fisico, media 3 run | 58,489 s | 17,511 t/s |

Il primo gate adattivo eliminava la regressione del batching fisico forzato
(-21,5%) e conservava la baseline entro lo 0,5%. Il percorso successivo esegue
HC e proiezioni Q/KV una volta sul batch request-major, applica RoPE con la
posizione assoluta di ogni riga e lascia KV, compressor e attention timeline
privati per richiesta. Il verifier CUDA isolato e' passato da `170,371 ms` a
`151,302 ms` (`1,417x -> 1,616x` rispetto alle verifiche sequenziali,
`29,35 -> 33,05` righe/s) con `rel-rmse=0`.

Nel test HTTP automatico i tre run R=2 hanno prodotto `18,24`, `17,24` e
`17,08 t/s`, media `17,52 t/s`: +7,7% rispetto al baseline fisico da
`16,27 t/s`, senza forzare la modalita'. Il run telemetrico separato ha
confermato `17,55 t/s`: prima della conclusione del primo client sono stati
formati 243 cohort R=2 e un solo R=1; gli 87 singleton successivi erano tutti
la coda reale della richiesta piu' lunga. Il confronto R=1 ha misurato
`13,79 t/s` al primo giro legacy, `15,74 t/s` a grafi caldi e `16,99 t/s`
nel physical executor gia' caldo.

Con la stessa build, prompt sintetico e 512 token per client, il canary R=3 ha
prodotto `18,75` e `19,05 t/s` aggregate, media `18,90 t/s`: +7,9% rispetto
al riferimento R=2 caldo da `17,52 t/s` e +16,2% rispetto al vecchio baseline
da `16,27 t/s`. Le singole lane hanno misurato da `6,25` a `8,23 t/s`, come
atteso quando tre richieste condividono la banda della stessa GB10. Un controllo
R=2 sulla build N=8 ha misurato `16,73` e `17,52 t/s`, confermando l'assenza di
regressioni sul profilo predefinito.

#### Drafter DSpark fisico R=n

Il drafter non viene piu' eseguito lane per lane quando una coorte fisica
contiene due o tre richieste. I cinque token non causali vengono appiattiti
request-major e condividono embedding, Q/KV, i tre blocchi Transformer, FFN e
vocabulary head. L'attention usa un solo gather e un solo lancio CUDA, ma legge
ring KV privati; la coda Markov viene proiettata depth-major con W1/W2
batchate, mantenendo sampling, confidence e cronologia separati per lane.

Non e' stata aggiunta memoria permanente: logits e hidden Markov temporanei
riusano l'arena routed-MoE quando quella fase e' terminata. Il percorso `R=1`
resta quello consolidato e non entra nelle nuove funzioni. La regressione
CUDA confronta l'attention fisica `R=3` con tre esecuzioni indipendenti,
includendo ring di lunghezza e wrap diversi; la tolleranza massima e'
`1e-6`.

Su un carico ripetitivo lossless, con tre lane e HybridLC largo 16, il tratto
omogeneo `R=3` ha prodotto:

| Metrica per coorte R=3 | Prima | Drafter fisico | Delta |
| --- | ---: | ---: | ---: |
| DSpark draft | 29,762 ms | 25,628 ms | -13,9% |
| Target verifier | 427,774 ms | 431,203 ms | +0,8% |
| Ciclo completo | 457,536 ms | 456,831 ms | -0,2% |

Gli hash greedy `R=1` e `R=3` coincidono con il baseline. Il throughput
end-to-end resta piu' variabile del tempo per coorte: quando una lane termina
prima, il batch passa a `R=2` e poi `R=1`, quindi il wall time non misura
soltanto il nuovo kernel. Nei tre run ripetuti comparabili la mediana aggregata
e' passata da 59,26 a 60,45 t/s (`+2,0%`), ma ulteriori run hanno mostrato
oscillazioni maggiori quando le lane si disallineano. Il risultato robusto e'
quindi la riduzione del 13,9% del drafter; il verifier target rimane il costo
dominante e limita il beneficio strutturale sull'intero ciclo a circa l'1%.

Questi numeri sono throughput aggregato: una singola GB10 non raddoppia la
banda disponibile, quindi due client non mantengono ciascuno il rate R=1.
Il risultato corretto da osservare e' la somma dei token completati divisa per
il wall time comune. La regressione CUDA copre parita' R=2 e R=3 di RoPE/raw
ring, MXFP4 score + exact Top-K, compressor frontier, attention indicizzata,
Q8 weight-reuse e MoE aligned; non sono stati aggiunti buffer permanenti oltre
allo stato privato delle lane configurate.

Un test operativo lungo con agenti reali ha chiarito perche' lo scaling non e'
lineare. R=2 era realmente attivo, ma la banda dei pesi resta condivisa:
QKV, FFN, output e vocab vengono appiattiti, mentre KV, RNG e frontier restano
private per preservare la distribuzione target. Rejection e commit sono
racchiusi nella stessa transazione R=n, ma applicano ancora per-sessione le
decisioni p/q e il rollback della frontier. Nel run osservato R=1 ha misurato
circa `19,23 t/s`; R=2 e' sceso a `11,40 t/s` per lane ma `22,80 t/s`
aggregati, cioe' circa +18,6% invece di 2x. I batch HybridLC larghi possono
superare i 40 t/s aggregati, ma sono rari nei testi liberi.

Da questa modifica la telemetria distingue `verify_begin`, `verify_reject`
e `verify_finish` nelle righe `dspark timing`. Registra inoltre
`attn_phys`, `attn_local`, `attn_indexed` e `attn_dense`, cosi' si vede quanti
layer di attention passano davvero dal path fisico e quanti ricadono sul
percorso request-local. `analyze-dspark-log.sh` usa ora `hw_r` e `hw_batch`
dalle righe timing, non solo dalle righe scheduler, e stampa rate lane e rate
aggregato della coorte. Il server, con `DS4_TELEMETRY=1`, registra anche la
dimensione del rendezvous (`cohort selected`, lane attive e wait_us).

Il coordinatore lascia decidere ai profili misurati fisico/seriale anche per i
microbatch:

```bash
DS4_DSPARK_RN_MIN_PHYSICAL_ROWS=0
```

Il valore `0` e' il default di produzione. I test GB10 mostrano che le forme
R=2/R=3 piu' comuni occupano appena 5-9 righe ma restano piu' veloci nel
percorso fisico; la precedente soglia conservativa `10` le serializzava quasi
tutte. Un valore maggiore di zero resta disponibile come guardia diagnostica,
mentre `DS4_DSPARK_COORDINATOR_MODE=physical` forza il percorso fisico.

La rimozione del gate statico e il rendezvous phase-aware sono stati validati
con lo stesso carico sintetico controllato. Il timeout raccoglie ora tutte le
lane gia' pronte, anziche' rilasciare soltanto quella che lo ha raggiunto; una
lane attiva ma uscita dalla coorte da oltre 5 ms usa il rendezvous breve.

| Concorrenza | Prima | Coordinator aggiornato | Delta |
| --- | ---: | ---: | ---: |
| R=2 aggregato | 14,94 t/s | 16,54 t/s | +10,7% |
| R=3 aggregato | 16,40 t/s | 18,92 t/s | +15,4% |

In un secondo run phase-aware, 705 delle 765 coorti sono state fisiche. Il
rendezvous medio e' stato 0,379 ms, senza attese da 10 ms o piu'. R=2 fisico
ha misurato 16,80 t/s contro 14,52 t/s seriali; R=3 fisico 19,59 t/s. Il
profilo del verifier attribuisce in media 167,20 ms a `verify_begin`, 0,44 ms a
rejection e 1,19 ms a commit. Sono gia' fisici 40,61 layer attention per ciclo,
contro 2,39 request-local: fondere ulteriormente rejection/commit avrebbe un
tetto inferiore all'1% e aumenterebbe il rischio sulla semantica lossless.

Il successivo test operativo con tre agenti e telemetria completa ha formato
3.717 coorti, 2.727 delle quali fisiche: la copertura sale quindi dal precedente
40,5% al 73,4%. Il throughput aggregato delle coorti e' passato da 24,383 a
28,294 t/s (`+16,0%`). R=3 fisico ha misurato 33,351 t/s contro 23,595 t/s
seriali (`+41,3%`), mentre R=2 fisico ha misurato 24,150 t/s contro 21,904 t/s.
In media 29,20 dei 43 layer attention per ciclo sono stati eseguiti dal percorso
fisico. Il prefill non e' regredito: un prompt da 23.335 token ha mantenuto
1.001,38 t/s.

La stessa misura aveva inizialmente attribuito 4,821 t/s alle 2.310 lane K=1
dei 7.675 cicli neurali. L'audit successivo ha chiarito che questo non era il
throughput delle coorti K=1: ogni riga per-lane riportava l'intero tempo della
coorte R2/R3, mentre altre lane dello stesso batch potevano usare K2/K3. Nei
run controllati, per esempio, 78 lane R3 erano K1 ma soltanto due coorti
avevano budget aggregato B=1. L'analizzatore distingue ora esplicitamente il K
della lane dal budget B e dal throughput aggregato della coorte. Il target
verifier resta invece un limite reale: occupava in media 251,54 ms dei 277,54
ms del ciclo.

Il default e':

```bash
DS4_SERVER_DSPARK_LANES=3
DS4_SERVER_DSPARK_HOT_LANES=2
DS4_SERVER_DSPARK_COALESCE_US=500
DS4_DSPARK_RN_MIN_PHYSICAL_ROWS=0
```

`DS4_SERVER_DSPARK_LANES=1` ripristina il worker seriale consolidato. La seconda
lane alloca il proprio contesto ma non duplica i pesi; nel canary a 262K ha
richiesto circa 2,25-2,37 GiB di stato privato. Il terzo slot e' configurato per
default ma resta elastico; `R=4` resta escluso perche' non lascia un margine UMA
sicuro sul GB10 da 128 GiB.

Le richieste oltre la capacita' residente non causano nuove allocazioni CUDA.
Un canary con quattro client simultanei ha registrato `resident=3`,
`assigned=4` e `admission=bounded-queue`: il quarto lavoro e' rimasto sulla
coda di una lane esistente. Ha completato 252 token a 12,18 t/s aggregate; e'
una verifica di sicurezza della memoria, non una modalita' di scaling. Sopra
tre client residenti il comportamento promosso e' quindi coda bounded e
servizio seriale sulle lane disponibili.

#### Stato operativo e limiti della release

La configurazione promossa parte con due lane residenti e una capacita' massima
di tre. La terza lane viene materializzata soltanto quando entrambe le lane
calde hanno gia' lavoro assegnato e `MemAvailable` puo' contenere il suo stato
privato piu' una riserva esplicita. Sul profilo misurato richiede
`2368,66 MiB`; il launcher conserva per default altri `1536 MiB` e non usa lo
swap Linux come tier di capacita' CUDA. `R=4` non e' considerato sicuro con il
margine UMA osservato su Athena. Pesi, cache Q8->F16 e arena transiente sono
condivisi, mentre ogni lane conserva il proprio stato di conversazione.

```bash
DS4_SERVER_DSPARK_LANES=3
DS4_SERVER_DSPARK_HOT_LANES=2
DS4_SERVER_DSPARK_LANE_RESERVE_MB=1536
```

La prima prova operativa a tre agenti non era in realta' un test `R=3`: con
l'arena prefill precedente rimanevano circa 3,25 GiB disponibili, contro
3,90 GiB necessari per una lane da 2368,66 MiB piu' la riserva. Il coordinatore
ha quindi mantenuto correttamente `resident=2/3` e servito la terza
conversazione attraverso il tier KV su disco.

La correzione non riduce contesto o cache Q8->F16. Gli score indexer, gli
ausili attention e i buffer routed-MoE hanno lifetime disgiunti e ora occupano
una sola arena CUDA: a 262K/chunk 8192 l'arena misura 2850,06 MiB e sostituisce
3968,06 MiB di allocazioni separate, recuperando 1118 MiB. I buffer di contesto
riportati all'avvio scendono da 4214,65 a 3096,65 MiB. Con cache Q8->F16 ancora
a 12 GiB, la terza frontier full-context e' stata attivata come
`resident=3/3`; dopo l'allocazione restavano 2,70 GiB disponibili e swap zero.
Un canary simultaneo da tre richieste e 256 token ha completato 768 token in
19,455 secondi, pari a `39,48 t/s` aggregati.

Nel test operativo finale con tre agenti la lane elastica e' stata attivata una
sola volta e il server ha completato 7.123 cicli DSpark senza fallback
ordinari, errori CUDA, OOM o swap del processo. Le 3.622 coorti comprendevano
1.467 esecuzioni fisiche e 2.155 seriali: il throughput aggregato complessivo
e' stato `24,383 t/s`. Le coorti `R=3` fisiche hanno raggiunto
`35,391 t/s`, contro `23,154 t/s` per la corrispondente forma seriale
(`+52,8%`); `R=2` fisico ha misurato `21,432 t/s`, contro `19,043 t/s`
seriali (`+12,6%`). L'acceptance complessiva e' stata `66,62%`.

La quota fisica e' quindi ancora circa il 40,5%. In media 13,70 dei 43 layer
attention per ciclo sono passati dal percorso fisico e 29,30 sono rimasti
locali: e' il principale margine ancora aperto per aumentare lo scaling
aggregato senza modificare la distribuzione target.

Il batching fisico copre QKV, FFN, output, vocabulary head, compressor/indexer
e quasi tutte le forme CSA/HCA. Ogni lane mantiene KV, RNG e frontier proprie;
rejection sampling e commit consumano viste del batch fisico ma applicano
per-sessione decisioni e rollback. Questa separazione e' necessaria per
mantenere esatte posizione, RNG e distribuzione target, ma spiega perche' R=2
aumenta il throughput aggregato senza raddoppiarlo.

Le coorti si formano soltanto quando due richieste raggiungono insieme un
confine speculativo. Se una lane esegue una tool call o un prefill, oppure una
richiesta termina prima, l'altra continua correttamente come `R=1`; non viene
trattenuta artificialmente per mantenere pieno il batch.

All'avvio sono calde due frontier KV indipendenti; dopo l'attivazione elastica
la terza lane conserva la propria frontier finche' il server resta in
esecuzione. Le conversazioni oltre le lane residenti usano checkpoint su disco
e replay; una
canonicalizzazione che cambia la coda puo' ancora produrre `token-mismatch` e
richiedere il ripristino del miglior checkpoint compatibile. Il disco e' il
tier freddo gestito dall'applicazione: lo swap Linux non viene considerato un
tier affidabile per lo stato CUDA attivo, perche' introdurrebbe stall e rischio
OOM non prevedibili.

Infine, prefill e decode vengono interleaved soltanto ai confini completi dei
chunk da 8192 token. Una preemption fra layer non e' ancora corretta con gli
scratch condivisi; il kernel prefill R=1 resta quindi invariato e il verifier
ottiene la GPU solo quando KV e checkpoint della lane sono consistenti.

### Routing KV cost-aware e scheduling prefill/verifier

Il dispatcher multi-lane non usa piu' il solo conteggio dei job. Le
continuation legate a un `call_id` restano vincolate alla sessione che possiede
la frontier corretta; per tutte le altre richieste viene minimizzato:

```text
costo_lane = attesa_stimata + token_da_riprodurre / prefill_tps_misurato
token_da_riprodurre = prompt_tokens - prefisso_KV_residente
```

L'attesa comprende i job in coda e il residuo della richiesta attiva. Il
residuo decresce con il tempo gia' trascorso, ma conserva un pavimento pari a
meta' dell'intervallo di servizio EWMA: una risposta reasoning lunga non puo'
quindi apparire improvvisamente gratuita soltanto perche' ha superato la durata
media. La velocita' di replay viene aggiornata dai chunk reali; i piccoli tail
finali hanno peso proporzionale e non deformano la stima quanto un chunk pieno.
Anche il primo campione viene fuso con il bootstrap conservativo da 900 t/s:
un fringe di poche centinaia di token, dominato dalla latenza di lancio, non
puo' sostituire da solo l'intera stima.
I pareggi restano deterministici: costo, carico, prefisso piu' lungo e infine
indice della lane. Con `DS4_TELEMETRY=1`, `coordinator dispatch` espone
prefisso, replay, attesa, costo di replay, costo totale ed entrambe le EWMA.
E' lo stesso principio del
[KV-aware cost model di NVIDIA Dynamo](https://docs.nvidia.com/dynamo/latest/components/router/routing-concepts):
localita' della cache e lavoro attivo vengono valutati insieme, invece di
massimizzare il solo cache hit.

Il secondo intervento porta nel coordinatore il principio di chunked prefill di
[SARATHI](https://arxiv.org/abs/2308.16369), conservando il chunk MoE da
8192 che ha dato il throughput migliore sul GB10. Al termine di un chunk
intermedio completo la KV e il checkpoint della sessione sono validi; se
esiste un verifier in attesa, il prefill:

1. cede la GPU a un solo ciclo decode/verifier;
2. conserva inalterata la KV privata della sessione;
3. riprende prima di un altro prefill o di un secondo verifier;
4. ripete l'handoff soltanto al successivo checkpoint di chunk.

I callback iniziale, senza avanzamento, e finale non cedono la GPU: il primo
non ha ancora prodotto lavoro riutilizzabile, mentre dopo il secondo il prefill
sta gia' per rilasciare normalmente l'executor. Questo mantiene gratuiti i
prompt che non attraversano una frontier interna.

Senza decode concorrenti non avviene alcun handoff GPU; con una sola lane il
controllo ritorna prima di acquisire il mutex. Non vengono aggiunti buffer CUDA,
copie KV, logits o approssimazioni: Top-K, attention, RNG, rejection p/q e
distribuzione del target restano identici.

E' stata esclusa deliberatamente la preemption a ogni layer. Le lane R=n
mantengono private KV e frontiere, ma condividono gli scratch transitori
`batch_cur_hc` e `batch_next_hc`; un verifier fra due layer potrebbe quindi
sovrascrivere l'attivazione del prefill. La tecnica di
[Layered Prefill](https://arxiv.org/abs/2510.08055), specifica per MoE, resta
interessante ma richiede un'arena privata o un executor fisico unico prima di
essere corretta in questa architettura.

Le proposte piu' recenti del 2026 rafforzano la scelta di un costo misurato
anziche' di un budget statico. Il lavoro
[Fairness-Aware and Latency-Controllable Scheduling](https://arxiv.org/abs/2606.09061)
usa tempo di attesa, lavoro rimanente e previsione della latenza; il successivo
[Load-Aware Prefill Deflection](https://arxiv.org/abs/2607.02043) cerca il
massimo lavoro prefill che non violi il TBT dei decode attivi. Entrambi sono
studiati soprattutto per serving disaggregato o per acceleratori con piu'
capacita'. Su una singola GB10 non conviene introdurre trasferimenti KV o pool
prefill/decode separati: adottiamo il criterio cost/SLO, ma manteniamo dati e
calcolo sul device locale.

Questa prima integrazione e' un interleaving temporale ai confini dei chunk, non
ancora un singolo GEMM fisico contenente righe prefill e verifier. Fondere anche
le operazioni dense richiedera' di mantenere attention e KV request-local e di
calcolare il vocabulary head soltanto sulle righe verifier e sull'ultima riga
prefill. Un eventuale kernel attention misto dovra' seguire il principio di
[POD-Attention](https://arxiv.org/abs/2410.18038), adattato a CSA/HCA, e non
puo' essere sostituito da un normale kernel MHA.

Il test host copre due verifier contemporaneamente in attesa: ogni handoff ne
esegue esattamente uno e restituisce la proprieta' della GPU al prefill. La
promozione su Athena e' stata subordinata a tre gate:

- prefill R=1 senza contesa entro l'1% del baseline;
- due richieste reali con assenza di stalli multi-secondo e throughput
  aggregato non inferiore al coordinatore precedente;
- hash greedy, acceptance DSpark, RAM e swap invariati.

La telemetria di un run concorrente riassume gli handoff a ogni chunk:

```text
ds4-server: cooperative chunk scheduler lane=... chunk=... handoffs=... waited=...
```

### Validazione finale del coordinatore - 29 luglio 2026

Il primo test misto del routing cost-aware ha individuato due costi puramente
di scheduling: 32 handoff su 42 avvenivano prima di produrre una frontier
intermedia oppure dopo l'ultimo chunk, accumulando 6,82 secondi di attesa; un
primo prompt di 167 token aveva inoltre sostituito il bootstrap da 900 t/s con
una stima di 154 t/s. Il commit `e2036ae` limita quindi l'handoff ai soli
checkpoint interni produttivi e pesa ogni campione prefill in proporzione a
`interval_tokens / 8192`. Il prefill senza attesa e' rimasto circa 933 t/s,
mentre nel run corretto i prompt da almeno 1K hanno misurato 904,63 t/s e gli
handoff sono scesi a 18.

I profili per forma richiedono normalmente due campioni prima di diventare
maturi. Questa regola poteva pero' bloccare una forma R=2 promettente dopo il
primo campione: il gate per i piccoli batch la rimandava al seriale e il secondo
campione non arrivava mai. Il commit `09fa4c3` aggiunge un solo
`confirm-probe` quando:

- la coorte contiene almeno due richieste;
- esiste esattamente un campione della stessa forma fisica;
- quel campione supera la previsione seriale di almeno il 10%.

Il secondo campione riporta immediatamente la forma alle normali regole di
maturita'. R=1 e' escluso dal meccanismo. La tabella host dei profili passa da
128 a 512 slot, circa 18 KiB per sessione; non vengono aggiunti buffer CUDA,
copie KV o memoria proporzionale al contesto.

Nel test operativo successivo, con sidecar Q2, due lane, contesto fisico 262K
e lo stesso launcher di produzione:

| Metrica | Coordinatore precedente | Con `confirm-probe` | Delta |
| --- | ---: | ---: | ---: |
| Throughput aggregato delle coorti | 23,101 t/s | 24,884 t/s | +7,7% |
| Decode pesato per richiesta | 12,048 t/s | 13,578 t/s | +12,3% |
| Prefill prompt >=1K | 904,632 t/s | 915,831 t/s | +1,2% |
| R=2 fisico, segmenti attivi | 27,842 t/s | 37,947 t/s | dipende dalla forma |

Nel nuovo run 18 conferme hanno sbloccato 504 decisioni fisiche R=2 previste.
All'interno dello stesso carico, i segmenti R=2 fisici hanno misurato
37,947 t/s aggregati e acceptance del 73,03%, contro 21,131 t/s e 67,18% del
seriale. Il dato da usare per la prestazione end-to-end resta 24,884 t/s:
37,947 t/s descrive soltanto i segmenti in cui entrambe le richieste sono
effettivamente riunite nel verifier fisico.

L'acceptance globale e' stata 66,77% contro 71,20% nel run precedente, ma i due
carichi non contenevano lo stesso numero e la stessa composizione di risposte;
non e' quindi un confronto di qualita'. Il target continua a verificare ogni
token con la stessa regola lossless. R=1 non usa il nuovo probe e la coda
operativa finale ha mantenuto 20,007 t/s. La macchina e' rimasta a circa
117 GiB usati, 3,8 GiB disponibili e `VmSwap=0` per il processo.

La KV append e' rimasta attiva: le continuation coerenti hanno processato
soltanto la coda. Tre cambi di ramo hanno prodotto `token-mismatch`; quando era
disponibile un checkpoint canonico, il turno finale ha ripreso da 14.407 token
e ha eseguito il prefill soltanto fino a 19.760, invece di ripartire da zero.
Regressione CUDA `sm_121a`, test scheduler, test server e `git diff --check`
sono passati.

### Curva hardware persistente per forma - 30 luglio 2026

La curva `SPS(B)` per sole righe non descrive completamente il verifier
multi-sessione: a parita' di `B`, forme come `[1,3]`, `[2,2]` e `[0,4]`
possono avere costi diversi per padding, riga ombra `K=0`, attention privata e
distribuzione del lavoro fra lane. Il planner production consulta ora anche un
profilo esatto indicizzato da:

```text
executor, neural/HybridLC, bucket contesto, R, righe fisiche, forma K ordinata
```

La chiave conserva l'associazione fra bucket di contesto e `K` della singola
lane, ordinando coppie `(bucket,K)` invece dei soli `K`: una lane lunga con
`K=1` non viene quindi confusa con una lane corta con `K=1`, ne' con la forma
ottenuta scambiando i due prefissi fra contesti diversi.

Le misure esatte possono guidare `K`, ma soltanto quando coprono tutte le
forme visitate dalla ricerca e ogni forma dispone di almeno otto campioni.
Il planner esegue quindi un preflight senza modificare la storia causale. Se
la curva fisica e' incompleta, l'intero candidato fisico viene ricalcolato
con la curva rows-only offline o generica; non essendo scomponibile per lane,
non viene mai stimata una sua singola forma mancante.

Il candidato seriale ha invece un modello esatto naturale: la somma dei costi
robusti misurati per ciascuna lane e per il suo `K`. Se tutte le forme online
sono presenti usa il profilo esatto; se la prima passata mescola profilo e
modello, la seconda disabilita il profilo e usa il modello per tutte le forme.
Se anche questo replay non copre ogni forma, un ultimo replay disabilita
l'intera curva shape e usa la curva rows-only. Una curva offline completa
resta il fallback hardware misurato; solo un profilo shape completo e maturo
puo' sostituirla. In nessun caso una stessa ottimizzazione confronta righe
provenienti da sorgenti diverse.

Entrambi i candidati vengono valutati su copie dello stesso stato `t-2`; solo
dopo il confronto viene avanzata una volta la storia condivisa. Il preflight
non vede acceptance o token correnti e non altera RNG, logits, KV o rejection
sampling.

Le misure neurali e HybridLC sono separate. Il profilo registra la forma
realmente verificata dopo l'eventuale estensione retrieval, non quella
originariamente pianificata; in questo modo un verifier `N=8/12/16` non
contamina il costo del prefisso neurale `K=0..5`.

Il launcher salva il profilo in:

```text
$DS4_MODEL_DIR/dspark-hardware-q2.profile
$DS4_MODEL_DIR/dspark-hardware-q4.profile
```

Il percorso si puo' personalizzare con `DS4_DSPARK_HW_PROFILE`. Il file non
contiene prompt, token, KV, probabilita' o output: soltanto EWMA, numero di
campioni e chiavi hardware. Un fingerprint versionato include binario, modello
target, sidecar, backend e power target; se uno cambia, il vecchio profilo
viene ignorato automaticamente. Il salvataggio avviene atomicamente durante
lo shutdown pulito.

La prima validazione GB10 del formato V1 aveva raccolto 51 forme. Prima del riavvio la curva
esatta ha coperto 7.593 valutazioni su 10.372 (`73,21%`); al primo run dopo il
riavvio il profilo e' stato caricato con lo stesso fingerprint e ha coperto
1.484 valutazioni su 1.571 (`94,46%`). Questi numeri verificano persistenza e
warm-start, non costituiscono ancora un claim di incremento end-to-end:
l'effetto va misurato sullo stesso carico operativo e contro il fallback
rows-only.

Il successivo test operativo ha mostrato una regressione da 28,557 a
26,619 t/s aggregati. L'audit ha individuato due cause nella chiave V1:
associazione lane-contesto persa dall'ordinamento dei soli `K` e forme abilitate
dopo appena due campioni. Il formato V2 corregge entrambe; il fingerprint
versionato invalida automaticamente il file V1. I numeri V1 restano qui come
traccia diagnostica e non rappresentano uno stato promosso.

I numeri V1 restano leggibili come traccia storica. Nel formato V2,
`shape_curve=hit/query` e i contatori separati physical/serial descrivono il
preflight; `sps_curve`, `physical_curve` e `serial_curve` indicano invece le
curve coerenti effettivamente usate dal planner.

### SPS(B) offline autorevole - 30 luglio 2026

L'implementazione production segue ora il contratto del paper: una curva
hardware viene profilata fuori dal percorso di inferenza e caricata in sola
lettura. La chiave e':

```text
executor, neural/HybridLC, bucket contesto 32K, R, batch logico B
```

`batch` e `rows` sono entrambi presenti nella telemetria. `batch` e' la
dimensione logica usata da Algorithm 1; `rows` e' il lavoro fisico reale e puo'
essere maggiore quando una lane `K=0` richiede la shadow row del verifier.

Un gruppo e' attivabile soltanto se contiene tutte le misure
`B=R..R*(Kmax+1)`. Il loader valida versione, fingerprint, coordinate,
duplicati, tempi finiti e completezza. Se manca anche una sola misura, il
gruppo non viene esposto e l'intera decisione usa la curva generica. Non
esiste fallback per singola riga.

Il fingerprint SPS include modello target, sidecar, backend, power target e
una versione esplicita dell'ABI kernel. Non include il timestamp
dell'eseguibile: una ricompilazione identica non invalida la calibrazione,
mentre una modifica del percorso fisico deve incrementare l'ABI.

Il launcher cerca per default:

```text
$DS4_MODEL_DIR/dspark-sps-q2.conf
$DS4_MODEL_DIR/dspark-sps-q4.conf
```

Se il profilo non e' presente nella model directory, il launcher cerca il
corrispondente artefatto versionato in `profiles/`. Un percorso esplicito usa
`DS4_DSPARK_SPS_PROFILE=/path/file.conf`; il fingerprint impedisce di applicare
la curva a una combinazione target/sidecar/backend incompatibile.

Per raccogliere tutte le batch `B` per `R=1..3`, confrontare fisico e seriale
e calibrare nello stesso avvio:

```bash
DS4_DSPARK_SPS_PREFIX=98304 ./profile-dspark-sps.sh
```

La smoke usa lo stesso verifier production, confronta numericamente il
microbatch fisico con le lane seriali e ripristina la frontier fra le prove.
Per ogni `B` enumera tutte le partizioni non equivalenti delle righe tra le
lane, per esempio `[1,5]`, `[2,4]` e `[3,3]`: una sola forma bilanciata
produrrebbe una curva troppo ottimistica. Ogni partizione viene scaldata
separatamente per escludere l'eventuale istanziazione della CUDA Graph e deve
superare la parita' numerica col verifier seriale prima di essere misurata
dieci volte. Il calibratore applica il filtro MAD e calcola la mediana dentro
ciascuna forma; la riga `SPS(B)` usa poi il costo peggiore tra le mediane delle
forme complete. Il profilo rimane quindi conservativo anche quando il planner
incontra una partizione sfavorevole. `DS4_DSPARK_SPS_MAX_R=2` limita una
raccolta diagnostica a `R=1..2`; il default resta `R=1..3`.

La calibrazione GB10 Q2 a 98.304 token ha inoltre esposto una scogliera reale:
con `R=2` il verifier fisico resta esatto e vale circa `1,5-1,6x` fino a
cinque righe complessive; a partire da sei righe la diversa geometria delle
riduzioni modifica leggermente attention e il MoE quantizzato amplifica lo
scarto. Il runtime non accetta questa approssimazione: per `N>=6` esegue le
lane complete separatamente e ricompone i logits nella stessa arena fisica,
ottenendo `rel-RMSE=0` rispetto al verifier indipendente. La curva registra
anche questo costo quasi seriale, consentendo allo scheduler di preferire la
frontiera fisica veloce senza sacrificare la distribuzione target. La smoke
iniziale ha prodotto 34 record e quattro gruppi completi (fisico e seriale,
`R=1..2`), con parita' finale esatta e speedup `1,496x` sulla forma R=2 di
controllo. La calibrazione Q2 completa su GB10 ha poi raccolto `7.120`
campioni, generando `1.248` record e `128` gruppi completi: `R=1` copre tutti i
bucket fino a circa 1M token, mentre `R=2..3` coprono i bucket fino a 512K per
rispettare il limite di memoria della raccolta multi-lane. Il profilo promosso
e' `profiles/dspark-sps-q2.conf`, fingerprint `8751cba0622dd9a3`. Questa
modifica incrementa l'ABI SPS: un server precedente non puo' caricare
accidentalmente la curva misurata sul percorso lane-partitioned.

Lo script dedicato non scarta altri campioni perche' il warm-up e' gia'
esplicito e richiede almeno otto misure residue per ogni partizione. Per
calibrare un altro bucket 32K basta cambiare `DS4_DSPARK_SPS_PREFIX`. Piu' log
dedicati possono essere uniti passando ripetutamente `--log` a
`tools/calibrate_dspark_sps.py`. I normali log operativi non hanno copertura
garantita delle forme e vengono accettati soltanto con
`--allow-operational`.

Un file parziale puo' essere scritto con `--allow-no-complete-group` per
analisi, ma il runtime continuera' a ignorarne i gruppi incompleti.

Il runtime costruisce due candidati dallo stesso stato causale `t-2`: il
candidato fisico usa la curva `executor=physical`, mentre quello
lane-partitioned usa `executor=serial`. Solo dopo aver ottimizzato
indipendentemente i due prefissi confronta il throughput previsto e avanza la
storia dello scheduler una volta sola. Quando entrambe le curve offline sono
complete non servono warm-up o probe dell'esecutore. Nei bucket non calibrati
il gate con replay coerente ammette un profilo esatto soltanto con copertura
totale; altrimenti usa per tutto il candidato la curva generica fisica o il
modello robusto per-lane seriale. Se anche il replay model-only seriale e'
incompleto, l'intero candidato torna rows-only.

La scelta dell'executor richiede invece che il candidato fisico superi quello
seriale dell'1% con due curve offline accoppiate, del 5% con due sorgenti
mature e del 15% durante il cold start. Il modello per-lane seriale e' piu'
affidabile di una curva fisica generica ancora fredda; il margine evita quindi
di inseguire una previsione fisica ottimistica. Restano probe fisici radi per
raccogliere evidenza e non cristallizzare la decisione.

La riga `dspark admission` espone la curva dell'esecutore scelto tramite
`sps_curve=offline|generic|shape|lane-model` e aggiunge
`physical_curve=... serial_curve=...`; il report mostra:

```text
Offline SPS curves: ... cohorts (...%), generic=...
  physical candidate: offline=... generic=...
  serial candidate:   offline=... generic=...
Shape-aware SPS selections: shape=... lane-model=...
  exact/model curves: physical-shape=... serial-shape=... serial-lane-model=...
```

La smoke finale del gate ha elaborato 64 coorti con acceptance `99,45%`: la
copertura grezza delle forme era soltanto `39,89%`, quindi nessuna curva
esatta parziale e' stata ammessa. Il candidato fisico ha usato integralmente
il fallback rows-only e quello seriale integralmente il modello per-lane.
Nel successivo controllo warm con tre richieste identiche, i tre output sono
risultati esatti, l'acceptance e' stata `99,79%` e il throughput aggregato
`31,58 t/s`. Due probe fisici R=3 hanno misurato `8,48 t/s`; il coordinatore
e' quindi passato al seriale R=3 per 34 coorti, misurando `33,30 t/s`. Il
modello seriale aveva previsto circa `34,4 t/s`: il margine conservativo ha
impedito alla curva fisica generica, ancora ottimistica, di dominare la
selezione. Sono smoke controllate, non un claim sul carico operativo.

Una futura curva adattiva di acceptance dovra' essere gerarchica,
regolarizzata verso il profilo STS e aggiornata piu' lentamente; non viene
mescolata a questi costi fisici.

### STS offline autentico

Il paper richiede temperature immutabili calibrate su un held-out set, da
sinistra a destra. Il runtime supporta ora un artefatto versionato
`DS4_DSPARK_STS_V1`. Quando e' caricato, la confidence del planner e'
esattamente `sigmoid(logit/T[k])`: non vengono applicati bias, prior, EWMA o
aggiornamenti online. La trasformazione resta order-preserving e modifica solo
la lunghezza proposta al verifier; target logits, rejection sampling, RNG, KV e
qualita' restano invariati.

Raccolta held-out, in un run dedicato:

```bash
DS4_DSPARK_STS_CAPTURE=/tmp/dspark-sts-heldout.csv \
DS4_TELEMETRY=1 ./run-dspark-server.sh
```

La capture forza la verifica del blocco neural completo `K=5` soltanto nel run
di calibrazione, ignora gate e circuit breaker del serving e registra una riga
solo quando sono disponibili tutte le cinque label cumulative tramite
rejection sampling lossless (`temperature > 0`, `top_k=0`, `top_p=1`). Le altre
policy non contaminano il CSV. Non va usata come modalita' benchmark. Terminato
il corpus held-out, con almeno 512 cicli validi:

```bash
python3 tools/calibrate_dspark_sts.py \
  --input /tmp/dspark-sts-heldout-grouped.csv \
  --output /tmp/dspark-sts-q2-candidate.conf \
  --report /tmp/dspark-sts-q2-validation.json
```

Il calibratore esegue cinque grid search 1D consecutive. A posizione `k`
minimizza l'ECE del prodotto cumulativo, tenendo fisse le temperature gia'
scelte per `1..k-1`, come nella sezione 3.2.1 del paper. Il launcher cerca prima
`dspark-sts-q2.conf` o `dspark-sts-q4.conf` nella model directory e poi il
profilo versionato in `profiles/`; un percorso esplicito usa
`DS4_DSPARK_STS_PROFILE=/path/file.conf`.
Un artefatto mancante o malformato interrompe l'avvio invece di ricadere
silenziosamente su una calibrazione diversa.

Per evitare leakage, il calibratore raggruppa tutte le varianti della stessa
famiglia di prompt ed esegue una cross-validation deterministica a cinque fold:
una famiglia non compare mai sia nel fit sia nella sua valutazione. Il gate di
promozione usa esclusivamente l'ECE out-of-fold; solo dopo il superamento viene
eseguito il fit finale su tutto il corpus per produrre l'artefatto immutabile.
Per CSV non raggruppati rimane disponibile il vecchio split 80/20. Il corpus
riproducibile incluso si ferma sul numero di cicli K=5 realmente catturati:

```bash
python3 tools/capture_dspark_sts.py \
  --capture /tmp/dspark-sts-heldout.csv \
  --grouped-capture /tmp/dspark-sts-heldout-grouped.csv \
  --target-samples 2500
```

Il profilo Q2 incluso e' stato promosso su `2.535` cicli lossless, `26` prompt e
`16` famiglie. La ECE cumulativa media out-of-fold e' passata da `0,03867` a
`0,03477` (`-10,1%`), con temperature
`[0,98219, 0,98809, 0,67947, 0,99105, 0,75007]`. Questo risultato misura la
calibrazione della confidence, non un guadagno t/s: il throughput va ancora
validato end-to-end sul carico operativo.

Con `DS4_TELEMETRY=1` il launcher abilita automaticamente lo shadow:

```bash
DS4_TELEMETRY=1 ./run-dspark-server.sh
```

Lo shadow calcola la decisione hardware-aware e la registra, ma lascia il K
effettivo allo scheduler stabile. Il canary applicativo si abilita esplicitamente:

```bash
DS4_TELEMETRY=1 DS4_DSPARK_SCHEDULER_DETERMINISTIC=1 ./run-dspark-server.sh
```

Il canary entra soltanto dopo almeno tre campioni robusti per tutte le forme.
Verifier target, logits, rejection sampling p/q, RNG, KV e HybridLC non vengono
modificati. La telemetria confronta `hw_sync_k` con la decisione production,
registra `hw_async`, `hw_capacity`, `hw_age`, `hw_r`, `hw_batch` e i causal
stop. `analyze-dspark-log.sh` aggrega coorte, batch, capacita' e quante volte la
barriera asincrona supera la scelta causale. Il test host copre `R=1`,
allocazione globale `R=2`, `K=0`, tie deterministici, causal stop e una curva
hardware frastagliata che la ricerca storica riesce ad attraversare. Copre
inoltre join/leave, riordino della coorte, gap temporali, clamp ai confini
EOS/context e layout flattened con marker e posizioni assolute.

## Sidecar DSpark Q2 compatto promosso - 24 luglio 2026

Il launcher e il builder supportano ora due sidecar DSpark. Il Q2 e' il
default `memory-first`; il Q4 resta disponibile come profilo conservativo:

| Variante | Routed gate/up | Routed down | Dimensione | Uso consigliato |
| --- | --- | --- | ---: | --- |
| `q2` | IQ2_XXS | Q2_K | circa 5,64 GiB | default, recupero di circa 5,06 GiB UMA |
| `q4` | Q4_K | Q4_K | circa 10,70 GiB | confronto conservativo opzionale |

Il Q2 conserva Q8 per attention, shared expert e proiezioni dense. Se non viene
fornita una imatrix DSpark, il quantizer usa la sua importance sintetica
deterministica basata sull'energia delle colonne. Il target verifier non cambia
e continua a decidere i token finali con rejection sampling lossless.

Nel test pulito su Athena, senza subagent, il Q2 ha completato 15 richieste e
generato 7866 token a 20,88 t/s ponderati. Le tool call hanno misurato
24-26 t/s; la risposta finale di 3311 token ha misurato 17,40 t/s. Il prefill
append e' rimasto fra circa 800 e 900 t/s, il draft DSpark medio a 15,28 ms e
l'acceptance cumulativa al 63,98%. La risposta lunga e' risultata coerente e
completa nel controllo manuale. Il vantaggio di decode non e' universale:
l'acceptance dipende dal testo e puo' assorbire il drafter piu' veloce. Il
beneficio ripetibile della variante e' il margine di memoria.

Costruzione:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_VARIANT=q4 ./build-dspark-sidecar.sh
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_VARIANT=q2 ./build-dspark-sidecar.sh
```

La costruzione Q2 va eseguita su Athena. Richiede in
`/home/athena/ds4/dspark-v4flash-hf` l'indice Safetensors e i tre shard che
contengono il drafter DSpark:

```text
model.safetensors.index.json
model-00046-of-00048.safetensors
model-00047-of-00048.safetensors
model-00048-of-00048.safetensors
```

Il comando completo compila il quantizer nativo e scrive direttamente il
sidecar promosso nella directory modelli di Athena:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_VARIANT=q2 DS4_DSPARK_HF_DIR=/home/athena/ds4/dspark-v4flash-hf DS4_DSPARK_GGUF=/home/athena/ds4/DeepSeek-V4-Flash-DSpark-IQ2XXS-Q2K-Q8.gguf ./build-dspark-sidecar.sh 2>&1 | tee /tmp/ds4-dspark-q2-convert.log
```

Il builder conserva Q8 per i tensori densi e applica `IQ2_XXS` a routed
gate/up e `Q2_K` a routed down. La Q2 validata usa l'importance sintetica
deterministica del quantizer; non richiede una imatrix esterna. Al termine:

```bash
ls -lh /home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf /home/athena/ds4/DeepSeek-V4-Flash-DSpark-IQ2XXS-Q2K-Q8.gguf
```

Avvio:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./run-dspark-server.sh
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_VARIANT=q4 ./run-dspark-server.sh
```

`DS4_DSPARK_MODEL=/percorso/custom.gguf` continua ad avere priorita' sul
percorso derivato dalla variante.

## Riuso MQA completo nella sparse attention scartato - 24 luglio 2026

Il principio DSA descritto nel paper DeepSeek-V3.2 e' stato applicato al kernel
indexed: una sola CTA per token avrebbe condiviso ogni riga Top-512/KV fra tutti
i 64 head, invece delle due CTA da 32 head del percorso stabile M32/R32. Sono
state implementate e misurate tre organizzazioni complete:

| Variante | Registri | Stack/thread | Tempo kernel | Velocita' vs M32/R32 |
| --- | ---: | ---: | ---: | ---: |
| M32/R32 stabile | 128 | 16 B | 6,58-6,96 ms | 1,000x |
| M64/R16, 768 thread | 80 | 112 B | 11,60 ms | 0,600x |
| M64/R32, 512 thread | 128 | 0 B | 7,01 ms | 0,939x |
| M64/R16, 1024 thread | 64 | 232 B | 13,01 ms | 0,532x |

Tutte le varianti mantenevano Top-512, maschere, online softmax e KV accessibili.
La regressione numerica e' rimasta allo stesso livello del percorso stabile
(`rel-rmse` circa 0,00067 e `max-abs` circa 0,000002).

Il riuso M64 elimina la seconda lettura globale delle KV, ma sul GB10 quella
lettura trova gia' circa il 90% dei dati in L2. M32/R32 compensa meglio la
duplicazione sovrapponendo QK del tile corrente e PV del precedente. M64/R32
deve serializzare le due fasi; M64/R16 raddoppia stage e barriere e, con 768 o
1024 thread, forza frammenti nello stack. Il traffico risparmiato non compensa
questi costi. Tutto il codice M64 e il microbenchmark sono stati rimossi; resta
il kernel M32/R32 validato.

## Fusioni compressor/KV e overlap verifier scartati - 24 luglio 2026

Sono state integrate e misurate tre tecniche ispirate alla pipeline prefill di
vLLM:

- compressor ratio-4 fuso `pool + RMS + RoPE + FP8`, con scrittura diretta
  nella compressed KV F16;
- fusione `KV RoPE + FP8 + raw-cache insertion`;
- overlap multi-stream della trasformazione KV con la proiezione Q nei batch
  piccoli del verifier DSpark.

La regressione `sm_121a` confrontava sia il cold prefill sia il replay
append-prefill. Compressor F32, bit F16, KV trasformata e raw cache coincidevano
esattamente con il percorso precedente:

```text
cuda-regression: fused prefill pipeline compressor-max=0 f16-changed=0 replay-max=0 replay-f16-changed=0 kv-max=0 raw-max=0
cuda long-context regression: OK
```

L'overlap multi-stream ha ridotto il decode ed e' stato rimosso per primo. La
variante sequenziale specializzata ha eliminato un kernel per ciascuno dei 43
layer compressi del verifier (`4513 -> 4470` nodi CUDA Graph), ma il beneficio
end-to-end e' rimasto troppo piccolo:

| Gate 65K, 256 token | `e4ec80f` | Fusioni sequenziali | Variazione |
| --- | ---: | ---: | ---: |
| Prefill | 928,41 t/s | 931,35 t/s | +0,32% |
| Decode sintetico DSpark | 11,47 t/s | 11,58 t/s | +0,96% |
| Cicli / token per ciclo | 134 / 1,9104 | 134 / 1,9104 | invariati |
| RSS massima osservata | 1945,7 MiB | 2089,8 MiB | +144,1 MiB |

Hash greedy e acceptance erano invariati. Anche lo sweep 32K/65K/98K e'
risultato sostanzialmente neutro, con una coda lunga non stabile. L'intervento
non raggiunge quindi il gate prestazionale e peggiora il margine UMA: il codice
e' stato rimosso integralmente. La patch sperimentale resta archiviata fuori
dal repository in `ds4-experiments/vllm-prefill-kv-fusions-2026-07-24.patch`.

## FlashMLA direct-F16 dinamico oltre 131K - 23 luglio 2026

Portando il context fisico da 131072 a 262144 e' emersa una soglia artificiale:
il token-tile indexed attention rifiutava `n_comp > 32768`. Con ratio-4 questa
soglia coincide esattamente con 131072 token e causava il passaggio dal fast
path FlashMLA/HMMA al precedente kernel indexed. Nel run 256K il chunk che
terminava a 131072 misurava circa 860 t/s, mentre il successivo scendeva a
circa 648 t/s.

Il limite da 32768 righe e' necessario soltanto quando la compressed KV F32
deve essere convertita nella mirror F16 preallocata. Il percorso operativo
direct-F16 non crea tale mirror e legge soltanto le 512 righe selezionate.
L'eleggibilita' direct-F16 usa ora dinamicamente la capacita' effettiva del
tensore compressed KV; il limite F32 e quello del percorso dense restano
invariati. Non vengono aggiunti buffer o scratch.

La regressione CUDA esegue realmente il kernel a 32769 righe compresse e lo
confronta con il riferimento indexed:

```text
ds4: CUDA FlashMLA direct-F16 dynamic compressed capacity enabled (rows=32769 tensor-cap=32769)
cuda-regression: token-tile indexed direct-F16 attention above 32K rows rel-rmse=0.000091 max-abs=0.000040
cuda long-context regression: OK
```

La parita' numerica e il dispatch sono quindi validati. Il successivo run
end-to-end con lo stesso carico operativo ha confermato il recupero oltre la
soglia:

| Intervallo | Prima | Direct-F16 dinamico | Variazione |
| --- | ---: | ---: | ---: |
| primo chunk oltre 131K | 648,39 t/s | 858,48 t/s | +32,40% |
| append profondo | 624,92 t/s (147208 -> 207064) | 836,16 t/s (127827 -> 180844) | +33,80% |

Nel run aggiornato i sei chunk completi oltre la vecchia soglia hanno misurato
858,48, 849,68, 846,14, 842,61, 833,60 e 828,20 t/s. Il decadimento residuo e'
progressivo con la crescita del contesto, non piu' un salto netto a 131072.
L'append da 53017 token e' terminato in 63,406 secondi; il decode DSpark
immediatamente successivo ha generato 290 token a 19,892 t/s. Non sono apparsi
errori CUDA, OOM o fallback.

I due intervalli profondi hanno estremi diversi e il decode dipende dal
contenuto, quindi il confronto non va letto come un microbenchmark
position-matched. Dimostra pero' che il limite artificiale e' stato rimosso e
che il percorso ottimizzato rimane attivo nel regime per cui il context 256K
era stato introdotto.

## Arena condivisa indexer/MoE - 23 luglio 2026

Il prefill allocava contemporaneamente due workspace con vite disgiunte:

- `indexer_scores`, circa 1024 MiB con context 131072 e chunk 8192;
- gate, up, mid e down del routed MoE, usati soltanto dopo Top-K e attention.

`indexer_scores` occupa ora la regione gate/up/mid dell'arena routed-MoE.
La query indexer F32/MXFP4 rimane nella regione down, quindi scorer e query non
si sovrappongono. Dopo Top-K e attention gli score non sono piu' necessari e
il routed MoE puo' sovrascrivere la stessa memoria. Kernel, Top-512, precisioni
e ordine delle operazioni non cambiano.

La regressione CUDA completa e' passata, incluse parita' MoE N=2..6, scorer
MXFP4, Top-512 esatto e attention long-context. Il conteggio dei buffer
dichiara:

| Configurazione | Context buffer |
| --- | ---: |
| allocazioni separate | 2453,90 MiB |
| arena condivisa | 1429,84 MiB |

Il risparmio allocato e' esattamente 1024,06 MiB. La prima misura di
`MemAvailable` non viene usata come prova: il percorso diagnostico senza alias
allocava per errore sia l'arena completa sia i quattro buffer separati,
sovrastimando il delta fisico. Il fallback e' stato corretto e il dato UMA
andra' rimisurato. L'A/B prestazionale sullo stesso binario non ha mostrato
una regressione sistematica a 82K/98K:

| Frontiera | Arena | Allocazioni separate |
| --- | ---: | ---: |
| 65536 | 930,97 t/s | 949,49 t/s |
| 81920 | 904,61 t/s | 904,95 t/s |
| 98304 | 870,87 t/s | 867,56 t/s |

Il 65K resta rumoroso; le due frontiere profonde sono equivalenti entro circa
lo 0,4%. Gli hash greedy a 82K e 98K coincidono. La modifica non va confusa con
un'accelerazione del decode: il riferimento operativo DSpark rimane 22-25 t/s,
mentre i 9-11 t/s di `ds4-bench` derivano da prompt sintetici con acceptance
molto piu' bassa e valgono soltanto per confronti A/B identici.

### Reinvestimento del GiB e tuning time-boxed

Il margine recuperato e' stato provato anche come budget prestazionale, senza
promuovere risultati che violassero i gate:

- lo staging di otto indexer head ha mantenuto gli hash ma ha perso circa
  5,4-5,8% di prefill ed e' stato rimosso;
- spostare in shared il puntatore uniforme del token-tile attention ha ridotto
  lo stack SASS dense da 16 a 8 byte, ma il prefill pesato e' peggiorato di
  circa il 6,9%; anche questa patch e' stata rimossa;
- chunk 12288 con raw ring 8192 non e' valido e viene rifiutato durante
  l'attention; dimensionare il raw ring e usare chunk 10240 resta entro il
  vecchio budget, ma passa da 966,08 a 924,29 t/s pesati (-4,33%);
- padding Tensor Core del verifier da 8 a 4 ha migliorato il decode sintetico
  da 11,58 a 12,41 t/s (+7,17%), ma solo 3/5 risposte greedy della fixture
  coincidevano, con 82 contro 90 cicli. E' stato respinto per il requisito
  qualita' ASIS.

Il default resta quindi chunk 8192 e padding verifier 8. Nessuno dei quattro
esperimenti sopra rimane nel codice.

Il tentativo di reinvestire il margine in una cache Q8->F16 da 15 GiB non ha
dato un vantaggio ripetibile: rispetto a 12 GiB ha aggiunto solo circa 0,5 GiB
di cache effettiva e variazioni entro il rumore. Il chunk 16384 ha invece
superato il limite pratico UMA ed e' stato scartato; il chunk stabile resta
8192.

## Tentativo dedup esperti verifier scartato - 23 luglio 2026

E' stato adattato al kernel GB10 Q8_K il first-owner expert dedup del verifier:
per le righe con lo stesso esperto, una CTA avrebbe caricato il peso una volta
e prodotto piu' righe. Parita' N=2..6 e regressione CUDA erano corrette, ma
l'A/B K=3 controllato ha dimostrato che la pressione aggiuntiva su registri e
shared memory annullava il riuso:

| Variante | Cicli | Acceptance | Target | Ciclo fuso | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| stabile | 264 | 31,43% | 144,792 ms | 167,345 ms | 11,611 t/s |
| dedup | 264 | 31,43% | 145,372 ms | 167,933 ms | 11,570 t/s |

Prompt, hash e scheduler erano identici. La variante dedup, circa 0,4% piu'
lenta, e' stata rimossa integralmente.

## Tentativi di tuning aggiuntivi scartati - 23 luglio 2026

- Cache Q8->F16 ridotta a 11 GiB: prefill peggiore del 34-38%.
- `launch_bounds(512,2)` sul token-tile attention: 64 registri ma spill in
  local memory e prefill peggiore del 14-17%.
- Gate/up D2R N32 a una CTA senza spill: prefill peggiore del 2,5-3%.
- Chunk 4096: riduce i context buffer di circa 837 MiB e puo' migliorare il
  decode sintetico, ma perde circa 1,4-2,3% di prefill; rimane una possibile
  configurazione memory-first, non il default prestazionale.

## Tentativo sparse attention M16/R16 scartato - 23 luglio 2026

E' stata valutata una specializzazione del token-tile indexed attention con
un token e 16 head per CTA, stage KV da 16 righe, 384 thread divisi in quattro
warp score e otto warp PV. Lo scopo era passare dal kernel M32/R32 limitato a
una CTA per SM a due CTA residenti, senza cambiare Top-512, maschere o KV.

L'implementazione ha raggiunto tutti i gate architetturali previsti:

- shared memory dinamica ridotta da 88576 a 39168 byte per CTA;
- due blocchi residenti per SM verificati dalla CUDA occupancy API;
- 80 registri per thread e zero local memory/spill;
- regressione long-context completata con successo;
- confronto M16/R16 contro M32/R32: `rel-rmse=0.000218` e
  `max-abs=0.000000`, sia con cache F32 sia con compressed KV F16.

Il benchmark append-only standard ha pero' mostrato che la maggiore occupancy
non compensa la perdita di riuso delle righe KV sui contesti lunghi:

| Frontiera | M32/R32 stabile | M16/R16 | Variazione |
| --- | ---: | ---: | ---: |
| 65536 | 913,61 t/s | 945,24 t/s | +3,46% |
| 81920 | 937,77 t/s | 917,71 t/s | -2,14% |
| 98304 | 913,54 t/s | 877,52 t/s | -3,94% |

Il throughput pesato sui 98304 token passa da circa 917,54 a 928,65 t/s,
soltanto **+1,21%**, con un decadimento contrario all'obiettivo principale.
M16 richiede quattro CTA per coprire i 64 head invece delle due di M32 e R16
raddoppia iterazioni, sincronizzazioni e aggiornamenti dell'online softmax.
Il latency hiding ai contesti iniziali migliora, ma con la crescita della
compressed KV prevalgono le richieste L2 aggiuntive e il minore riuso
head-major per caricamento.

La variante e' stata quindi rimossa integralmente. Il percorso stabile resta
M32/R32; futuri interventi sulla sparse attention devono conservarne il riuso
di 32 head per riga KV e ridurre il costo interno senza aumentare il numero di
CTA che rileggono lo stesso Top-K. Il decode DSpark non era coinvolto nel
dispatch M16/R16 e non ha richiesto rollback separati.

## Tentativo overlap routed/shared verifier scartato - 23 luglio 2026

E' stato provato un fork/join CUDA Graph fra i due rami indipendenti della FFN
target: routed MoE su uno stream ausiliario e shared expert sullo stream
principale. Kernel, precisione e somma finale erano invariati e non venivano
allocati nuovi buffer di modello o scratch.

Il confronto per singola dimensione del verifier mostrava un segnale locale:
`K=4` passava da 170,918 a 160,871 ms di target e da 194,278 a 181,720 ms per
il ciclo completo. Il test end-to-end eseguito con lo stesso prompt ha pero'
violato i gate prioritari:

- verifier acceptance: 85,02% -> 65,05%;
- weighted request decode: 22,234 -> 19,751 t/s;
- il scheduler si e' spostato da quasi tutto `K=3` a 1420/1478 cicli `K=4`;
- mean fused cycle: 163,169 -> 180,487 ms.

Il diverso mix di K spiega una parte del ciclo medio, ma non giustifica
l'approvazione di una variante con throughput e acceptance peggiori sul test
operativo stabile. Non e' stato dimostrato se la causa fosse una dipendenza
CUDA Graph nascosta o l'interazione fra costo misurato e scelta adattiva di K.
L'overlap e le relative API stream/event sono stati rimossi integralmente.
Futuri interventi sul verifier devono essere serialmente deterministici oppure
superare un A/B greedy con hash identico prima della misura prestazionale.

## Resoconto rollback proiezioni — 22 luglio 2026

Il branch `main` è stato riallineato alla versione pubblicata su `origin/main`,
commit `3d113b8` (`Add detailed DSpark verifier profiling`). Il rollback ha
rimosso da `main` sia il commit locale non pubblicato `0af7406` sia le modifiche
non committate sviluppate durante l'esperimento sulle proiezioni.

Prima del rollback è stato creato un backup completo e recuperabile:

- branch locale: `codex/backup-projection-work-20260722-220423`;
- commit del backup: `79c5072`;
- bundle verificato: `/Users/MarcoPalaferri/Documents/Personale AI/ds4-gb10-lab-projection-backup-20260722-220423.bundle`.

Il backup comprende l'infrastruttura per separare le policy di proiezione del
target verifier e del DSpark decode, le policy sperimentali `auto`, i percorsi
Q8/F16, la telemetria dei fallback, gli aggiornamenti del benchmark GB10, il
microbenchmark dei blocchi di proiezione e la documentazione del piano.

### Motivo del rollback

L'obiettivo era aumentare il decode di almeno il 10% senza modificare la
semantica del modello target. Nessuna delle policy sperimentali ha superato
contemporaneamente i gate di prestazione e correttezza:

- `target/auto` generale mostrava un aumento apparente del decode del 15,25%,
  ma il tempo del target peggiorava del 5,72%, con hash differenti, fallback e
  variazioni dell'acceptance;
- `dspark/auto` peggiorava il decode del 5,49% e il drafter del 21,51%;
- il solo riuso Q8 sul target produceva circa +1,31% di decode e +0,14% sul
  target, insufficienti e accompagnati da hash e acceptance differenti;
- la variante `narrow-f16` produceva +2,69% di decode, ma il target peggiorava
  del 4,03% e l'hash non coincideva.

Anche la promozione di `fixed-k2`, scelta da un benchmark greedy deterministico,
non era rappresentativa del server reale con rejection sampling p/q. Nel
carico server osservato, `fixed-k2` raggiungeva 18,619, 18,764 e 17,385 token/s
ai contesti confrontabili. Ripristinando il draft completo e lo scheduler
adaptive sono stati misurati 21,856, 22,289 e 21,548 token/s: un recupero
rispettivamente del 17,39%, 18,79% e 23,95%, circa 19,86% aggregato. Questo è
un recupero della regressione introdotta, non un miglioramento rispetto al
baseline storico, che aveva mostrato richieste tra 25 e 29 token/s.

La conclusione dell'esperimento è quindi negativa: l'obiettivo del +10% non è
stato raggiunto e le modifiche non devono entrare nella pipeline di produzione.
Il ramo pubblicato resta il riferimento stabile; le proiezioni devono rimanere
sulla policy `legacy` finché un candidato non supera un confronto controllato
end-to-end sul server reale.

Per ispezionare nuovamente il lavoro senza modificare `main`:

```bash
git switch codex/backup-projection-work-20260722-220423
```

Gli eseguibili e gli oggetti di compilazione sono ignorati da Git e non vengono
modificati da un reset del sorgente. In questo checkout sono stati rimossi con
`make clean`; prima di distribuire il rollback occorre compilare nuovamente la
versione `3d113b8`.

## Patch DSpark circoscritte successive al rollback

Il lavoro successivo non riapre le proiezioni e non modifica scheduler,
sampling, quantizzazione o target verifier. Comprende soltanto due correzioni
reversibili:

1. la confidence head usa per default l'hidden DSpark dopo l'HC collapse e
   prima della RMSNorm, come `forward_head()` nel modello DeepSeek pubblicato;
2. ciascuna variante del verifier DSpark può conservare due topologie CUDA
   Graph. Se `cudaGraphExecUpdate` non è compatibile con la prima, la seconda
   viene riutilizzata invece di distruggere e ricostruire alternativamente gli
   stessi eseguibili da 4K+ nodi osservati nei log Athena.

Rollback A/B senza ricompilazione:

```text
DS4_DSPARK_CONFIDENCE_POST_NORM=1
DS4_CUDA_DSPARK_GRAPH_TOPOLOGY_CACHE_DISABLE=1
```

`run-dspark-server.sh` abilita esplicitamente entrambi i percorsi per default e
stampa `confidence-input=pre-RMSNorm` e
`verifier-topology-cache=two-slot` nella configurazione iniziale. Accetta le
variabili di rollback precedenti soltanto con valore `1`; il valore `0` viene
normalizzato rimuovendo la variabile dall'ambiente prima dell'avvio.

Con `DS4_CUDA_DSPARK_GRAPH_VERBOSE=1`, il contatore `topology_reuses` indica
quante ricostruzioni sono state evitate trovando una topologia compatibile nel
secondo slot.

### Risultato A/B Athena del 22 luglio 2026

Un confronto sul server reale con telemetria attiva ha misurato il candidato
con entrambe le correzioni abilitate e la baseline ottenuta impostando insieme
le due variabili di rollback precedenti:

| Metrica | Baseline | Correzioni attive | Variazione |
| --- | ---: | ---: | ---: |
| Decode pesato delle richieste | 19,362 t/s | 22,234 t/s | **+14,83%** |
| Throughput dei cicli verifier | 19,438 t/s | 21,744 t/s | **+11,86%** |
| Tempo medio target | 158,389 ms | 142,749 ms | **-9,87%** |
| Tempo medio ciclo fused | 179,444 ms | 163,169 ms | **-9,07%** |
| Acceptance verifier | 63,21% | 85,02% | **+21,81 punti** |
| Ricostruzioni CUDA Graph nei primi 1000 launch | 77 | 33 | **-57,14%** |

Il risultato è coerente con entrambi gli interventi. L'hidden pre-RMSNorm
porta lo scheduler prevalentemente a `K=3` (605 cicli su 646), mentre la
baseline post-RMSNorm sceglie prevalentemente `K=4` (760 cicli su 797). Le
righe medie del target scendono così da 4,936 a 3,997. In parallelo, il secondo
slot CUDA Graph registra 196 riusi di topologie che altrimenti avrebbero
richiesto una nuova istanziazione.

Il guadagno osservato supera quindi l'obiettivo del 10%. Il confronto conserva
una cautela metodologica: l'aggregato candidato contiene sette request summary,
quello baseline quattro. Le metriche interne concordano nella stessa direzione,
ma una certificazione strettamente appaiata deve ripetere lo stesso insieme di
richieste con telemetria disattivata.

## Stato corrente

Stato al 18 luglio 2026:

- contesto fisico: 131072 token;
- contesto pubblicizzato ai client: 85%;
- profilo predefinito: `DS4_MEMORY_PROFILE=balanced`;
- cache calda Q8→F16: 12288 MiB;
- prefill chunk: 8192;
- target e sidecar DSpark copiati in device memory;
- pagine sorgente GGUF rilasciate dopo la copia CUDA;
- append prefill, checkpoint canonico e long anchor NVMe attivi;
- binding visibile RAM post-tool e protezione del checkpoint NVMe entrante;
- routed-MoE prefill MMQ sui GGUF canonici, senza artifact SoA da 76,5 GiB;
- stream-K MMQ limitato strutturalmente dal numero di token;
- CUDA Graph DSpark K-aware e verifier target `K+1`;
- sampling speculativo lossless p/q;
- context guard HTTP coerente con l'85% pubblicizzato.

Il riferimento validato prima dell'ultimo intervento attention è circa
`399,92 t/s` medi su 58,5K token di append prefill. Sul tratto confrontabile
24,5K–57,3K sono stati misurati circa `425,9 t/s`; il decode DSpark a 83K ha
prodotto 401 token a `23,00 t/s`.

### Fast path MoE aligned del verifier DSpark validato

Il verifier target con batch `N=2..6` usa ora un kernel aligned dedicato a
GB10. Gate/up conserva la riduzione quarter-warp precedente, ma calcola i
segni IQ2 validati nei registri, copia le attivazioni Q8_K in shared memory con
accessi coalescenti e porta il row span da 128 a 256. Il down carica una sola
volta le sei attivazioni Q8_K per CTA e le riusa su due wave da 32 righe; le
scale Q2_K restano in registri durante il dot product.

Il dispatch e' limitato strutturalmente a `N=2..6`. Il decode target `N=1`, il
drafter Q4 DSpark e il prefill con piu' di 16 token conservano i rispettivi
percorsi precedenti. Non sono state aggiunte allocazioni persistenti o flag.
La regressione Athena `sm_121a` ha esercitato tutte le cinque forme e ha
confermato un risultato bit-identico al kernel aligned precedente:

```text
cuda-regression: GB10 aligned MoE verifier N=2..6 parity max=0 bad=0
cuda long-context regression: OK
```

Nel run end-to-end successivo il decode DSpark ha misurato `29,24 t/s` a
27,7K token, `26,47 t/s` a 63,5K e `25,26 t/s` a 88,8K. Contenuto e acceptance
non rendono questi valori un A/B isolato, ma sui contesti confrontabili il
guadagno osservato rispetto ai riferimenti precedenti e' nell'ordine del
5–9%. Il prefill non ha mostrato regressioni: i due chunk centrali del cold
prefill 25K hanno prodotto `856,25` e `839,38 t/s`, circa `847,8 t/s` medi.

Log di attivazione atteso alla prima verifica speculativa:

```text
ds4: CUDA GB10 aligned MoE verifier enabled (computed IQ2 signs, coalesced Q8 staging, sum6 row span=64)
```

### Tentativo Q8 exact-MMA del vocab head scartato

E' stato portato in forma circoscritta l'exact-MMA Q8 pubblicato da
`antirez/ds4` sul solo vocab head del target verifier `N=2..6`. Il kernel usava
`mma.sync.m16n8k32.s8`, pesi Q8_0 raw e un tile interno completato a otto
righe. La regressione ha confermato parita' bit per bit con il precedente
kernel Q8 per tutte le cinque forme (`max-abs=0`, `bad=0`). Anche il confronto
end-to-end contro il binario stabile `99b8dee` ha prodotto cinque risposte
identiche, 82 cicli, 275 token proposti e 182 token committati in entrambi i
casi.

Il gate prestazionale e' invece fallito. Nel run di produzione il decode a
66,7K token ha ottenuto `23,09 t/s`, contro il riferimento stabile di
`26,47 t/s` a 63,5K, circa `-12,8%`; gli altri turni non hanno mostrato un
incremento ripetibile. Il prefill e' rimasto invariato, con circa `976 t/s`
sui primi due chunk lunghi. Sul profilo GB10 attuale il vocab head e' gia'
servito dalla cache F16 calda: quantizzazione delle attivazioni, padding e
riduzione del percorso Q8 raw costano piu' del traffico risparmiato. Il kernel,
il dispatch e il relativo self-test sono stati quindi rimossi completamente.

Rimane disponibile `tools/dspark_acceptance_fixture.py` come gate generico per
le prossime ottimizzazioni. Confronta due binari entrambi con DSpark attivo e
verifica contenuto, reasoning, finish reason, cicli, token proposti e token
committati. Una comparazione target-only contro DSpark non e' un gate ASIS
valido, perche' misura due percorsi di esecuzione differenti gia' nel baseline.

### Tentativo DSpark HMMA attention scartato

Dopo `5d54db3` e' stato provato un percorso dedicato alla forma DSpark
`5 token x 64 head x 512`, con QK Tensor Core FP16, softmax FP32 e value FP32
tile da otto query. Il percorso non ha prodotto un miglioramento end-to-end ed
e' stato rimosso.

Il profilo Nsight del baseline spiegava il risultato: 42 chiamate al kernel di
attenzione costavano complessivamente `10,70 ms` in 14 cicli DSpark da
`2623,89 ms`, cioe' appena lo `0,408%` del percorso completo. Anche eliminando
interamente quel costo, il limite teorico sarebbe stato circa `+0,41%`. La
pipeline HMMA aumentava inoltre i nodi del CUDA Graph K5 da 213 a 225 e la
conversione Q/K FP16 poteva modificare l'acceptance del drafter. Il run freddo
e' rimasto sostanzialmente invariato (`17,927` contro `17,958 t/s`), senza un
beneficio che giustificasse complessita' e rischio. Il collo di bottiglia macro
resta il target verifier, pari a circa l'`86,7%` del ciclo profilato.

### Tentativo verifier F16/CUTLASS scartato

Sul commit `5d54db3` e' stato provato un planner `(M,N,K,label)` per N reale,
4 e 8, con `cublasLt`, tre kernel CUTLASS narrow-N, grouped gate/up e q/kv,
output padded diretto, riuso delle attivazioni e un percorso interleaved per
`attn_output_a`. La regressione era corretta e `dspark_output` isolato
migliorava di circa il 21,8%, ma grouped GEMM era sempre piu' lento. Il decode
stabile e' rimasto circa 19,4-20,0 t/s e il tuning piu' la costruzione dei
Graph penalizzavano pesantemente la prima risposta. L'intervento e' stato
quindi rimosso: il beneficio pesato non giustificava oltre 3.000 righe di
planner e una dipendenza CUTLASS vendorizzata. Lo snapshot completo e' stato
conservato fuori dal repository in `ds4-verifier-cutlass-backup-20260719`.

### Token-tile HMMA: gate prestazionale superato

Il worktree corrente aggiunge token-tile HMMA per l'attenzione prefill:

- tile di 16 token e due head per CTA;
- stessa selezione Top-K esatta da 512 righe nel percorso indexed ratio-4;
- unione dei candidati per tile senza sort globale dei Top-K;
- variante dense per i layer raw e mixed;
- mirror KV F16 temporanei, senza copie permanenti dei pesi;
- scratch dedicato con high-water massimo di circa 168 MiB;
- guardie strutturali: almeno 128 token, 64 head, head dimension 512 e window
  128. Decode a token singolo e verifier DSpark da 2–6 righe sono esclusi.

Il run Athena del 17 luglio ha attivato entrambi i percorsi e ha superato
ampiamente il gate prestazionale. Sullo stesso intervallo assoluto
`24576..81920` (57.344 token) il throughput pesato e' passato da 404,46 a
509,14 t/s, pari a **+25,88%**. I sette chunk completi migliorano tutti, in una
banda compresa tra +23,25% e +26,52%. L'intera richiesta da 61.214 token ha
chiuso a 496,57 t/s; il decode successivo a circa 86K ha prodotto 802 token a
23,58 t/s, contro il riferimento di 23,00 t/s.

Non sono comparsi BOS, non-finite o errori CUDA e la risposta ha concluso 11
tool call valide. Resta da annotare il dato di memoria residente del processo;
il codice limita lo scratch aggiuntivo a circa 168 MiB. I log di regressione
richiesti sono:

```text
cuda-regression: token-tile indexed attention rel-rmse=...
cuda-regression: token-tile raw/mixed attention rel-rmse=...
cuda long-context regression: OK
```

Un run da 99.143 token ha inoltre scoperto il confine esatto
`n_comp=24576`: la bitmap union occupa 48 KiB dinamici, ai quali si aggiungono
circa 2 KiB statici del kernel. Il vecchio launcher non richiedeva l'opt-in
CUDA a questa dimensione e falliva con `invalid argument`. Il launcher
configura ora la dimensione dinamica reale fin dal primo uso e la regressione
esercita esplicitamente questo confine.

### Port FlashMLA head-major SM121a: validato su Athena

Il worktree successivo sostituisce, per il solo sparse prefill indexed, la
bitmap union multi-token con il mapping token-level usato da FlashMLA. Il
riferimento studiato e' `deepseek-ai/FlashMLA` al commit
`9241ae3ef9bac614dd25e45e507e089f888280e0` (MIT). Non vengono importati
PyTorch, CUTLASS SM90/SM100, WGMMA, TMA gather o TMEM non disponibili sul
GB10: il port conserva le primitive HMMA e `cp.async` gia' validate su
`sm_121a`.

La nuova fattorizzazione mantiene invariata la matrice CTA `M=32`, ma passa da
`16 token x 2 head` a `1 token x 32 head`. Ogni CTA legge quindi direttamente
i Top-512 del proprio token e riusa ciascuna riga KV su 32 head. Gli indici
vengono caricati 32 alla volta nella ring shared, validati contro la frontiera
causale e consumati senza creare una copia globale `int2` e senza scandire
tutta la bitmap `n_comp`. Restano invariati:

- Top-K 512 e relativo ordine;
- raw window da 128 token, sink e maschera causale;
- KV compressed F16 e raw mirror F16 transiente;
- QK, online softmax e accumulazione output FP32;
- decode, verifier DSpark e percorso dense ratio-128.

Il numero complessivo di CTA HMMA resta uguale: due CTA per token coprono i 64
head. La griglia usa head-group sull'asse `x`, cosi' le due CTA dello stesso
token vengono offerte consecutivamente allo scheduler e possono riusare le KV
selezionate in L2. Il lavoro sparse di ogni CTA e' limitato a `128 + 512` righe,
indipendentemente dalla sovrapposizione dei Top-K fra token adiacenti. Anche lo
scratch migliora: il percorso indexed non materializza piu' i record globali;
il percorso dense dimensiona il record stride su `n_comp` reale invece del
massimo fisso 32768. Non viene aggiunta memoria permanente.

Il log atteso e':

```text
ds4: CUDA FlashMLA-style exact sparse prefill enabled (token=1, heads=32, stage=32, direct-topk=512, comp-kv=direct-f16)
```

Il gate `cuda-regression` su `sm_121a` ha superato i quattro confronti
token-tile e la regressione long-context. Le relative RMSE sono rimaste
nell'ordine di `6,5e-4`/`6,7e-4` sia con compressed KV F32 sia con il percorso
direct-F16; `cuda long-context regression: OK` ha chiuso il test.

Nel confronto sui chunk completi alle stesse frontiere, il throughput medio e'
passato da 794,56 a 864,24 t/s, pari a **+8,8%**. I quattro intervalli
confrontabili hanno misurato:

```text
32768..40960: 823,58 -> 879,17 t/s  (+6,7%)
40960..49152: 807,45 -> 877,45 t/s  (+8,7%)
49152..57344: 798,85 -> 871,53 t/s  (+9,1%)
65536..73728: 748,37 -> 828,82 t/s (+10,7%)
```

Il mapping e' circoscritto al prefill indexed: decode, verifier DSpark e
percorso dense non lo possono selezionare. Non aggiunge una copia permanente
dei pesi o della KV; rimuove invece i record globali del percorso indexed.

Il gate qualitativo API del 19 luglio 2026 ha usato 12 casi fissi
`gsm8k_cot_zeroshot` e 12 casi IFEval, temperatura zero e 2.200 token massimi.
Ha ottenuto **12/12** su GSM8K `flexible-extract`, **12/12** IFEval
`prompt_level_strict_acc` e **100%** IFEval `inst_level_strict_acc` in 12m28s.
Il valore GSM8K `strict-match=0` e' soltanto formato: DS4 emette
`\\boxed{...}` invece della frase letterale attesa dal filtro stretto. Il
primo tentativo con il default corto non e' valido: interrompeva il modello
prima di `</think>` e faceva valutare come risposta il reasoning incompleto.

### Pipeline SM121 nel worktree: implementata, gate Athena pendente

Il worktree successivo al riferimento sopra integra tutte le fasi del piano
indexer, MoE, attention e deep decode. I valori attesi del piano non sono
riportati come risultati: questa versione deve ancora superare compilazione,
regressione e benchmark end-to-end su Athena.

- `tools/benchmark_gb10.py` esegue tre processi per sweep cold e append alle
  frontiere 12K, 32K, 64K, 80K e 96K. Registra startup, prefill, decode DSpark,
  RSS/HWM prima e dopo la creazione della sessione, picco osservato e hash FNV
  dei token greedy in CSV e JSON. Frontiere mancanti o hash non deterministici
  fanno fallire il comando;
- la index cache ratio-4 usa righe MXFP4 da 68 byte: 64 byte E2M1 e quattro
  scale UE8M0. Le frontiere del compressor restano F32 e le query packed sono
  transienti. Il checkpoint corrente e' v4 packed; restore locale e distribuito
  continuano a leggere il precedente v3 F32;
- lo scorer SM121a usa MMA block-scaled nativo, mentre l'exact Top-512 passa al
  Radix Select oltre 8192 righe. La score matrix resta intenzionalmente
  materializzata: eliminarla ricreerebbe il costo del LiteTopK gia' scartato;
- gate/up IQ2_XXS e down Q2_K del solo target sono ripaccati SoA direttamente
  nelle stesse regioni device. Il catalogo richiede gate, up e down completi per
  ogni layer; non esiste una seconda copia permanente e il sidecar DSpark non
  viene trasformato;
- fino a 16 token, quindi anche decode e verifier `K+1`, gate/up, SwiGLU,
  routing weight, down e somma dei sei slot usano il tier vector diretto. Il
  prefill piu' largo usa D2R expert-major e una riduzione finale deterministica
  in ordine di slot; una scatter atomica cambierebbe l'ordine numerico;
- la compressed attention KV persistente e' F16. Poiche' la riga viene prima
  arrotondata nel formato FP8 del modello, il successivo storage F16 non perde
  informazione. Token-tile la legge direttamente e applica conditional softmax;
- lo stage resta staticamente a 32 righe e occupa 88.576 byte. Uno stage 64
  non e' incompleto: i soli due ring KV richiederebbero 131.072 byte, oltre il
  limite shared-memory da 90 KiB della GB10;
- la conversione Q e' caricata una sola volta nei registri della CTA che
  possiede quella coppia di head. Uno staging globale F16 non introdurrebbe
  riuso tra CTA e richiederebbe circa 512 MiB a chunk 8192, incompatibile con
  il margine RAM disponibile;
- il deep decode a token singolo usa GVR exact oltre 12K righe quando possiede
  un hint valido. Verifica e refinement conservano Top-512 esatto e ricadono su
  Radix; hint e validita' seguono snapshot, partial accept e rollback DSpark.

Il dispatch e' strutturale e automatico. Non sono stati aggiunti flag al
launcher, non cambiano Top-K, maschere, sink, softmax online, pesi o sampler del
target. I vecchi percorsi rimangono soltanto per forme non eleggibili e come
riferimento numerico della regressione.

Log di attivazione attesi, quando le relative forme vengono realmente eseguite:

```text
ds4: CUDA target MoE replaced in place: ... (zero permanent duplication)
ds4: CUDA in-place aligned MoE execution active ...
ds4: CUDA complete fused MoE D2R prefill enabled ...
ds4: CUDA packed MXFP4 indexer scorer enabled ...
ds4: CUDA exact radix Top-512 enabled ...
ds4: CUDA FlashMLA-style exact sparse prefill enabled (... direct-topk=512, comp-kv=direct-f16)
ds4: CUDA Blackwell exact GVR Top-512 enabled ...
```

### Fusione completa MoE D2R validata su Athena

Il percorso target SoA con piu' di 16 token dispone ora di una pipeline
`m128n32` che elimina le materializzazioni globali fra gate/up e down:

1. una CTA carica una sola volta il tile Q8_1 dell'attivazione;
2. due ring IQ2 indipendenti accumulano gate e up in registri, nello stesso
   ordine K del D2R precedente;
3. sanitizzazione, clamp, SwiGLU e routing weight sono applicati prima che i
   risultati lascino la CTA;
4. la shared memory dei ring viene riutilizzata come tile F32 e il risultato
   viene quantizzato direttamente nel layout Q8_1 D2S6 expert-major richiesto
   dal Q2_K down;
5. il down D2R e la somma deterministica dei sei slot restano invariati.

Il tile da 32 colonne mantiene due CTA residenti: shared di calcolo, staging e
route weights occupano 45.408 byte per CTA, quindi 90.816 byte per due CTA,
sotto il limite di 90 KiB della GB10. Il Q8_1 intermedio riusa lo scratch gate
gia' allocato; non viene introdotta memoria permanente o transiente aggiuntiva.
Il decode e il verifier DSpark, limitati al tier fino a 16 token, non entrano in
questo kernel. I dump diagnostici conservano automaticamente il percorso
materializzato precedente.

La regressione Athena `sm_121a` ha confermato:

```text
cuda-regression: complete fused D2R MoE parity final=0.00000000 max=0 bad=0
cuda long-context regression: OK
```

Durante un prefill reale deve inoltre comparire una sola volta:

```text
ds4: CUDA complete fused MoE D2R prefill enabled (preallocated workspace, register gate/up, direct SwiGLU Q8 down)
```

Su tre chunk con gli stessi intervalli di contesto del riferimento, il
throughput e' passato da 780,88/777,00/761,57 t/s a
798,07/800,24/785,47 t/s: `+2,77%` pesato. Un cold prefill da 92.995 token ha
chiuso a 758,39 t/s medi, mantenendo 727,01 t/s sul chunk 73.728..81.920.
Il decode successivo ha prodotto 811 token a 26,97 t/s; un secondo controllo
a 85.831 token ha misurato 25,89 t/s. I context buffer sono rimasti invariati
a 2.453,90 MiB.

Il trace Nsight sul chunk 32.768..40.960 mostra 10 operazioni GPU per layer nel
percorso MMQ contro le 12 precedenti. Il tempo MoE proiettato scende da 2,698
a 2,618 secondi (`-3,0%`): il kernel combinato costa 38,88 ms/layer contro
40,81 ms/layer complessivi di gate/up, SwiGLU pesata e quantizzazione separate.
Il risultato e' quindi promosso come miglioramento incrementale a qualita',
memoria e decode invariati; non rappresenta da solo un incremento macro del
prefill.

### Workspace MoE D2R senza allocazioni validato su Athena

Il profilo Nsight successivo alla fusione completa ha attribuito 1,003 secondi
dei 10,478 secondi del chunk a 258 chiamate `cudaMallocAsync`: esattamente sei
allocazioni per ciascuno dei 43 layer. Il fast path target con piu' di 16 token
ora riusa buffer batch che in quel ramo erano gia' residenti ma inutilizzati:

1. `up` contiene il Q8_1 dell'attivazione raccolta;
2. `gate` contiene il Q8_1 prodotto direttamente da gate/up + SwiGLU;
3. `mid` ospita `ids_src1`, `ids_dst`, confini esperto e una sola worklist
   condivisa in sequenza da gate/up e down.

Capienza e non sovrapposizione delle quattro regioni, incluso l'output down,
vengono controllate prima di accodare i kernel. Se una forma non e' compatibile,
il dispatcher conserva il fallback materializzato precedente. Non cambiano i
kernel D2R, l'ordine delle riduzioni, i pesi, i routing weight o il decode
DSpark; non viene inoltre riservata memoria aggiuntiva. Il gate Athena e':

```text
cuda-regression: complete fused D2R MoE parity final=0.00000000 max=0 bad=0
cuda long-context regression: OK
```

Il trace Nsight sullo stesso chunk da 8192 token ha confermato che le 258
`cudaMallocAsync`, pari a 1,003 secondi, sono scomparse completamente. Il tempo
host del range `mmq_fused` e' sceso da 1,019 secondi a 4,45 ms, mentre il tempo
GPU MoE e' rimasto sostanzialmente invariato, da 2,618 a 2,615 secondi. Questo
conferma che sono state eliminate attese di allocator senza modificare i
kernel numerici.

Una parte dell'attesa si e' spostata sul `cudaDeviceSynchronize` finale, salito
da 9,326 a 9,919 secondi. Il bilancio resta positivo: il chunk profilato e'
passato da 10,475 a 10,040 secondi (`-4,15%`, equivalente a circa `+4,33%` di
throughput). Nel run ordinario, quattro intervalli globali direttamente
confrontabili fra 24.576 e 57.344 token migliorano in media del `+3,0%`; il
chunk 73.728..81.920 passa da 727,01 a 757,28 t/s (`+4,16%`). Il decode resta
sopra il target, con 22,68 t/s a 64K e 21,97 t/s a 91K, e i context buffer
restano invariati a 2.453,90 MiB.

### Ripristino del fast path D2R con arena condivisa

L'audit del 26 luglio 2026 ha trovato una regressione di dispatch introdotta
dall'arena condivisa usata per estendere il contesto oltre 131K. Il tensor
`batch_routed_gate` possiede l'intera arena da 1920 MiB, ma il segmento gate
logico di un chunk da 8192 token occupa soltanto 384 MiB. Passare alla guardia
di overlap la capienza del proprietario faceva apparire gate sovrapposto alle
view adiacenti `up`, `mid` e `down`; il complete fused D2R veniva rifiutato e
tutti i layer usavano il fallback materializzato.

Il dispatcher passa ora la capienza logica del solo segmento gate. Non cambia
puntatore, allocazione, kernel o ordine numerico: viene soltanto resa corretta
la guardia fra quattro regioni gia' disgiunte. Codici di ritorno distinti e un
messaggio una tantum rendono immediatamente visibile l'eventuale causa di un
fallback senza aggiungere telemetria al percorso caldo.

La regressione Athena `sm_121a` e' rimasta bit-exact:

```text
cuda-regression: complete fused D2R MoE parity final=0.00000000 max=0 bad=0
cuda long-context regression: OK
```

Con contesto fisico 262144, chunk 8192 e sidecar DSpark Q2, un A/B controllato
ha portato il primo blocco da 960,94 a 1001,94 t/s medi (`+4,27%`) e il blocco
8192..16384 a 1006,61 t/s. I singoli run sono stati
1002,75/1001,12 t/s a 8K e 1006,52/1006,69 t/s a 16K. Gli hash greedy sono
rimasti rispettivamente `1fb5af6f4176aaee` ed `e860341538aa8276`; non viene
aggiunta memoria persistente e il verifier DSpark da 1-6 righe non e'
modificato.

## Requisiti

- NVIDIA GB10 / DGX Spark con Linux ARM64;
- CUDA Toolkit 13 con `/usr/local/cuda/bin/nvcc` e supporto `sm_121a`;
- toolchain C/C++ (`make`, `cc`, `git`);
- almeno 128 GB di memoria unificata per il profilo `balanced`;
- spazio disco per target GGUF, sidecar DSpark e KV cache;
- porta TCP 30007 libera.

Verifica minima:

```bash
/usr/local/cuda/bin/nvcc --version && nvidia-smi && git --version && make --version
```

## Avvio rapido

### Installazione automatica

Su una macchina nuova, lo script seguente scarica il target finale Antirez
DeepSeek-V4-Flash-0731 Q2/imatrix in `/home/athena/ds4`, scarica gli shard
ufficiali DSpark, costruisce entrambi i sidecar Q4 e Q2, esegue
`cuda-regression` e compila il server:

```bash
cd /home/athena && git clone https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA.git && cd /home/athena/DS4-GB10-GX10-DSpark-CUDA && ./install-gb10.sh --install-deps --dspark both
```

Per costruire una sola variante usare `--dspark q4` oppure `--dspark q2`.
Download e GGUF gia' completi vengono riutilizzati; `--force-sidecar` forza
soltanto la rigenerazione dei sidecar.

Il file target predefinito e':

```text
/home/athena/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

Se non e' presente localmente, l'installer prova a scaricarlo dal repository
Hugging Face `antirez/deepseek-v4-gguf`. Un file remoto assente o un download
interrotto lascia invariato il collegamento del modello attivo. Il GGUF 0731
occupa 86,72 GB decimali: non esiste una patch binaria piccola e affidabile da
applicare ai pesi preview, quindi l'aggiornamento richiede il file completo.

La regression CUDA richiede la UMA libera e, per default, l'installer si ferma
se rileva un qualunque processo `ds4-server`, anche appartenente a un altro
checkout. Con `--skip-regression` download, conversione e compilazione possono
invece proseguire mentre quel server rimane attivo; prima di promuovere il build
occorre fermarlo ed eseguire separatamente `make -B cuda-regression
CUDA_ARCH=sm_121a`.

### 1. Checkout

Nuova installazione:

```bash
git clone https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA.git ~/DS4-GB10-GX10-DSpark-CUDA && cd ~/DS4-GB10-GX10-DSpark-CUDA
```

Aggiornamento di un checkout pulito:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && git fetch origin && git pull --ff-only origin main
```

Non usare `git pull` sopra modifiche locali. Durante lo sviluppo dal Mac usare
il deploy rsync descritto sotto.

### 2. Modelli

Il launcher predefinito richiede:

```text
/home/athena/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
/home/athena/ds4/DeepSeek-V4-Flash-DSpark-IQ2XXS-Q2K-Q8.gguf
```

La variante Q4 opzionale usa:

```text
/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf
```

Il target finale 0731 Q2/imatrix puo' essere scaricato con:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && mkdir -p /home/athena/ds4 && DS4_GGUF_DIR=/home/athena/ds4 ./download_model.sh q2-imatrix-0731
```

Dopo il download, creare o aggiornare il collegamento atteso dal launcher:

```bash
ln -sfn /home/athena/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf /home/athena/ds4/ds4flash.gguf
```

Aggiornamento di un'installazione preview esistente, senza cancellare il
vecchio target:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && git fetch origin && git pull --ff-only origin main && ./upgrade-target-0731.sh --model-dir /home/athena/ds4
```

Lo script riprende i download interrotti, valida la dimensione completa e
aggiorna `ds4flash.gguf` soltanto alla fine. Non modifica i sidecar DSpark. Il
launcher usa comunque il nome 0731 esplicito come default; se manca, termina
con il comando di upgrade invece di avviare silenziosamente il preview. Per un
rollback intenzionale usare
`DS4_MODEL=/percorso/del/precedente.gguf ./run-dspark-server.sh`.

Per costruire il sidecar servono gli shard Hugging Face 46–48 e
`model.safetensors.index.json` in
`/home/athena/ds4/dspark-v4flash-hf`. Quando sono presenti:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./build-dspark-sidecar.sh 2>&1 | tee /tmp/ds4-dspark-convert.log
```

Percorsi differenti possono essere passati con `DS4_MODEL`,
`DS4_DSPARK_MODEL`, `DS4_DSPARK_HF_DIR` e `DS4_DSPARK_GGUF`. Per costruire e
usare il sidecar compatto non serve alcun override: il valore predefinito e'
`q2`. Per selezionare Q4 impostare
`DS4_DSPARK_VARIANT=q4`. Entrambi i percorsi predefiniti sono sotto
`/home/athena/ds4`.

### 3. Regressione CUDA obbligatoria

Fermare il server prima del test per evitare pressione di memoria, quindi:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && make -B cuda-regression CUDA_ARCH=sm_121a
```

Il comando compila anche gli oggetti `cuda/mmq`. L'esito valido termina con
`cuda long-context regression: OK`; warning, errori di parità, non-finite o
fallimenti precedenti a quella riga bloccano il deploy.

### 4. Build del server

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && make -B cuda-spark-graph-sm121
```

La riga NVCC deve contenere
`-gencode=arch=compute_121a,code=sm_121a`,
`-DDS4_CUDA_SM121A_MXF4_MMA`, `--default-stream per-thread` e
`-DDS4_CUDA_TOKEN_GRAPH_BUILD`. Il `-gencode` esplicito e' necessario: su
alcune toolchain lo shorthand `-arch=sm_121a` perde il suffisso `a` nel PTX e
produce `.target sm_121`, che non abilita le istruzioni MMA block-scaled MXFP4.

### 5. Avvio

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-dspark-server.log
```

Avvio con il sidecar Q2 compatto:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && DS4_DSPARK_VARIANT=q2 ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-dspark-q2-server.log
```

Il launcher usa per default `balanced`, chunk 8192, context 262144, advertise
85%, cache Q8→F16 da 12 GiB, porta 30007 e KV cache in
`/tmp/ds4-gb10-dspark-kv`.

Verifica:

```bash
curl -fsS http://127.0.0.1:30007/v1/models
```

Log principali attesi durante startup e primo prompt:

```text
ds4: CUDA pipelined model copy ...
ds4: CUDA complete fused MoE D2R prefill enabled ...
ds4: CUDA token-tile HMMA raw/mixed prefill enabled ...
ds4: CUDA FlashMLA-style exact sparse prefill enabled ...
```

I messaggi dei percorsi prefill compaiono soltanto quando una forma larga
eleggibile viene realmente eseguita.

## Deploy dal Mac

Lo script usa `rsync` e non modifica il repository Git su Athena:

```bash
cd "<local-ds4-checkout>" && ATHENA_HOST=192.168.254.62 ATHENA_USER=athena ATHENA_DEST=/home/athena/DS4-GB10-GX10-DSpark-CUDA/ SSH_KEY=~/.ssh/id_ed25519 ./deploy-athena.sh
```

Per trasferire esclusivamente i file tracciati da Git, il comando verificato è:

```bash
cd "<local-ds4-checkout>" && git ls-files -z | rsync -avi --from0 --files-from=- ./ -e "ssh -i ~/.ssh/id_ed25519" athena@192.168.254.62:~/DS4-GB10-GX10-DSpark-CUDA/
```

Non aggiungere `-n`: in rsync significa dry-run e non trasferisce alcun file.
Un file sorgente nuovo deve essere tracciato da Git oppure incluso
esplicitamente, altrimenti `git ls-files` non lo invia.

Durante la validazione della pipeline SM121 usare `deploy-athena.sh`: finche' i
nuovi moduli `cuda/indexer/` e `tools/` non vengono tracciati, `git ls-files`
li omette. Lo script trasferisce l'intero albero sorgente, inclusi i
file nuovi, ma esclude `.git`, binari, oggetti e risultati di benchmark.

## Benchmark riproducibile GB10

Dopo `make -B cuda-spark-graph-sm121`, usare un prompt che tokenizzi ad almeno
98.304 token. Il comando seguente esegue cold e append, tre volte ciascuno, e
scrive mediane e dati grezzi sotto `benchmark-results/gb10`:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && python3 tools/benchmark_gb10.py --model /home/athena/ds4/ds4flash.gguf --dspark /home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf --prompt /path/to/same-long-prompt.txt --repeats 3
```

Gli artefatti principali sono `summary.json`, `summary.csv`, `raw.csv` e i log
di ogni processo. Un valore `MISMATCH:` in `greedy_token_hash` blocca il gate;
per il confronto A/B vanno mantenuti identici prompt, binari di riferimento,
frontiere, chunk, draft depth e condizioni termiche.

Per un gate rapido, ma ancora rappresentativo, usare soltanto lo sweep append
alle frontiere 32K, 64K e 96K, due ripetizioni e 128 token greedy per frontiera.
Include prefill corto/medio/lungo, decode DSpark, determinismo e picco RSS e
richiede indicativamente 8–12 minuti su GB10, oltre a un singolo avvio per il
gate qualitativo `ds4-eval`:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && python3 tools/benchmark_gb10.py --model /home/athena/ds4/ds4flash.gguf --dspark /home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf --prompt speed-bench/promessi_sposi.txt --output-dir /tmp/ds4-gb10-quick-gate --frontiers 32768,65536,98304 --gen-tokens 128 --repeats 2 --append-only && column -s, -t /tmp/ds4-gb10-quick-gate/summary.csv
```

Il gate qualitativo breve usa `lm-evaluation-harness` 0.4.10 sul Mac, senza
consumare la RAM unificata di Athena. Con il server gia' avviato, il comando
fissa casi, seed, temperatura e budget di reasoning:

```bash
OPENAI_API_KEY=dummy "$HOME/.venvs/ds4-lm-eval/bin/lm-eval" run --model local-chat-completions --model_args "model=deepseek-v4-flash,base_url=http://192.168.254.62:30007/v1/chat/completions,num_concurrent=1,max_retries=1,tokenized_requests=False,max_length=222822,eos_string=<|endoftext|>" --tasks gsm8k_cot_zeroshot,ifeval --apply_chat_template --samples '{"gsm8k_cot_zeroshot":[0,1,2,3,4,5,6,7,8,9,10,11],"ifeval":[0,1,2,3,4,5,6,7,8,9,10,11]}' --batch_size 1 --seed 1234 --gen_kwargs "max_gen_toks=2200,temperature=0.0" --output_path /tmp/ds4-lm-eval-quick-2200 --log_samples
```

Per questo gate va confrontata la metrica GSM8K `flexible-extract`; il filtro
`strict-match` richiede una forma testuale specifica e non misura la correttezza
del risultato `\\boxed{...}` emesso da DS4.

## Configurazione corrente

`run-dspark-server.sh` è la fonte autorevole dei default. Il profilo
`balanced` imposta:

```text
DS4_CTX=262144
DS4_ADVERTISE_CONTEXT_PCT=85
DS4_MEMORY_PROFILE=balanced
DS4_PREFILL_CHUNK=8192
DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=112
DS4_CUDA_Q8_F16_CACHE_MB=12288
DS4_CUDA_COPY_SECONDARY_MODEL=1
DS4_CUDA_SECONDARY_COPY_PIPELINED=1
DS4_CUDA_DROP_COPIED_MODEL_PAGES=1
DS4_CUDA_DSPARK_CACHE_PRIORITY=1
DS4_CUDA_TOKEN_GRAPH=1
DS4_CUDA_DSPARK_GRAPH=1
DS4_DSPARK_ALWAYS_DRAFT=1
DS4_DSPARK_NO_CIRCUIT_BREAKER=1
DS4_PREFILL_FINAL_LOGITS_ONLY=1
DS4_KV_PREFILL_CHECKPOINT_POLICY=canonical-only
DS4_KV_CACHE_COLD_MAX_TOKENS=262144
DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS=131072
DS4_KV_LONG_COLD_ANCHOR_TRIM_TOKENS=16384
```

Il context guard pubblica circa 222822 token e sottrae anche il budget di
output, 2200 token per default. Questo lascia margine per la compaction del
client prima del limite fisico.

### Profilo sperimentale da 1M token

Il server puo' allocare un context fisico da 1048576 token mantenendo
l'advertise all'85%. Sul GB10 questo profilo deve usare chunk 4096: chunk 8192
porta i context buffer da circa 10,36 a 14,78 GiB e ha prodotto OOM durante
l'avvio con cache Q8->F16 da 12 GiB.

Fermare prima il server corrente e avviare il launcher dedicato:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./run-dspark-server-1m.sh 2>&1 | tee /tmp/ds4-1m.log
```

`run-dspark-server-1m.sh` fissa context 1048576, chunk 4096, advertise 85%,
anchor 524288 e trim 65536. Usa per default la KV cache isolata
`/tmp/ds4-gb10-dspark-1m-kv` e abilita la telemetria; il percorso puo' essere
cambiato con `DS4_EXPERIMENT_KV_DIR` e la telemetria disabilitata con
`DS4_TELEMETRY=0`. Il server pubblica 891289 token e, con il budget di output
predefinito, accetta al massimo circa 889089 token di input.

Il profilo e' una modalita' capacity-first: il chunk dimezzato e la crescita
dell'attenzione riducono il prefill rispetto al default 256K/8192. Non
impostare manualmente chunk 8192 sul context 1M: nella configurazione GB10
misurata ha prodotto OOM. Durante una prima prova il server ha superato 163K
token reali senza errori CUDA o fallback; la validazione completa a profondita'
maggiori resta separata dal profilo operativo consigliato.

Per tornare al profilo predefinito 256K non impostare le tre variabili
precedenti:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-dspark-server.log
```

## Criteri di accettazione

Con lo stesso prompt, stessa posizione assoluta e server appena avviato:

1. `cuda-regression` deve terminare con `OK`;
2. nessun BOS inatteso, non-finite o errore CUDA;
3. qualità della risposta coerente con il target precedente;
4. almeno `440 t/s` medi sul test da 58,5K, oppure almeno `469 t/s` sul
   tratto 24,5K–57,3K;
5. decode DSpark vicino al riferimento di `23 t/s` e comunque non sotto
   18–20 t/s;
6. aumento memoria dovuto al token-tile non oltre circa 168 MiB;
7. secondo turno append/canonical senza full prefill quando il prefisso è
   riutilizzabile.

Per la pipeline SM121 in attesa di validazione si aggiungono questi gate:

8. stesso hash greedy fra le tre ripetizioni deterministiche;
9. RAM RSS/HWM non superiore al baseline e nessun aumento permanente dei pesi;
10. scorer indexer almeno 1,6x e indexer combinato almeno +10% end-to-end;
11. MoE almeno +10% end-to-end, token-tile attention almeno 1,5x;
12. decode DSpark almeno al 98% della mediana baseline; GVR resta solo con
    vantaggio superiore al 3% sul deep decode.

## Cronologia tecnica

Le sezioni seguenti documentano l'evoluzione del lab. Quando un valore storico
contrasta con **Avvio rapido**, prevalgono sempre i default correnti riportati
all'inizio del documento e in `run-dspark-server.sh`.

### Esperimento scartato: LiteTopK ratio-4

Il 18 luglio e' stato implementato e verificato un percorso LiteTopK esatto
per il prefill ratio-4, attivo da `n_comp >= 16384`. Usava un campione di 1536
righe frequenti, soglia conservativa a 256 bin, scan single-owner, selezione
radix della frontiera e repair GPU batched. La regressione confermava lo stesso
set Top-K da 512 elementi e il run non mostrava regressioni qualitative o di
decode DSpark, rimasto intorno a 22,4 t/s.

Il gate prestazionale non e' stato superato. Sullo stesso chunk assoluto
73728..81920 il baseline token-tile ha misurato 461,96 t/s, mentre LiteTopK ha
misurato 424,13 t/s, pari a -8,2%. Nei primi chunk dopo l'attivazione la
regressione arrivava a circa -22%. Non e' emersa neppure una riduzione
materiale della memoria residente, perche' la matrice score completa restava
allocata per fallback e repair.

L'audit ha mostrato che il prototipo eliminava la materializzazione completa
degli score ma continuava a calcolare l'intero prodotto Q x KV. A questo
aggiungeva uno score pass separato sul campione, conteggi atomici delle
frequenze, selezione radix e record candidati a 64 bit. La scan riduceva inoltre
la griglia da molte CTA brevi a 512 CTA lunghe che serializzavano le tile KV.
Indexer, Top-K e token-tile attention restavano kernel separati, quindi mancava
la fusione che avrebbe dovuto compensare questi costi.

L'implementazione e' stata rimossa integralmente. La strada va riaperta solo
con una pipeline Blackwell realmente fusa, basata su TMA/TMEM, MMA asincrono,
warp specializzati e passaggio diretto alla sparse attention, e soltanto con un
guadagno end-to-end superiore al 10% a parita' di qualità, memoria e decode.

### Esperimento scartato: Packed QAT E2M1 con HMMA F16

Il 18 luglio e' stato provato un percorso prefill-only che conservava le query
QAT in forma packed. Ogni riga da 128 valori passava da 512 byte F32 a 80 byte:
64 byte di codici E2M1 FP4 e quattro scale FP32. Lo scorer espandeva i valori
direttamente nella tile F16 in shared memory e lasciava invariati K, HMMA,
ReLU, routing weights, causal mask e Top-K. I batch sotto 128 token, incluso il
decode e il verifier DSpark, restavano sul percorso precedente; il buffer
packed riusava lo scratch token-tile senza aumentare il context buffer.

La correttezza numerica era esatta nel test CUDA:

```text
cuda-regression: packed QAT indexer scores different=0 rel-rmse=0.000000000 max-abs=0.000000000
```

Il gate prestazionale e' pero' fallito nettamente sullo stesso prompt freddo da
25.280 token, con lo stesso cold anchor a 24.576:

```text
Intervallo ctx          Baseline    Packed QAT    Delta
0..8192                 687.28 t/s  511.33 t/s   -25.6%
8192..16384             680.90 t/s  598.84 t/s   -12.1%
16384..24576            635.66 t/s  568.92 t/s   -10.5%
Media 0..24576          667.13 t/s  557.27 t/s   -16.5%
Richiesta 0..25280      633.34 t/s  532.79 t/s   -15.9%
```

Il tempo totale e' salito da 39,916 a 47,448 secondi, pari a circa +18,9%.
Il decode non mostrava un cedimento strutturale e ha raggiunto 21,44 t/s in un
turno successivo, ma il deficit prefill rendeva inutile proseguire il test.

L'esito indica che lo scorer corrente non e' limitato dalla lettura DRAM delle
query: la tile Q F32 viene gia' riutilizzata efficacemente dalla L2 mentre le
CTA attraversano le tile compresse. Il packing aggiunge invece una passata
completa di lettura/scrittura e, soprattutto, dequantizzazione E2M1 software,
shuffle, scale e pressione sui registri dentro ogni CTA. L'HMMA F16 corrente
non consuma FP4 direttamente, quindi il costo di espansione supera il risparmio
di banda. L'implementazione e il relativo self-test sono stati rimossi
integralmente. Questa strada va riaperta solo con MMA FP4 block-scaled nativo o
un kernel CUTLASS equivalente realmente fuso, non con dequantizzazione software
anteposta allo stesso HMMA.

### 1. Modello residente nella memoria della GB10

`DS4_CUDA_COPY_MODEL=1` copia il modello sulla memoria del device invece di
leggere continuamente i pesi dal mapping host. Il limite è impostato a 96 GiB,
sufficiente per il modello da circa 80,76 GiB e con margine per gli altri
buffer.

Effetti osservati:

- memoria GPU del processo: circa 82.885 MiB prima delle cache aggiuntive;
- RSS host osservato: circa 30 GiB;
- avvio sensibilmente più lento, perché la copia viene eseguita prima di aprire
  la porta;
- esecuzione più stabile, senza dipendenza dal direct I/O durante il decode.

### 2. Cache selettiva Q8 → FP16

`DS4_CUDA_Q8_F16_CACHE_MB=12288` porta il limite della cache a 12 GiB ed evita
che si fermi a 4 GiB con il messaggio `cache limit reached`.

Nota importante: nel codice corrente il percorso cuBLAS basato sulla cache
FP16 è usato soprattutto con `n_tok > 1`, quindi è utile principalmente nel
prefill e nei matmul batch. Il decode autoregressivo a token singolo continua
in molti casi a usare i kernel Q8 nativi; non bisogna attribuire automaticamente
alla cache FP16 un aumento del decode.

### 3. Sincronizzazione finale differita

Con `DS4_CUDA_DEFER_END_SYNC=1` viene eliminato un
`cudaDeviceSynchronize()` ridondante alla fine del token. La successiva lettura
sincrona dei logits garantisce già il completamento del default stream.

La modifica è:

- attiva solo sul backend CUDA;
- attiva solo quando i logits vengono letti;
- disabilitata durante profiling e power throttling;
- compatibile con il percorso normale e con CUDA Graph.

### 4. Nessun flush intermedio dopo i primi layer

`DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0` disabilita il vecchio split/flush dopo i
primi quattro layer. Sulla singola GB10 non c'è lavoro CPU utile da sovrapporre
che compensi la sincronizzazione anticipata.

Il nome della variabile contiene `METAL` per ragioni storiche, ma il controllo
si trova nel percorso graph condiviso usato anche da CUDA.

### 5. Compressor ratio-4 fuso

`DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1` abilita un kernel CUDA specializzato per
l'aggiornamento del compressor ratio-4 durante il decode.

Il kernel unisce:

- inserimento della nuova riga nello stato;
- pooling quando il token emette una riga compressa;
- RMS norm;
- RoPE;
- avanzamento dello stato ricorrente.

Questo elimina diversi piccoli kernel con dipendenze strettamente seriali. Il
percorso generico resta disponibile non impostando la variabile d'ambiente.

Log atteso:

```text
ds4: CUDA fused ratio4 compressor update enabled
```

### 6. CUDA Graph a livello di token

`DS4_CUDA_TOKEN_GRAPH=1` cattura l'intero decode di un token e lo lancia come
CUDA Graph. DeepSeek V4 Flash ha tre topologie perché il compressor emette:

1. token normale;
2. emissione ratio-4;
3. emissione ratio-128.

Il primo token viene eseguito normalmente per inizializzare allocator, arena e
stato CUDA. I token successivi vengono catturati, aggiornati e lanciati come
graph. Se cattura o istanziazione falliscono, lo stato host viene ripristinato
e il token viene rieseguito sul percorso normale.

Sono inoltre gestiti:

- rilascio dei graph prima di liberare i tensor CUDA;
- cambio di topologia tra richieste o contesti differenti;
- contatori di launch, update e rebuild;
- compatibilità CUDA 13 e fallback per runtime precedenti.

Log osservati:

```text
ds4: CUDA token graph variant=0 nodes=1508 instantiated
ds4: CUDA token graph variant=1 nodes=1550 instantiated
ds4: CUDA token graph variant=2 nodes=1630 instantiated
ds4: CUDA token graph launches=1000 updates=994 rebuilds=6
```

I sei rebuild sono coerenti con tre varianti iniziali e tre ricostruzioni al
passaggio a una richiesta con una topologia più grande. I 994 update mostrano
che, una volta stabilizzata la forma, il graph non viene ricostruito a ogni
token.

### 6.1 Pipeline look-ahead del token graph

`DS4_CUDA_TOKEN_GRAPH_PIPELINE=1` toglie cattura e
`cudaGraphExecUpdate()` dal cammino critico. Mentre la GPU esegue il token N,
un thread prepara su un secondo per-thread stream il graph esatto per la
posizione N+1. La posizione, le righe KV e le dimensioni dell'attenzione sono
già note; il solo dato non ancora noto è il token campionato. L'embedding legge
quel singolo ID da uno scalare CUDA aggiornato subito prima del launch.

Sono usati 12 slot: tre topologie, due parità e due modalità di output
(full-logits o argmax). La parità garantisce che il background non aggiorni mai
l'eseguibile ancora in volo. Prima del direct launch vengono confrontati
posizione, puntatori HC e tutti i contatori compressor/indexer; a ogni mismatch
si torna automaticamente alla cattura sincrona corretta.

Log opzionale con `DS4_CUDA_TOKEN_GRAPH_PIPELINE_VERBOSE=1`:

```text
ds4: CUDA token graph look-ahead pipeline enabled (late-bound token, exact position/KV state)
ds4: CUDA token graph direct launches=1000 total=... prepares=... updates=... rebuilds=...
```

La pipeline è volutamente disattivata quando sono attivi timing dettagliato o
range Nsight, perché entrambi possiedono bookkeeping globale della cattura.

### 6.2 Argmax CUDA e logits lazy

`DS4_CUDA_GREEDY_ARGMAX=1` aggiunge l'argmax al token graph e legge soltanto
quattro byte invece dell'intero vocabolario. Si attiva esclusivamente per una
richiesta interamente greedy: temperatura zero, niente tools, niente thinking
con temperatura dinamica e niente MTP. Per richieste probabilistiche il
sampling CPU e i logits restano invariati.

I logits completi rimangono autorevoli sul device e vengono materializzati una
sola volta, on demand, se servono logprobs, sampling probabilistico, snapshot KV
o una richiesta successiva. L'argmax GPU usa lo stesso tie-break sull'indice più
basso del riferimento CPU.

### 6.3 Attention decode dinamica heads2 (scartata)

`DS4_CUDA_ATTN_HEADS2=1` abilita una variante specializzata per il decode a
token singolo. DeepSeek V4 Flash ha 64 query head e una sola KV head. Il kernel
normale assegna una query head a ciascuno dei 64 blocchi. La variante assegna
due query head a ciascuno di 32 blocchi e carica le stesse righe KV da 512
float una sola volta in shared memory prima di usarle con le due query
differenti.

La selezione era dinamica: `DS4_CUDA_ATTN_HEADS2_MIN_ROWS=384` manteneva il
kernel originale a 64 blocchi sotto la soglia e attivava heads2 sopra la
soglia, purché raw + compressed/top-k non superassero le 768 righe della tape
shared. L'indexed attention continuava a mantenere tutte le 512 righe top-k
previste dal modello.

```text
ds4: CUDA dynamic decode attention heads2 enabled (min_rows=384, score_cap=768)
```

Il log dimostra che non è avvenuto un fallback: CUDA Graph ha continuato a
preparare e lanciare normalmente le proprie varianti, mentre il backend ha
selezionato esplicitamente prima `path=dense rows=384` e poi
`path=indexed rows=640`. L'A/B interno alla stessa richiesta ha mostrato:

- circa `15,5–16,1 t/s` prima della soglia;
- `14,50 t/s` nel primo chunk successivo all'attivazione a 384 righe;
- `12,09 t/s` stabili sul percorso indexed a 640 righe;
- regressione di circa `-17,7%` rispetto al riferimento lungo di `14,689 t/s`.

Sulla GB10 il dimezzamento dei blocchi riduce occupancy e capacità di
nascondere la latenza. Ogni blocco esegue inoltre due accumulazioni indipendenti
con maggiore pressione su registri e shared memory. Il risparmio teorico delle
letture KV non compensa questa perdita, probabilmente perché il kernel a 64
blocchi beneficia già del riuso nella cache L2.

La trasformazione non modifica pesi, KV, numero di head o righe selezionate,
ma il risultato prestazionale è sufficiente per scartarla. Disabilitazione
senza ricompilare:

```bash
DS4_ENABLE_ATTN_HEADS2=0 ./run-experimental-server.sh
```

Non sono consigliati ulteriori test con soglie o un numero maggiore di head
per blocco: il collo di bottiglia osservato è il parallelismo perso, non la
soglia di selezione.

### 6.4 Benchmark per richiesta

Ogni risposta emette una riga stabile:

```text
ds4-server: decode summary req=... prompt=... gen=... seconds=... tps=... greedy_gpu=... finish=...
```

Analisi consigliata (salta la prima richiesta di warm-up e ignora risposte con
meno di 50 token):

```bash
./analyze-decode-log.sh /tmp/ds4-gb10-pipeline.log 1 50
```

Il risultato principale è `Weighted throughput = token totali / secondi
totali`; non viene più calcolata la media dei chunk, che può esplodere su un
ultimo intervallo molto corto.

### 7. Telemetria CUDA Graph opt-in

`DS4_CUDA_TOKEN_GRAPH_TIMING=1` abilita una misura aggregata del percorso di
decode senza stampare una riga per token. L'intervallo predefinito è 500 token
e può essere regolato con `DS4_CUDA_TOKEN_GRAPH_TIMING_EVERY`.

La telemetria separa:

- avvio della cattura;
- registrazione host dei kernel;
- `cudaStreamEndCapture`;
- `cudaGraphExecUpdate` o rebuild;
- bookkeeping e submit del graph;
- esecuzione GPU tramite CUDA Event;
- attesa e copia dei logits;
- sampling CPU;
- latenza totale eval + sampling.

Gli intervalli vengono chiusi automaticamente quando la posizione del contesto
salta tra due richieste, evitando di mescolare il test corto con quello lungo.
`read_wait` include l'attesa della GPU; `read_tail_est` sottrae il tempo GPU ed
è quindi soltanto una stima della coda di readback.

Esempio del log atteso:

```text
ds4: CUDA token timing reason=interval tokens=500 pos=365..864 begin=... encode=... end_capture=... update=... rebuild=... bookkeeping=... launch=... gpu=... read_wait=... read_tail_est=... eval=... sampling=... total=... updates=... rebuilds=... samples=...
```

La telemetria è disattivata nel normale comando di produzione e aggiunge due
CUDA Event per token soltanto quando esplicitamente richiesta.

### 8. Finestra Nsight Systems opt-in

Per analizzare i kernel del decode senza profilare gli oltre 80 GiB di copia e
preparazione del modello, una build `cuda-spark-graph` può delimitare una sola
finestra tramite la CUDA Profiler API:

```bash
DS4_CUDA_NSYS_CAPTURE_START_POS=6000
DS4_CUDA_NSYS_CAPTURE_TOKENS=20
```

Il primo parametro indica la posizione di contesto minima da cui avviare la
raccolta; il secondo il numero di token completi da includere. La raccolta
parte immediatamente prima della cattura del CUDA Graph e termina dopo il
readback dei logits dell'ultimo token. Il trigger è one-shot per processo e
non richiede `DS4_CUDA_TOKEN_GRAPH_TIMING`, così i CUDA Event della telemetria
non perturbano il profilo Nsight.

Il server deve essere eseguito sotto `nsys profile` con:

```text
--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --kill=none --cuda-graph-trace=node
```

Comando completo, mantenuto su una sola riga per evitare il prompt `>`:

```bash
cd /tmp/ds4-gb10-lab && /usr/local/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --kill=none --cuda-graph-trace=node --force-overwrite=true -o /tmp/ds4-gb10-pos6000 /usr/bin/env DS4_CUDA_COPY_MODEL=1 DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=96 DS4_CUDA_Q8_F16_CACHE_MB=12288 DS4_CUDA_DEFER_END_SYNC=1 DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0 DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1 DS4_CUDA_TOKEN_GRAPH=1 DS4_CUDA_TOKEN_GRAPH_VERBOSE=1 DS4_CUDA_NSYS_CAPTURE_START_POS=6000 DS4_CUDA_NSYS_CAPTURE_TOKENS=20 ./ds4-server --cuda -m /home/athena/ds4/ds4flash.gguf -c 131072 -n 2200 -t 10 --host 0.0.0.0 --port 30007 --kv-disk-dir /tmp/ds4-gb10-experiment-kv --kv-disk-space-mb 65536
```

`--cuda-graph-trace=node` introduce overhead intenzionalmente: i token/s del
run profilato non vanno confrontati con il benchmark normale. Il suo scopo è
attribuire i circa 69 ms GPU ai singoli kernel. Dopo la chiusura della finestra:

```bash
/usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum /tmp/ds4-gb10-pos6000.nsys-rep
```

Log attesi:

```text
ds4: CUDA Nsight capture started pos=6000 tokens=20
ds4: CUDA Nsight capture stopped after 20 tokens reason=window-complete
```

Per profilare il decode DSpark stabile sullo stesso workload senza includere il
warm-up dei graph, usare lo script dedicato. Il default esegue il prefill fino a
65.536 token, lascia 128 token al warm-up, cattura i 64 successivi e produce
automaticamente sia il `.nsys-rep` sia le statistiche NVTX/kernel/API/memoria:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./profile-nsys-decode.sh 2>&1 | tee /tmp/ds4-nsys-decode.log
```

Prima del lancio non devono essere attivi `ds4-server` o `ds4-bench`. Le
dimensioni possono essere cambiate con `DS4_NSYS_DECODE_FRONTIER`,
`DS4_NSYS_DECODE_WARMUP` e `DS4_NSYS_DECODE_TOKENS`. Il trace usa
`--cuda-graph-trace=node`, quindi serve per attribuire il tempo e non per
confrontare direttamente i token/s con il run di produzione.

### 8a. Trace NVTX one-shot del prefill

Il backend CUDA include range NVTX gerarchici che non modificano il calcolo e
non introducono sincronizzazioni. Sono inattivi nel server ordinario e vengono
abilitati con `DS4_CUDA_NVTX=1`. Impostare
`DS4_CUDA_NSYS_PREFILL_START_POS` abilita automaticamente NVTX e delimita con
la CUDA Profiler API un solo chunk: viene scelto il primo chunk il cui `pos0`
e' maggiore o uguale alla posizione richiesta. DS4 non riserva buffer host o
device, non crea eventi CUDA e non copia tensori per questa telemetria; lo
spazio del file `.nsys-rep` e' gestito esternamente da Nsight Systems.

Per confrontare piu' profondita' senza ricaricare il modello, la variabile
`DS4_CUDA_NSYS_PREFILL_START_POSITIONS` accetta fino a 16 posizioni strettamente
crescenti separate da virgole. Ogni posizione cattura un solo chunk. Nsight va
lanciato con `--capture-range-end=repeat-shutdown:N`, dove `N` e' il numero di
finestre, e genera un report separato per ciascuna. La forma singola precedente
resta compatibile.

I range principali sono:

```text
ds4/prefill/chunk
ds4/prefill/attention/{raw,ratio4,ratio128}
ds4/prefill/indexer/{score,score-mxfp4,topk}
ds4/prefill/attention/token_tile/{indexed,dense,visible_rows,raw_mirror,comp_mirror,hmma}
ds4/prefill/attention/flashmla/hmma
ds4/prefill/ffn
ds4/prefill/moe/{routed,mmq_fused,expert_map,input_quant_q8_1,iq2_gate_up_d2r,iq2_gate,iq2_up,swiglu_down_quant,q2_down,sum}
```

Per catturare, per esempio, il chunk che parte da 32768 token, avviare il
server sotto Nsight Systems e poi inviare da un altro terminale un prompt che
superi 40960 token:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --kill=none --force-overwrite=true -o /tmp/ds4-prefill-pos32768 /usr/bin/env DS4_CUDA_NSYS_PREFILL_START_POS=32768 ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-prefill-pos32768.log
```

Log attesi:

```text
ds4: CUDA Nsight prefill capture started pos=32768 tokens=8192
ds4: CUDA Nsight prefill capture stopped pos=32768 tokens=8192 reason=chunk-complete
```

I tre riepiloghi utili si ottengono in una sola riga:

```bash
/usr/local/cuda/bin/nsys stats --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum /tmp/ds4-prefill-pos32768.nsys-rep
```

`nvtx_gpu_proj_sum` attribuisce il tempo GPU ai macro-stage; `nvtx_kern_sum`
mostra quali kernel compongono ciascun range; `cuda_gpu_kern_sum` resta il
controllo indipendente per il totale dei kernel. Payload NVTX e nomi sono
statici: non vengono formattate stringhe nel percorso caldo. I token/s del run
profilato non sono un benchmark, mentre il run senza le variabili diagnostiche
mantiene NVTX disattivato.

Per il confronto riproducibile del decadimento a contesto lungo, lo script
`profile-nsys-prefill.sh` usa `ds4-bench`, il testo fisso dei Promessi Sposi e
la configurazione balanced del launcher. Con un solo caricamento acquisisce i
chunk `0`, `8192`, `32768`, `65536` e `98304`: il secondo separa il costo
one-shot del primo chunk dal primo percorso sparse. Per ciascun report estrae
range NVTX, kernel, CUDA API e copie GPU:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./profile-nsys-prefill.sh 2>&1 | tee /tmp/ds4-prefill-depth.log
```

Prima del lancio non devono essere attivi `ds4-server` o `ds4-bench`. I report
sono `/tmp/ds4-prefill-depth*.nsys-rep` e i riepiloghi corrispondenti terminano
in `.stats.txt`. I cinque report vanno interpretati nell'ordine delle finestre
stampato nel log; i token/s sotto profiler non costituiscono un benchmark.

### 9. Nsight Compute e permessi dei contatori

`ncu` usa i performance counter hardware, che sul driver NVIDIA 580 della
Spark sono riservati agli utenti amministrativi. Se il run mostra
`ERR_NVGPUCTRPERM`, il trigger e il filtro kernel possono essere corretti ma
non viene prodotto alcun file `.ncu-rep`. Per un test isolato si esegue
soltanto `ncu` con `sudo`; non è necessario cambiare i moduli NVIDIA o
abilitare permanentemente i contatori per tutti gli utenti. Il server
profilato usa una directory KV temporanea separata per non lasciare file di
root nella cache sperimentale ordinaria. Deve inoltre impostare un lock
separato, per esempio `DS4_LOCK_FILE=/tmp/ds4-ncu-root.lock`: il lock normale
`/tmp/ds4.lock` appartiene a `athena` e le protezioni dei file in `/tmp`
possono impedirne la riapertura al processo lanciato da `sudo`. Anche i
token/s osservati sotto `ncu` contengono overhead del profiler e non sono un
benchmark.

Lo script `profile-ncu-prefill.sh` profila una sola istanza reale del kernel
HMMA indexed a circa 98K. Usa un chunk di 1024 token: il lavoro interno di ogni
token-tile e la profondita' della KV restano rappresentativi del contesto
lungo. La modalita' predefinita `occupancy` raccoglie soltanto attributi di
lancio, registri, shared memory e occupancy. Usa application replay anche per
queste sezioni: NCU attiva comunque il proprio meccanismo di replay e, con
kernel replay, puo' tentare lo snapshot delle allocazioni CUDA pur senza una
raccolta estesa di performance counter:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && sudo -E ./profile-ncu-prefill.sh 2>&1 | tee /tmp/ds4-ncu-prefill-indexed.log
```

Il kernel replay non va usato con il modello residente: NCU tenta di
salvare e ripristinare le allocazioni CUDA accessibili e, sulla memoria
unificata quasi piena della Spark, puo' fallire a `0%` con codice applicazione
`9`. Dopo il passaggio occupancy, le metriche runtime si raccolgono con
application replay e buffer su file. Questa modalita' riavvia e ricarica il
modello per ogni passata necessaria, quindi e' molto piu' lenta ma non richiede
una seconda copia dello stato CUDA residente:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && sudo -E env DS4_NCU_PREFILL_MODE=runtime ./profile-ncu-prefill.sh 2>&1 | tee /tmp/ds4-ncu-prefill-runtime.log
```

Il percorso esatto del report `.ncu-rep` e della relativa esportazione `.txt`
viene stampato all'avvio e alla fine. Per profilare separatamente il percorso
dense, senza confonderne i contatori con quello indexed, si parte ancora dal
passaggio occupancy:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && sudo -E env DS4_NCU_PREFILL_RANGE=dense ./profile-ncu-prefill.sh 2>&1 | tee /tmp/ds4-ncu-prefill-dense.log
```

## Risultati osservati

I valori seguenti provengono dai log del server e non da un benchmark
scientifico con più ripetizioni. Il dato più confrontabile è il decode lungo
intorno a 5.700 token di contesto.

| Configurazione | Contesto iniziale | Decode medio | Note |
|---|---:|---:|---|
| Configurazione precedente | ~5.705 | 13,78 t/s | 1.675 token generati |
| CUDA Graph + compressor fuso | 361 | 15,46 t/s | 301 token generati |
| CUDA Graph + compressor fuso | ~5.708 | 14,22 t/s | almeno 1.000 token stabili |
| Graph + controlli differiti | ~5.711 | 14,22 t/s | nessun vantaggio; patch rimossa |
| Cubin nativo `sm_121` | ~5.708 | 13,99 t/s | circa -1,6%; target conservato solo per esperimenti |
| F16 decode coalescente | ~5.711 | 14,53 t/s | circa +2,2%; 1.350 token osservati |
| F16 coalescente + Q8 U16/warp quant | ~5.708 | 14,58 t/s | circa +0,4% sul test F16; 1.997 token |
| Token graph pipeline, heads2 disattivato | ~5.876 | **14,689 t/s** | 1.316 token; riferimento corrente |
| Attention heads2 indexed | ~5.876 | **12,09 t/s** | 1.150+ token; circa -17,7%, scartata |

### Profilo Nsight Systems a posizione 6000

Una finestra di 20 token con tracing dei singoli nodi CUDA Graph ha registrato
circa 92,04 ms/token di tempo kernel sommato. Il tracing `node` introduce
overhead, quindi sono affidabili soprattutto le proporzioni:

| Gruppo | Tempo tracciato per token | Quota |
|---|---:|---:|
| MoE gate/up, grouped e down | 27,81 ms | 30,2% |
| Proiezioni Q8 e HC | 28,65 ms | 31,1% |
| Proiezioni F16 | 17,57 ms | 19,1% |
| Attention indexed + decode | 8,49 ms | 9,2% |
| Norm e hyperconnection | 4,21 ms | 4,6% |
| Resto | 5,32 ms | 5,8% |

Il precedente valore elevato di `attn_hc_pre` comprendeva lavoro CUDA già in
coda e non identificava un singolo kernel. Nsight mostra invece che circa
l'80% del percorso è costituito da GEMV/proiezioni quantizzate e F16. Il primo
candidato di codice, dopo il test del cubin nativo, è il percorso F16 a singolo
token: i kernel `ordered_chunks` assegnano segmenti contigui a lane diverse e
generano accessi warp non coalescenti.

### F16 decode coalescente opt-in

`DS4_CUDA_COALESCED_F16_MATMUL=1` seleziona due kernel alternativi soltanto
per il decode a token singolo. Ogni blocco usa otto warp e ogni warp calcola
una riga leggendo elementi adiacenti tra le lane. La variante pair continua a
calcolare due matrici nello stesso kernel e a riusare l'attivazione.

Il percorso predefinito `ordered_chunks` non è stato rimosso. La nuova
riduzione cambia l'ordine delle somme FP32 e può quindi produrre differenze
minime nei logits; per questo l'abilitazione resta esplicita e facilmente
reversibile. Log atteso:

```text
ds4: CUDA coalesced warp8 F16 decode matmul enabled
```

Il confronto va eseguito con il target `cuda-spark-graph` precedente, non con
il cubin `sm_121`, per isolare una sola variabile rispetto al riferimento di
14,22 t/s.

Nel test del 13 luglio il percorso coalescente si è stabilizzato a 14,53 t/s
contro 14,22 t/s, riducendo la latenza da circa 70,32 a 68,82 ms/token. Il
guadagno osservato è quindi circa 0,31 t/s, pari al 2,2% complessivo.

### Caricamenti Q8 U16 per DP4A opt-in

`DS4_CUDA_Q8_U16_LOADS=1` sostituisce nel primitivo DP4A condiviso la
ricostruzione di ogni parola da quattro letture U8 con due letture U16
naturalmente allineate. La modifica raggiunge contemporaneamente i kernel
Q8 single, pair, HC expand e grouped senza cambiare pesi, scale, ordine delle
somme o operandi consegnati a `__dp4a`.

All'avvio viene eseguita sul device una validazione one-shot di 32 indirizzi,
comprendendo sia offset allineati a 32 bit sia offset allineati soltanto a
16 bit. In caso di qualsiasi differenza o errore CUDA il server torna
automaticamente ai byte load originali. Log atteso:

```text
ds4: CUDA Q8 U16 DP4A loads enabled (32/32 bit-exact samples)
```

La funzione che quantizza le attivazioni Q8 usa inoltre una riduzione
warp-shuffle al posto di shared memory e cinque barriere. La sequenza `fmax`
vista dalla lane zero resta la stessa, quindi la scala e gli int8 prodotti non
cambiano. Questa seconda ottimizzazione è sempre attiva nella nuova build e
resta indipendente dal flag U16.

Nel test lungo il caricamento U16 insieme alla nuova riduzione del quantizer ha
portato la media da 14,53 a circa 14,58 t/s, riducendo la latenza soltanto da
68,82 a circa 68,59 ms/token. Il miglioramento è quindi piccolo, circa 0,4%, e
non conferma l'ipotesi che la ricostruzione U8 fosse un collo di bottiglia
importante. Il flag resta opt-in e bit-identico.

### Kernel MoE decode dedicato GB10 opt-in

`DS4_CUDA_MOE_DECODE_GB10=1` seleziona una variante del solo kernel
`moe_gate_up_mid_decode_lut_qwarp32_kernel` per il decode Flash
4096→2048 con sei esperti. Il percorso precedente resta il fallback e viene
usato automaticamente per qualunque altra geometria.

La variante elimina tre lavori ripetuti osservabili direttamente nel sorgente:

- il byte `cuda_ksigns_iq2xs[i]` viene calcolato da `i` e dal suo bit di
  parità invece di essere riletto casualmente dalla shared memory; la proprietà
  viene verificata all'avvio su tutti i 128 elementi;
- i 16 blocchi di attivazioni Q8 vengono copiati in shared da tutti i 256
  thread su parole contigue, invece di assegnare un intero blocco da 292 byte
  a ciascuno di soli 16 thread;
- la LUT grid viene prima duplicata byte per byte in global memory e poi
  caricata in shared con accessi coalescenti. Il row span passa inoltre da 128
  a 256, riducendo da 96 a 48 i blocchi del gate/up per layer e dimezzando il
  costo di staging.

Per ogni riga restano invariati lane assignment, ordine dei prodotti DP4A,
riduzione quarter-warp e operazioni FP32 finali. La modifica è quindi pensata
come bit-identica, non come approssimazione numerica. Se la validazione o il
setup della LUT falliscono, il flag viene ignorato. Log atteso:

```text
ds4: CUDA GB10 MoE decode enabled (computed IQ2 signs, row span=256, 128/128 exact)
```

Il primo confronto va eseguito insieme alle ottimizzazioni già confermate,
contro il riferimento lungo di circa 14,58 t/s. Per il rollback è sufficiente
rimuovere `DS4_CUDA_MOE_DECODE_GB10=1` dal comando, senza ricompilare.

Sul confronto lungo il passaggio da 13,78 a 14,22 t/s vale circa **+3,2%** e
riduce la latenza da circa 72,57 ms/token a 70,32 ms/token.

Nello stesso avvio è stato osservato un prefill di 83 token a 121,58 t/s,
contro valori precedenti nell'ordine di 77–85 t/s. Questo dato non è ancora un
A/B isolato e non va attribuito interamente a CUDA Graph, che ottimizza
principalmente il decode.

### Prompt di confronto

Per confrontare richieste omogenee è stato usato:

```text
Spiega in italiano come funziona la memoria virtuale di Linux.
Descrivi paging, page fault, swap, TLB e differenza tra memoria virtuale e RAM.
Fornisci un esempio pratico e concludi con cinque punti riassuntivi.
Rispondi in modo dettagliato, generando almeno 500 token.
```

È importante confrontare lo stesso prompt, lo stesso contesto iniziale e una
generazione sufficientemente lunga. Un semplice `ciao` è troppo corto e viene
influenzato da warm-up, thinking e variazioni del sampler.

## Tentativi scartati o non conclusivi

### Stream FFN parallelo

Il tentativo di eseguire parti della FFN/MoE su stream CUDA paralleli ha ridotto
le prestazioni: sono stati osservati circa 12,0–12,9 t/s invece di 13,78 t/s.
La patch è stata rimossa. Non impostare:

```text
DS4_CUDA_PARALLEL_FFN
DS4_CUDA_PARALLEL_FFN_VERBOSE
```

La lezione è che i rami FFN condividono risorse e banda sulla singola GB10; il
costo di coordinamento tra stream supera il parallelismo disponibile.

### Cambio del percorso direct-down MoE

Il test con `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6` non ha prodotto un miglioramento
misurabile. Il percorso MoE non è quindi il candidato prioritario per la
prossima modifica minimale.

### Albero Markov tardivo su K3

Il 20 luglio e' stato provato un secondo candidato Markov soltanto dopo
`t1/t2` accettati e il rifiuto di `t3`. Il verifier target principale restava
lineare; il ramo conservava il residuo esatto `max(p-q1,0)`, campionava
un'alternativa senza replacement e, se accettata, eseguiva un singolo replay
target dalla frontier dopo `t2`. La regressione CUDA sul doppio rifiuto e sulla
seconda correzione residua e' passata, quindi la distribuzione target era
preservata.

Nel test end-to-end tra 9K e 66K token il decode e' rimasto nel profilo atteso,
ma non e' emerso un incremento materiale. Il ramo interviene solo nel
sottoinsieme dei cicli K3 che falliscono esattamente alla terza proposta, mentre
un'alternativa accettata paga comunque un decode target aggiuntivo. Il rapporto
beneficio/complessita' non era quindi sufficiente e l'intervento e' stato
rimosso integralmente. Per migliorare il decode bisogna ridurre il costo del
drafter/verifier gia' eseguito in tutti i cicli, non aggiungere lavoro dopo un
fallimento raro.

### Aumentare soltanto la cache FP16

Portare la cache da 4 a 12 GiB elimina il limite e può aiutare il prefill, ma
non ha mostrato da solo un miglioramento consistente del decode a token
singolo. Aumentarla ulteriormente senza cambiare il kernel rischia soprattutto
di consumare memoria utile a contesto e workspace.

## KV disk cache e memoria

`--kv-disk-space-mb` assegna un budget massimo alla cache su disco; non
prealloca necessariamente quella dimensione e non e' memoria GPU. Il launcher
GB10 usa `DS4_KV_DISK_SPACE_MB=16384`, cioe' 16 GiB, per evitare che decine di
checkpoint da 1-2 GiB restino candidati alla page cache della memoria unificata.
Per test A/B o recupero sessioni lunghe si puo' rialzare temporaneamente, per
esempio `DS4_KV_DISK_SPACE_MB=65536`.

Questa directory contiene checkpoint/snapshot persistenti usati all'inizio di
una richiesta per evitare parte del prefill. Non è la KV attiva letta dai
kernel attention a ogni token. La KV attiva viene allocata come tensor CUDA;
con la configurazione corrente è inclusa nei circa 3.112 MiB di context buffer
ed è già nella memoria accessibile direttamente dalla GB10. Solo per contesti
molto più grandi il backend può scegliere CUDA managed memory per evitare OOM.

Su Athena `/tmp` appartiene al filesystem NVMe principale, non a un tmpfs. Il
kernel Linux usa comunque automaticamente la page cache RAM per i file KV più
caldi. Spostare tutti i 64 GiB in `/dev/shm` potrebbe ridurre qualche decina di
millisecondi sui cache hit, ma non il decode token/s, e competerebbe con modello,
cache FP16, workspace e KV CUDA nella memoria unificata da 128 GB. Per questo
la configurazione consigliata mantiene gli snapshot su NVMe; un eventuale
hot-cache RAM piccolo va valutato soltanto come ottimizzazione del time-to-first
token.

Con piu' lane, una riga `live kv cache miss ... reason=token-mismatch` descrive
soltanto il rifiuto della frontier attualmente residente in quella lane: non
significa che sia fallita anche la ricerca nei checkpoint su disco. Il server
prosegue cercando il prefisso testuale piu' lungo nella KV persistente. La
sequenza che prova il riuso corretto e':

```text
live kv cache miss ... reason=token-mismatch
kv cache hit text tokens=82605 ...
chat ctx=82605..85184:2579 ... prompt start
```

In questo esempio non viene rifatto il prefill da zero: sono valutati soltanto
i 2.579 token successivi al checkpoint. Un vero prefill completo e'
riconoscibile invece da `chat ctx=0..N:N prompt start` senza un precedente
`kv cache hit`. Il dispatcher load-aware riduce proprio i falsi cambi di
frontier fra client indipendenti; i binding delle tool call continuano ad avere
precedenza assoluta per preservare la KV campionata corretta.

Il canary del coordinatore a due lane ha verificato anche i tre casi distinti:

```text
# richiesta identica sopra la soglia persistente
kv cache hit text tokens=1051 ...
chat ctx=1051..1051:0 prompt start

# vero turno successivo: assistant precedente + nuovo user message
kv cache hit text tokens=1052 ...
chat ctx=1052..1067:15 prompt start

# riscrittura dentro il vecchio user message
live kv cache miss live=1052 prompt=1063 common=1049 reason=token-mismatch
chat ctx=0..1063:1063 prompt start
```

Nel secondo caso l'API ha riportato `cached_tokens=1052` e
`cache_write_tokens=15`: sono stati prefilleati soltanto i 15 token nuovi. Il
terzo caso deve invece essere ricostruito, perche' i token modificati precedono
la frontier salvata. Prompt inferiori a `--kv-cache-min-tokens` non producono
una snapshot persistente del solo confine prompt; ripetere una richiesta corta
dopo una generazione lunga puo' quindi richiedere un nuovo prefill, pur essendo
il percorso KV per le chat lunghe perfettamente operativo.

Comandi utili:

```bash
du -sh /tmp/ds4-gb10-experiment-kv
df -h /tmp
ps -eo pid,rss,vsz,args | grep '[d]s4-server'
nvidia-smi
```

Con contesto `131072` sono stati riportati circa 3.112 MiB di context buffer.
Bisogna quindi lasciare margine oltre agli 80,76 GiB del modello e alle cache
di pesi.

### Persistenza KV `canonical-only` e retry

Il percorso scelto per Athena è `canonical-only`: il prefill viene completato,
poi la frontiera completa del prompt viene congelata una sola volta con la
chiave testuale canonica della richiesta. La cattura della sessione resta
sincrona per garantire che token, KV, compressor, logits e RNG appartengano
alla stessa frontier. Anche il trailer della memoria tool viene serializzato in
questa fase, per evitare che pruning o nuove tool call ne cambino il contenuto
mentre il writer lavora. Il payload immutabile viene poi accodato a un writer
bounded che completa eviction, header e rename mentre il decode puo' proseguire.
La coda contiene al massimo un payload pending oltre a quello in scrittura e
usa file temporanei, non snapshot RAM.

Prima di caricare o eliminare un checkpoint il server drena il writer. Un retry
non puo' quindi osservare un file parziale, perdere un checkpoint gia'
accettato o ripristinare una frontier piu' vecchia. Anche lo shutdown attende
il commit durevole prima di liberare sessioni e tool memory.

Nel test Athena con 15.839 token il checkpoint canonico da 196,32 MiB e' stato
accodato al termine del prefill e consolidato in 92,1 ms mentre iniziava la
generazione; `VmSwap` del server e' rimasto zero e nessun payload temporaneo e'
rimasto dopo il drain. Il vantaggio cresce con i checkpoint lunghi: i log
precedenti mostravano fino a circa 1,5 s per la sola seconda copia da 690 MiB.

Se una scrittura SSE fallisce, i token già prodotti ma non consegnati al client
non vengono salvati come nuova risposta e il server non tenta un rewind CUDA nel
job fallito. Viene armato un guard one-shot; soltanto il retry identico può
caricare il checkpoint canonico completo e ripartire senza ripetere il prefill.
Il guard confronta API, tipo richiesta, lunghezza token e SHA-1 del prompt, e
viene consumato anche da una richiesta non corrispondente.

Log attesi:

```text
kv checkpoint queued tokens=... reason=continued stage=... ms
kv canonical prefill checkpoint tokens=... accepted=1 async=1
kv cache stored tokens=... reason=continued ... save=... ms
stream-failure deferred restore ... abandoned_tail=...
stream retry guard consumed matched=1 ...
stream retry selecting canonical disk restore ...
```

Configurazione predefinita:

```text
DS4_KV_PREFILL_CHECKPOINT_POLICY=canonical-only
DS4_KV_CANONICAL_LONG_PREFILL=1
DS4_KV_CANONICAL_PREFILL_MIN_SEC=30
DS4_KV_KEEP_LONG_TEXT_HITS=1
```

### Nota negativa: snapshot RAM prima del decode

E' stata provata una frontier RAM per canonicalizzare tool call senza passare da
checkpoint NVMe: dopo il prefill, prima del primo token assistant, il server
copiava in RAM host logits, contatori/frontiere compressor e indexer, raw SWA
ring e ring DSpark. La semantica era corretta, ma su GB10 ha ridotto il decode
DSpark di circa 2 token/s.

La causa probabile e' il costo indiretto delle copie sincrone GPU->host
(`cudaMemcpyDeviceToHost`) subito prima della generazione: anche se fuori dal
timer del decode, raffreddano o disturbano il percorso caldo successivo. La
patch e' stata rimossa e il throughput e' tornato al profilo atteso.

Vincolo per patch future: non introdurre snapshot host-side o readback CUDA tra
prefill e decode. Una frontier per tool/canonicalizzazione deve restare
GPU-resident, usare copie device-to-device, oppure essere catturata soltanto in
un punto che non precede direttamente il decode caldo.

### Long context anchor per richieste ripetute

Per prompt molto grandi il retry canonico completo non basta: se la richiesta
successiva cambia solo la domanda finale, il checkpoint completo precedente non
e' piu' un prefisso testuale esatto. Il launcher GB10 quindi alza il limite cold
cache fino al contesto corrente e abilita un singolo checkpoint di prefisso lungo
prima della coda mutevole:

```text
DS4_KV_CACHE_COLD_MAX_TOKENS=$DS4_CTX
DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS=$((DS4_CTX / 2))
DS4_KV_LONG_COLD_ANCHOR_TRIM_TOKENS=$((DS4_CTX / 16))
```

Con `DS4_CTX=131072` questi default restano `65536` e `8192`, ma scalano
automaticamente se il server viene avviato con un contesto diverso. Con un
prompt da circa 125k token il primo giro puo' salvare, per esempio, un anchor
attorno a 114k-116k token. Un turno successivo con lo stesso file e una domanda
diversa puo' quindi partire da un log simile a `ctx=114688..124xxx:~10k` invece
di `ctx=0..124xxx:124xxx`. Questo non aumenta la memoria CUDA: aggiunge solo
uno snapshot su NVMe nella directory `--kv-disk-dir`. Il trim si puo' aumentare
se la parte variabile e' piu' lunga, oppure disabilitare mettendo
`DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS=0`.

### Frontier visibile dopo tool call

Il run del 17 luglio conferma che il nuovo kernel riduce il costo del prefill,
ma non risolve la frontier delle tool call. Dopo 802 token generati il prompt
successivo ha riportato:

```text
live kv cache miss live=86593 prompt=86821 common=28336 reason=token-mismatch
kv cache hit text tokens=24576 ...
chat ctx=24576..86821:62245 TOOLS prompt start
```

La canonicalizzazione ha quindi invalidato la frontier viva e il checkpoint
da 85.790 token e' stato anche espulso per budget disco; il server ha ripreso
dal vecchio anchor 24.576 e rifatto 62.245 token.

L'intervento non copia KV, scratch o buffer CUDA. Dopo una tool call il server
lega il transcript previsto alla frontier esatta gia' in RAM, anche se il
payload conserva reasoning nascosto di turni precedenti. Se il client estende
esattamente quella chiave, possono essere processati soltanto i nuovi tool
result. Il percorso e' riconoscibile da:

```text
tool live checkpoint remembered ... live=... visible=... raw_dsml=1
tool live continuation match=visible-prefix cached=... prompt=...
```

Nel client Athena provato il binding viene registrato, ma il prompt successivo
non coincideva ancora con quella rappresentazione e compariva
`live kv cache miss`. Il percorso Chat Completions ora usa anche il
`tool_call_id` come legame semantico con la frontier campionata. Il fast path
viene ammesso soltanto quando coincidono:

- tutti gli ID della tool call, senza risultati parziali;
- lo stesso contratto/schema degli strumenti;
- la posizione della frontier viva e la lane che la possiede.

In quel caso il server conserva la KV campionata, inclusi reasoning nascosto e
DSML esatto, e tokenizza soltanto `EOS + tool result + nuovo prefisso
assistant`. Il log atteso e':

```text
chat live continuation match=tool-call-ids ids=... cached=... prompt=... replay=...
```

Se uno solo dei vincoli non coincide, il percorso viene ignorato e resta
attiva la seconda rete di sicurezza: il salvataggio
`reason=evict` identifica il piu' lungo checkpoint testuale compatibile con la
richiesta entrante e lo espelle solo dopo tutti i candidati non correlati. Il
log osservato e':

```text
kv cache protecting next-request prefix tokens=...
kv cache hit text tokens=...
```

Il run Athena del 17 luglio ha protetto e ricaricato in successione i checkpoint
25.231, 28.902 e 30.911. Non si e' piu' verificato il ritorno all'anchor 24.576.
I replay sono rimasti limitati alle code effettivamente nuove: 3.671 token in
7,755 s, 2.009 token in 4,657 s e, dopo tre risultati `Read` molto grandi,
13.928 token in 27,784 s. Il decode delle tre tool call successive e' rimasto
tra 21,86 e 23,05 t/s.

La suite server copre sia la costruzione del transcript post-tool sia la
retention del prefisso lungo contro un anchor favorito dall'euristica. Copre
inoltre anchor differente, insieme di tool result incompleto e dispatch verso
la lane proprietaria. Il binding aggiunge soltanto ID e testo di controllo per
lane; non alloca una seconda KV e non aumenta il numero delle snapshot GPU.

Il test end-to-end controllato su Athena usa una Chat Completions tool call,
restituisce il risultato con lo stesso `tool_call_id` e omette il reasoning
nascosto dalla replay client. Prima della correzione il secondo turno riportava
`cached_tokens=0` e `live kv cache miss`. Con il binding semantico:

```text
chat tool binding remembered lane=0 live=398 ids=1 schema=18
chat live continuation match=tool-call-ids ids=1 cached=398 prompt=417 replay=19
```

Il secondo turno ha quindi riutilizzato tutti i 398 token della frontier
campionata e ha valutato soltanto 19 token di coda. La risposta e' rimasta
corretta; il test non modifica sampling, logits, KV accessibile o acceptance
DSpark.

### Nota storica sul deploy

I primi esperimenti usavano `/tmp/ds4-gb10-lab`. La procedura corrente e la
destinazione persistente sono documentate in **Deploy dal Mac** all'inizio del
README; non usare i vecchi comandi `/tmp` per il server DSpark di produzione.

## Esperimento MTP + Tensor Core

Prima di questo intervento è stato creato uno snapshot integrale del checkout:

```text
<local-backup-dir>/ds4-gb10-lab-backup-20260713-pre-mtp-tc
```

Lo snapshot contiene 211 file e ha lo stesso hash del diff Git del laboratorio
al momento della copia.

L'esperimento non converte il modello Q2-imatrix in NVFP4. Una conversione
completa non entra nella memoria della GB10. Usa invece i Tensor Core FP16 dove
MTP crea naturalmente un piccolo batch di due o più token, mantenendo IQ2/Q2
per i routed expert.

### Interventi implementati

1. **Residenza simultanea dei due modelli.** Il backend CUDA conserva la copia
   device degli 80,76 GiB del target quando viene caricato il GGUF MTP da circa
   3,5 GB. Prima il secondo `ds4_gpu_set_model_map_range()` sostituiva la mappa
   globale e poteva far perdere al target il percorso device diretto.
2. **Cache realmente multi-GGUF.** Le chiavi delle cache includono identità
   della mmap e offset. Offset uguali nei due file non collidono più.
3. **Cache FP16 assegnata al punto utile.** Il support model MTP resta
   quasi sempre con `N=1`: viene espanso soltanto `mtp.0.h_proj`, che elabora
   insieme le quattro righe HC. Il resto dei 12 GiB viene riservato alle
   proiezioni target usate dal verifier `N=2..16`, con priorità a output head,
   Q-B, attention output e shared expert.
4. **GEMM tiny-batch Tensor Core.** I matmul Q8-cache/F16 con `N=2..16` usano
   input FP16, accumulo FP32, padding token configurabile e cuBLAS autotune su
   CUDA 13. I risultati delle sole colonne reali vengono mantenuti.
5. **Matmul gemelli accorpati.** Q-A/KV, shared gate/up e compressor KV/gate
   convertono una sola volta la stessa matrice di attivazioni FP32→FP16 e
   lanciano due GEMM Tensor Core dalla stessa tile.
6. **CUDA Graph MTP dedicati.** Drafter e verifier hanno otto varianti ciascuno,
   separate dai tre graph del decode normale. Dopo warm-up indipendenti, il
   layer MTP e il verifier target da 43 layer vengono catturati e lanciati dai
   rispettivi graph. Draft e verifier hanno disabilitazione separata: un errore
   in una famiglia non elimina i graph validi dell'altra.
7. **Workspace e scratch persistenti.** 64 MiB di workspace cuBLAS e uno
   scratch MTP separato evitano allocazioni ripetute e non invalidano i
   puntatori contenuti nei CUDA Graph del decode normale. Se una topologia
   richiede uno scratch più grande durante la capture, l'allocazione viene
   rinviata al replay normale e i graph con puntatori obsoleti sono distrutti.
8. **Una sola barriera nel verifier.** I 43 layer target e l'output head MTP
   restano nello stesso command batch, eliminando una sincronizzazione device.
9. **Build nativa GB10.** `make cuda-spark-mtp-tc` usa `-arch=sm_121a` oltre al
   default stream per thread; non produce più il cubin generico `sm_75`.
10. **Thinking greedy esplicito.** `DS4_MTP_GREEDY_THINK=1` evita che thinking
   mode ripristini una temperatura non-zero, condizione che disabiliterebbe il
   verifier MTP attuale.
11. **Capture MoE e recovery affidabile.** Il clear dei contatori MoE ordinati
    usa `cudaMemsetAsync()` sullo stream catturato. Se capture, update o
    istanziazione falliscono prima del lancio, vengono ripristinati orientamento
    dei tensori, contatori KV/compressor e frontiere prefix-1; il lavoro viene
    eseguito una sola volta normalmente senza restituire `MTP verifier failed`.
12. **Sincronizzazione demand-driven.** Drafter e verifier evitano la barriera
    device ridondante quando il successivo readback sincrono di top-1/logits
    garantisce già il completamento dello stream.
13. **Copy prefix-1 compatibile con CUDA Graph.** Le copie device-to-device
    della frontiera compressor/indexer usano `cudaMemcpyAsync()` sullo stream
    per-thread soltanto durante una capture. Fuori dai graph resta la semantica
    sincrona preesistente.
14. **Modalità benchmark silenziosa.** `DS4_TELEMETRY=0` rimuove davvero timing
    MTP, log speculativi e contatori periodici graph/Tensor Core. In precedenza
    assegnare `0` alle variabili `*_VERBOSE` non funzionava perché il backend
    verificava soltanto che la variabile fosse presente.

I percorsi prestazionali sono opt-in; la residenza multi-GGUF è invece una
correzione di gestione memoria. `--quality` disabilita il percorso Q8→FP16
sperimentale. Il thinking greedy è deterministico e può produrre una risposta
diversa dal sampling standard, anche se non cambia i pesi del modello.

### Primo run MTP e causa del fallback

Il primo test completo su Athena ha verificato la residenza contemporanea di
`80,76 GiB + 3,55 GiB`, 155.000 GEMM Tensor Core e un'accettazione suffix del
`67,96%`. Ha terminato 1.838 token a `14,09 t/s`, contro il riferimento senza
MTP di `14,58 t/s`.

Il risultato non misurava però il verifier graph: sono avvenuti soltanto tre
lanci della famiglia draft, poi la cattura del primo layer target è fallita:

```text
CUDA routed_moe sorted counts clear failed: operation not permitted when stream is capturing
CUDA MTP graph capture aborted
```

Il costo in fallback era `93,902 ms` per ciclo e 1,359 token suffix committati
per ciclo, cioè circa `69,08 ms/token`, leggermente peggiore dei `68,59
ms/token` del riferimento. Questo run resta una diagnosi, non il risultato
prestazionale della build corretta.

Il run successivo, dopo recovery transazionale e separazione delle famiglie,
ha raggiunto `15,13 t/s`, con acceptance suffix `69,04%`. I 1.500 lanci draft
erano attivi, ma `verifier=0`: la capture arrivava al layer 2 e incontrava una
copia D2D sincrona della frontiera prefix-1. La patch al punto 13 rimuove questo
secondo blocco; il risultato va confermato osservando crescere `verifier=`.

### Preparazione su Athena

Build:

```bash
cd /tmp/ds4-gb10-lab && make cuda-spark-mtp-tc
```

Download persistente del support model, se assente:

```bash
cd /tmp/ds4-gb10-lab && DS4_GGUF_DIR=/home/athena/ds4/gguf ./download_model.sh mtp
```

Avvio e salvataggio log:

```bash
cd /tmp/ds4-gb10-lab && ./run-mtp-tc-server.sh 2>&1 | tee /tmp/ds4-mtp-tc.log
```

Analisi dopo il test:

```bash
cd /tmp/ds4-gb10-lab && ./analyze-mtp-log.sh /tmp/ds4-mtp-tc.log
```

Log di startup attesi:

```text
ds4: CUDA copying 80.76 GiB model to device memory
ds4: CUDA copying ... GiB secondary model to device memory
ds4: CUDA MTP Tensor Core cuBLAS workspace 64.00 MiB
ds4: CUDA MTP Tensor Core tiny-batch enabled (fp16 inputs, fp32 accumulate, pad_n=8, autotune=yes)
ds4: CUDA MTP draft/verifier graph enabled (per-thread stream, 16 topology variants)
ds4: CUDA MTP graph variant=... nodes=... instantiated
ds4: CUDA MTP graph launches=... draft=... verifier=... updates=... rebuilds=...
```

### A/B minimo

Il confronto deve usare lo stesso prompt lungo, KV iniziale e almeno 1.000
token generati.

```bash
# MTP normale: niente graph dedicato e niente Tensor Core sperimentale.
DS4_ENABLE_MTP_GRAPH=0 DS4_ENABLE_MTP_TC=0 ./run-mtp-tc-server.sh 2>&1 | tee /tmp/ds4-mtp-base.log

# Isola il valore del graph MTP, lasciando disattivato il nuovo backend GEMM.
DS4_ENABLE_MTP_TC=0 ./run-mtp-tc-server.sh 2>&1 | tee /tmp/ds4-mtp-graph.log

# Graph + Tensor Core senza padding né autotune.
DS4_CUDA_MTP_TC_PAD_N=2 DS4_CUDA_MTP_TC_AUTOTUNE=0 ./run-mtp-tc-server.sh 2>&1 | tee /tmp/ds4-mtp-tc-n2.log

# Configurazione completa GB10.
./run-mtp-tc-server.sh 2>&1 | tee /tmp/ds4-mtp-tc-n8.log

# Stessa configurazione senza telemetria per il benchmark finale.
DS4_TELEMETRY=0 ./run-mtp-tc-server.sh 2>&1 | tee /tmp/ds4-mtp-tc-quiet.log
```

La prima richiesta è riscaldamento e non va inclusa nel risultato. Prima di
valutare i token/s, il contatore `verifier=` deve crescere fino a centinaia di
lanci e nel log non devono comparire `capture aborted` o `MTP verifier failed`.
Il criterio go/no-go finale è velocità netta superiore a `14,58 t/s` con
acceptance sufficiente a compensare il drafter.

## Progetto DeepSeek-V4-Flash-DSpark

DeepSeek ha pubblicato
[`DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark),
lo stesso target V4 Flash con un modulo speculativo addestrato specificamente
per il modello. Non vanno usati come sostituti i checkpoint DSpark Qwen3 o
Gemma della collezione DeepSpec: sono associati a target, tokenizer e hidden
state differenti.

Il config ufficiale del modulo V4 Flash dichiara:

- tre blocchi speculativi `mtp.0`, `mtp.1` e `mtp.2`;
- blocchi di cinque token candidati (`dspark_block_size=5`);
- hidden state target prelevati dai layer 40, 41 e 42;
- Markov head di rango 256;
- confidence head per limitare dinamicamente la parte da verificare.

L'indice ufficiale dei pesi è stato analizzato senza scaricare il checkpoint
completo. Tutti i 4.705 tensor `mtp.*` sono isolati negli ultimi tre shard:

```text
model-00046-of-00048.safetensors  3.610.455.184 byte
model-00047-of-00048.safetensors  3.560.111.960 byte
model-00048-of-00048.safetensors  3.692.775.244 byte
```

Il modulo pesa quindi circa 10,86 GB decimali, equivalenti a 10,12 GiB. Non è
necessario scaricare i precedenti 45 shard né sostituire il GGUF Q2/imatrix da
80,76 GiB già presente su Athena.

Download persistente su Athena, riprendibile e indipendente dal checkout in
`/tmp`:

```bash
mkdir -p /home/athena/ds4/dspark-v4flash-hf; cd /home/athena/ds4/dspark-v4flash-hf; nohup bash -c 'for f in config.json model.safetensors.index.json model-00046-of-00048.safetensors model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do wget -c "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/resolve/main/$f" || exit 1; done' >/tmp/dspark-download.log 2>&1 & echo "Download avviato, PID=$!"
```

Controllo del download:

```bash
tail -f /tmp/dspark-download.log
du -sh /home/athena/ds4/dspark-v4flash-hf
ls -lh /home/athena/ds4/dspark-v4flash-hf
```

### Implementazione nel lab

Il ramo sperimentale ora implementa il primo percorso DSpark completo:

1. `deepseek4-quantize --dspark-sidecar` converte soltanto i tensor `mtp.*`
   degli shard 46–48 in un GGUF autonomo; embedding e output head continuano a
   essere letti dal target Q2/imatrix;
2. `--dspark FILE` carica e valida i tre blocchi, il main projector, Markov head
   e confidence head, in un namespace separato dal vecchio MTP;
3. durante prefill e decode vengono catturate le medie HC dei layer target 40,
   41 e 42; la loro proiezione aggiorna tre ring KV indipendenti da 128 righe;
4. il token corrente resta **pending**, come nell'evaluator DeepSpec ufficiale:
   non viene prima eseguito dal target. Da quel token i cinque draft
   attraversano i blocchi HC/MoE già ottimizzati di ds4 e una nuova attention
   CUDA non causale sulla finestra main più le cinque righe draft. La prima
   riga usa la posizione RoPE del current (non `current+1`), in accordo con
   `anchor_position + [0..block_size)` usato in training;
5. l'head DSpark applica in GPU i cinque bias Markov sequenziali e restituisce
   gli ID senza readback dei cinque vettori vocab;
6. il target verifica in **un solo microbatch** `[current + K draft]`, quindi
   `K+1` righe. Il current viene sempre committato; un full accept conserva
   direttamente il frontier, mentre un partial accept committa uno dei frontier
   1..5 catturati nello stesso pass. Non esiste più un decode anchor separato e
   non vengono rieseguiti i token già verificati;
7. le tre ring DSpark salvano fino a sei righe prima del verifier e ripristinano
   soltanto la coda rifiutata. Questo è necessario quando la ring da 128 righe
   è piena, perché le scritture future sovrascrivono storia ancora visibile;
8. la confidence head ufficiale viene eseguita sulla GPU usando l'hidden
   DSpark dopo la RMSNorm finale, lo stesso tensore sul quale è stata addestrata.
   Uno scheduler
   calibrato sul target Q2 sceglie dinamicamente `K=0..5` confrontando il rate
   del decode normale con `E[1 + draft accettati] / (drafter + verifier K+1)`.
   Dopo la calibrazione riduce anche il batch del drafter al K
   scelto, con una nuova esplorazione completa ogni 64 cicli. Un circuit
   breaker sospende temporaneamente DSpark quando il rendimento osservato è
   inferiore al decode normale;
9. sampling con temperatura, top-k, top-p e min-p è lossless: ogni proposta
   greedy viene confrontata con un campione reale dei logits target. In caso di
   mismatch il campione target già estratto resta pending per il ciclo seguente,
   senza consumare due volte l'RNG. DSpark non forza thinking greedy e non
   modifica la distribuzione delle risposte;
10. il payload KV versione 3 persiste anche le tre ring main-KV (circa 768 KiB
    aggiuntivi). Store e load su disco restano quindi attivi con `--dspark`.

Il primo prototipo ha misurato un drafter stabile di circa `27 ms`, verifier
`K=5` di circa `200 ms` e `9,322 t/s` sulla prima risposta. L'analisi del codice
ufficiale DeepSpec ha poi individuato due costi artificiali: replay dei partial
accept e, soprattutto, un target decode del current da circa 67 ms pagato
**prima** del verifier. Il commit diretto elimina il replay; la macchina a stati
fusa elimina completamente il decode anchor e porta il target al contratto
ufficiale `K+1`. Lo scheduler evita inoltre di pagare sempre K=5 quando K più
piccoli hanno rendimento migliore.

Build del sidecar su Athena:

```bash
cd /tmp/ds4-gb10-lab
./build-dspark-sidecar.sh 2>&1 | tee /tmp/ds4-dspark-convert.log
```

Build CUDA e avvio sulla porta abituale:

```bash
cd /tmp/ds4-gb10-lab
make cuda-spark-graph-sm121
DS4_TELEMETRY=1 ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-dspark.log
```

Il launcher è silenzioso per default (`DS4_TELEMETRY=0`), perché timing e log
per ciclo non devono contaminare il benchmark. Usare telemetria 1 soltanto per
calibrare e diagnosticare lo scheduler.

Il launcher usa ora una cache Q8→F16 da 12 GiB: il run da 6 GiB arrivava al
limite durante la preparazione del target. Con circa 80,8 GiB di target, 10,7
GiB di sidecar, 12 GiB di cache e 3,1 GiB di contesto rimane margine sui 128 GB
della GB10. `DS4_CUDA_DSPARK_CACHE_PRIORITY=1` prepara prima output, attention
e shared-expert Q8 usati dal verifier, invece di consumare il budget in ordine
di file.

Il launcher DSpark imposta `--prefill-chunk` dal profilo memoria
(`DS4_PREFILL_CHUNK` per A/B) e, con policy `canonical-only`, abilita
`DS4_PREFILL_FINAL_LOGITS_ONLY=1`: durante un cold prefill lungo vengono evitati
output-head e readback dei logits sui chunk intermedi, calcolandoli soltanto sul
chunk finale usato dal decode.

`DS4_MEMORY_PROFILE=prefill-fast` mantiene 12 GiB di cache Q8→F16 per non
sottrarre le proiezioni calde al prefill target, usa chunk 4096 per contenere lo
scratch e imposta `DS4_CUDA_COPY_SECONDARY_MODEL=0`, quindi il sidecar DSpark
viene mappato invece di essere copiato in device memory. Questo profilo è solo
per A/B del prefill: può liberare molta memoria, ma va confrontato con il
throughput DSpark di generazione.

Il default torna a `DS4_MEMORY_PROFILE=balanced`: 12 GiB di cache Q8→F16, chunk
8192 e copia device del sidecar DSpark. I risparmi sugli scratch sotto servono a
pagare questo profilo senza togliere memoria al prefill. Se la macchina torna
troppo vicina al limite, `DS4_MEMORY_PROFILE=lean` conserva DSpark copiato ma usa
11 GiB di cache Q8→F16 e chunk 4096.

Il profilo DSpark rilascia inoltre le pagine sorgente dei GGUF dopo una copia
CUDA riuscita del target e del sidecar. Il launcher imposta
`DS4_CUDA_DROP_COPIED_MODEL_PAGES=1`, quindi `posix_fadvise(DONTNEED)` e
`posix_madvise(DONTNEED)` lasciano residenti i pesi device senza tenere calde
anche le stesse pagine file/mmap lato host. Nel test del 16 luglio 2026 questo
ha liberato circa 2 GiB di RAM senza ridurre prefill o throughput DSpark. Per
diagnostica si può disattivare dal launcher con
`DS4_CUDA_DROP_COPIED_MODEL_PAGES=0`; usando il binario direttamente resta
valido il rollback `DS4_CUDA_KEEP_MODEL_PAGES=1`.

Il prefill CUDA usa inoltre una maschera compressa densa a una sola riga: il
percorso batched ratio-4 passa gli indici top-k a `comp_selected` e non richiede
più una slab `comp_cap * prefill_cap`. Questo libera circa 512 MiB con chunk
4096, circa 1 GiB con chunk 8192 e ctx 131k, senza cambiare i 512 top-k. Per
diagnostica si può ripristinare il vecchio buffer con
`DS4_PREFILL_DENSE_COMP_MASK=1`.

I buffer batch con lifetime non sovrapposti condividono inoltre la stessa
allocazione device: `batch_flat_hc`, `batch_attn_low`, `batch_group_tmp` e
`batch_low_tmp` riusano regioni di `batch_q`; `batch_q_half` riusa
`batch_routed_down`; `batch_ffn_cur` e `batch_ffn_norm` riusano i corrispondenti
buffer attention. Con Flash, chunk 4096, questo vale circa altri 850 MiB. Il
risparmio scala a circa 1.7 GiB con chunk 8192. Il rollback diagnostico è
`DS4_PREFILL_NO_SCRATCH_ALIAS=1`.

### Routed MoE prefill MMQ Entrpi sui GGUF originali

Il collo di bottiglia MoE del target usa ora il backend CUDA MMQ già validato
nel fork [Entrpi/ds4](https://github.com/Entrpi/ds4). I sorgenti importati sono
descritti in `cuda/mmq/VENDOR.md`: i kernel MMQ derivano dallo snapshot
llama.cpp `5c0e9468`, mentre `ds4_mmq.*` e gli adattatori sono il raccordo
Entrpi per DS4.

Il percorso automatico esegue, nell'ordine:

1. costruzione della mappa degli assignment top-6 in ordine expert-major;
2. quantizzazione Q8_1 unica dell'attivazione comune e MMQ batched IQ2_XXS
   accoppiato per gate e up;
3. clamp, SwiGLU e routing weight calcolati in una sola passata nel `mid`;
4. gather e quantizzazione Q8_1 D2S6 tramite il quantizzatore MMQ validato;
5. MMQ batched Q2_K down con la stessa mappa e gli stessi `expert_bounds`,
   senza un secondo `mm_ids_helper`;
6. somma delle sei uscite direttamente nel risultato MoE del token.

Per chunk 8192 il down presenta 49152 assignment. La prima versione ricostruiva
la mappa interpretando gli assignment come 49152 token da un esperto e finiva
nella variante globale dell'`mm_ids_helper`. La pipeline corrente conserva
invece la mappa top-6 costruita per gate/up: `ids_dst` è già la permutazione
esatta tra ordine expert-major e riga `[token, slot]` richiesta dal down.

Il primo tentativo di scrivere direttamente la Q8_1 D2S6 dall'epilogo ha
superato la parita' gate/up ma ha prodotto `final=7.19737 rel-rmse` sul test
Q2_K; e' stato quindi rimosso prima del deploy. La pipeline usa il buffer `mid`
FP32 gia' allocato e il quantizzatore D2S6 collaudato, mantenendo il vantaggio
della mappa unica senza introdurre un layout quantizzato non validato. La Q8_1
di gate/up viene liberata sullo stream prima di allocare quella del down,
quindi il riuso della mappa non somma i due grandi scratch al picco del pool.

Questa integrazione include intenzionalmente soltanto il percorso sui pesi
GGUF canonici già copiati dal loader. Non chiama `ds4_repack`, non costruisce
artifact SoA/aligned, non sostituisce le range del modello e non cambia la
copia CUDA o la policy già esistente di rilascio delle pagine sorgente. In
particolare, non torna il repack da circa 76.5 GiB che aveva portato Athena al
99.6% di memoria e azzerato il decode.

Il confine con il decode è strutturale, senza flag:

- sono accettati soltanto `IQ2_XXS gate/up`, `Q2_K down`, top-6 e
  `n_tokens >= 1024`;
- decode target a una riga e verifier DSpark da 2..6 righe restano sui kernel
  precedenti;
- le code sotto 1024 token tornano automaticamente al routed-MoE batch
  precedente, evitando MMQ nei batch con poca occupazione;
- il sidecar DSpark Q4, CUDA Graph e SSD expert streaming restano invariati;
- gate, up, mid FP32 e down riusano i quattro buffer batch già allocati da DS4;
  soltanto mappe e attivazioni quantizzate temporanee passano dal pool CUDA
  asincrono e vengono liberate in ordine di stream.

Gli epiloghi SwiGLU e sum azzerano i non-finiti mentre leggono gli intermedi.
Le entry `consumer_sanitizes` possono quindi saltare tre passate standalone su
gate, up e down mantenendo la stessa semantica dei kernel precedenti.

Il primo chunk compatibile deve stampare:

```text
ds4: CUDA Entrpi batched MMQ MoE prefill enabled (single-map IQ2 gate/up + Q2 down, token-bound stream-K; decode excluded)
```

Prima del deploy è obbligatorio eseguire:

```bash
make -B cuda-regression CUDA_ARCH=sm_121a
```

La regressione genera pesi IQ2_XXS/Q2_K e attivazioni deterministici non
correlati ed esegue la stessa pipeline di produzione con il clamp reale
`10.0`. Il log `raw-GGUF MMQ MoE parity ... rel-rmse/bad` deve riportare zero
elementi gate/up post-clamp fuori tolleranza, `mid` e `final` sotto la soglia,
e concludersi con
`cuda long-context regression: OK`. MMQ usa attivazioni Q8_1, mentre il
percorso DS4 precedente usa Q8_K: pesi, top-6, clamp, routing weight e formula
MoE non cambiano, ma i logits non sono attesi bit-identici.

Il commit locale `4eb7441` congela la prima integrazione raw-GGUF validata su
Athena: circa 448 t/s sul chunk 8192, 413-418 t/s sull'intero prompt da
10-13K e decode DSpark 23-26 t/s nei run osservati. Il commit locale `2cc0fcd`
aggiunge la mappa unica e il crossover a 1024 descritti sopra. La regression
CUDA e il run Athena da 63K hanno confermato correttezza e decode: 450.51 t/s
sul primo chunk, 376.02 t/s medi sull'intero prefill e 20.81 t/s dopo i primi
100 token di decode lungo. Il guadagno sul primo chunk rispetto a `4eb7441` e'
pero' soltanto circa 0.6%, quindi `2cc0fcd` e' una base stabile ma non soddisfa
da solo il successivo obiettivo prestazionale di almeno +10%. La fusione
diretta SwiGLU-Q8 resta esclusa finche' un test isolato del layout D2S6 non
dimostra parita'.

#### Bound stream-K confinato al target prefill

La pipeline fused usa top-6 esatto: per costruzione ciascun token seleziona un
esperto al massimo una volta. Il numero di righe di qualsiasi bucket esperto e'
quindi limitato da `n_tokens`, non dalle `n_tokens * 6` righe raccolte. Sul
chunk 8192 il dominio X dello stream-K passa da 49152 a 8192 colonne logiche.

Il cambio e' intenzionalmente confinato a
`ds4_mmq_iq2_xxs_q2_K_moe_fused`. I wrapper MMQ generici, DSpark/MTP e decode
mantengono il bound conservativo `ne_get_rows`. Questa separazione e'
importante perche' il commit Entrpi `82b2622` aveva associato un precedente
cambio globale a `n_tokens` a output incompleti e BOS, prima della correzione
del write-back e dell'azzeramento del fixup stream-K ora presenti nel codice.

La regression usa un routing top-6 valido ma massimamente sbilanciato: esperto
zero compare una volta in tutti i token e raggiunge esattamente il nuovo bound;
gli altri cinque slot restano distinti. Il log runtime che conferma il deploy
deve contenere `token-bound stream-K`. Su Athena la regression skew ha chiuso
con `final=0.01851 rel-rmse` e `cuda long-context regression: OK`. Nel tratto
sovrapposto 24.5K-57.3K il throughput aggregato e' salito da circa 364.7 a
425.9 t/s, pari a +16.8%; la richiesta calda successiva e' partita a 488.19
t/s a 24.5K. Il decode a 83K ha prodotto 401 token a 23.00 t/s, senza BOS,
non-finite o regressioni DSpark.

#### Token-tile HMMA per attenzione prefill

Il percorso validato sul throughput porta nel fork single-session il token-tile
HMMA dei commit Entrpi `47438d7` e `9de3044`, senza importare serving multi-sequence,
KV FP8 o artifact SoA. Sedici token e due head condividono una CTA; query e KV
vengono convertite tile per tile in F16 e score/PV usano
`mma.sync.m16n8k16` con accumulo FP32.

Nel percorso ratio-4 indexed, ogni tile costruisce l'unione delle 512 righe
Top-K esatte dei suoi token e conserva una mask a 16 bit per riga compressa.
Non cambia K, non approssima la selezione e non materializza un nuovo score
matrix. La variante raw/mixed genera invece l'intervallo compresso causale con
lo stesso formato di record e riusa il medesimo kernel HMMA.

Le KV del modello restano nel formato e nei buffer esistenti. Due mirror F16 e
i record per tile vivono in uno scratch CUDA dedicato, separato dall'arena
referenziata dai graph del decode. Con chunk 8192 e `n_comp <= 32768` il suo
high-water e' circa 168 MiB; viene allocato una volta e non cresce a ogni
chunk.

Il confine prestazionale e' strutturale, senza flag runtime: servono almeno 128
token, 64 head, dimensione 512 e raw window 128. L'indexed richiede inoltre
Top-K 512 e ratio non nullo. Decode monoriga, verifier DSpark 2–6 e forme non
supportate continuano sui kernel precedenti.

La regressione confronta indexed e raw/mixed con i rispettivi kernel online
precedenti su input deterministici, controlla finitezza, relative RMSE e
massimo errore assoluto. La conversione F16 cambia l'ordine numerico e non e'
bit-identica; prima dell'accettazione restano obbligatori controllo qualitativo
reale, prefill almeno +10%, memoria entro il budget e decode DSpark invariato.

Log runtime attesi quando entrambi i percorsi diventano eleggibili:

```text
ds4: CUDA token-tile HMMA raw/mixed prefill enabled (tile=16, heads=2)
ds4: CUDA token-tile HMMA indexed prefill enabled (tile=16, heads=2, exact-topk=512)
```

Risultato Athena del 17 luglio, confronto alla stessa posizione assoluta:

```text
Intervallo ctx          Prima      Token-tile   Delta
24576..32768          412.63 t/s   508.57 t/s  +23.25%
32768..40960          451.08 t/s   569.65 t/s  +26.29%
40960..49152          432.05 t/s   546.17 t/s  +26.41%
49152..57344          410.39 t/s   517.30 t/s  +26.05%
57344..65536          395.45 t/s   500.24 t/s  +26.50%
65536..73728          378.16 t/s   478.44 t/s  +26.52%
73728..81920          364.65 t/s   460.02 t/s  +26.15%
Pesato 57.344 token   404.46 t/s   509.14 t/s  +25.88%
```

La richiesta completa, leggermente piu' lunga del riferimento precedente, ha
chiuso 61.214 token in 123,273 secondi, pari a 496,57 t/s. Il decode a circa
86K e' rimasto integro e ha chiuso 802 token a 23,58 t/s; il costo iniziale di
warm-up dei graph si riassorbe entro i primi blocchi da 50 token.

#### Epilogue prefill fuse: HC, RMS, RoPE e MoE sum

Il profilo Nsight del chunk indexed a 98K attribuiva circa 0,813 s alla
conversione F32->F16, 0,644 s alla RMS plain, 0,547 s a HC expand, 0,448 s
alla head RMS, 0,342 s al packing e 0,168 s alla sola somma dei sei esperti.
Sono passate di memoria dipendenti che preparano GEMM o lo stato HC, non
calcolo attention/MoE utile.

Il nuovo percorso CUDA largo, eleggibile da 128 token, riduce queste
materializzazioni senza aggiungere buffer permanenti:

- RMS HC scrive direttamente l'attivazione F16 gia' consumata dal GEMM
  successivo; la stessa riga F16 viene portata da attention a FFN e fra layer;
- HC expand scrive ancora lo stato canonico FP32, ma produce nello stesso CTA
  anche la RMS F16 per il blocco successivo;
- la proiezione attention applica inverse RoPE durante il packing group-major
  F16, evitando la passata in-place separata sui 64 head;
- il routed MoE largo puo' lasciare i sei down output nel layout per-esperto;
  un'unica epilogue esegue somma finita in ordine di slot, add dello shared
  expert, HC expand e RMS F16.

Decode monoriga, verifier target/DSpark 2..6, debug dump, steering e percorsi
shared-down F16 conservano la pipeline precedente. La patch riusa scratch gia'
allocato (`batch_flat_hc` e routed-down), quindi non aumenta il picco RAM del
server. Non modifica pesi, Top-K, KV o formule del modello; rimuove soltanto
round-trip intermedi. `cuda-regression` confronta la pipeline materializzata e
quella fusa su 128 token, compresi NaN/Inf nel down MoE, e richiede parita'
FP32 entro `2e-6` e scarto massimo di un ULP F16. Lo stesso limite di un ULP
vale per inverse-RoPE + packing; il test registra anche indice e bit della
prima differenza e rifiuta qualunque scarto superiore.

L'integrazione ha superato il gate Athena del 20 luglio 2026. La regression
`sm_121a` ha chiuso con `cuda long-context regression: OK`; il confronto fra
pipeline materializzata e fusa ha rilevato al massimo un ULP F16 e nessun
errore FP32 oltre `2e-6`.

Risultati end-to-end osservati con profilo `balanced`, chunk 8192, contesto
fisico 131072 e sidecar DSpark copiato:

```text
Richiesta / intervallo       Prefill medio   Chunk significativi
cold 0..25352                902.67 t/s      835.81, 991.09, 977.94 t/s
cold 0..13376                952.97 t/s      1009.78, 952.66 t/s
append 57846..78243          730.56 t/s      653.93, 898.45 t/s
append 77157..90505          760.77 t/s      642.33, 891.13 t/s
append 90505..93565          628.27 t/s      coda singola da 3060 token
```

Il decode DSpark e' rimasto nel profilo atteso: 21.85 t/s dopo il cold prefill
da 25K, 24.00 t/s a circa 90K e 23.46 t/s a circa 93.5K. Tool call, checkpoint
canonici e ripresa append hanno continuato a funzionare.

La memoria di sistema era 118 GiB usati su 121 GiB, con circa 2.9 GiB
disponibili. Dei circa 1.6 GiB in swap, soltanto 141 MiB appartenevano al
processo `ds4-server`; dieci campioni `vmstat` non hanno mostrato paging
sostenuto. Il margine GB10 resta stretto, ma non e' emersa crescita persistente
attribuibile alle fusioni.

#### D2R MoE tail-aware

La pipeline fused D2R non assegna piu' sempre una CTA larga al residuo di ogni
bucket esperto. Gate/up usa specializzazioni da 8, 16 e 32 righe; il down Q2_K
usa 8, 16, 32 e 64 righe. Una singola costruzione expert-major separa in modo
deterministico i tile pieni dalle code e conserva ordine degli assignment,
routing weight, formule SwiGLU e accumulo esistenti. Le nuove liste richiedono
soltanto pochi KiB nello scratch preallocato e non aggiungono copie permanenti
dei pesi o delle attivazioni.

Il self-test MMQ costruisce bucket con tutte le classi di coda usate in
produzione e confronta la pipeline D2R con il riferimento. La regression CUDA
`sm_121a` ha chiuso con `cuda long-context regression: OK`. Il log runtime che
conferma il percorso e':

```text
ds4: CUDA complete fused MoE D2R prefill enabled (tail-aware 8/16/32 gate-up, 8/16/32/64 down, preallocated workspace)
```

Risultati Athena del 20 luglio 2026, con profilo `balanced`, chunk 8192 e lo
stesso carico applicativo usato per il riferimento precedente:

```text
Richiesta / intervallo       Prefill medio   Chunk significativi
cold 0..25785                940.93 t/s      994.65, 974.03, 958.98 t/s
append 15877..57754          909.96 t/s      905.84, 940.13, 941.65, 938.11, 927.28 t/s
append 28050..69918          844.86 t/s      942.80, 937.55, 930.84, 885.79 t/s
append 69918..83345          751.28 t/s      625.11, 879.78, 577.96 t/s
append 83345..98266          636.72 t/s      637.87, 635.77 t/s
```

Il cold prefill da circa 25K migliora da 902.67 a 940.93 t/s (`+4.2%`); sui
tre chunk pieni il guadagno e' circa `+5%`. Il beneficio piu' importante e' la
tenuta dei chunk pieni mentre cresce il contesto. Oltre 83K il costo della
sparse attention torna dominante e limita il prefill a circa 637 t/s: questo
tratto non e' un limite del residuo MoE.

Il decode DSpark e' rimasto integro: 27.44 t/s a circa 70K, 26.00 t/s a 83K e
24.66 t/s a 98K. Checkpoint canonici e append hanno continuato a funzionare e
la RAM e' rimasta costante rispetto al riferimento. La patch viene quindi
accettata come ottimizzazione MoE senza aumento del budget di memoria; non va
interpretata come soluzione al decadimento attention oltre 80K.

Un esperimento successivo ha provato un tile gate/up N64 con 16 warp, divisi
in due gruppi N32 che condividevano il caricamento IQ2 senza raddoppiare gli
accumulatori per thread. Insieme e' stato provato il bilanciamento dell'ultima
coppia di chunk, trasformando un `8192 + coda breve` in due chunk di dimensione
simile. La prova del 20 luglio 2026 e' stata respinta.

Sul cold, i primi 24.576 token sono scesi da circa 976 a 908 t/s (`-6,9%`).
L'append lungo confrontabile e' sceso da circa 845 a 789 t/s (`-6,6%`), mentre
il decode e' rimasto sano a 27,85 t/s intorno a 68K. Il tile N64 riduceva la
flessibilita' di scheduling delle CTA senza recuperarla con il riuso IQ2. Il
bilanciamento trasformava invece un lancio pieno efficiente e una coda breve
in due lanci da circa 5.230 token, entrambi osservati intorno a 630 t/s.
Inoltre il cold anchor a 24.576 spezzava il range prima dello scheduler, quindi
la coda cold da 776 token non poteva essere bilanciata. Il codice N64 e il
bilanciamento sono stati rimossi; resta il percorso N32 tail-aware del commit
`453eec4`.

#### Scorer MXFP4 N128 ping-pong

Lo scorer indexer prefill SM121a elabora ora due gruppi N64 indipendenti per
CTA, formando un tile N128 che riusa la stessa query MXFP4. Due buffer shared
ping-pong precaricano la testa successiva e riducono da due a una le barriere
per testa, conservando l'ordine crescente dell'accumulo e il Top-K esatto. Il
percorso verifier da 1-6 righe non cambia e non vengono aggiunte allocazioni
persistenti. Il kernel compilato usa 64 registri, 3328 byte di shared statica e
non genera spill locali.

La regressione CUDA esercita il percorso prefill con 17 token e 641 righe
compresse, incluse entrambe le code N128. Il confronto fra forme ha misurato al
massimo 2 ULP su 1457 valori di 98073, con tutti gli 8704 indici Top-K identici:

```text
cuda-regression: ... shape-consistency-max=0.00012207031 ulp=2 changed=1457 topk=8704/8704 causal-topk=8704/8704
cuda long-context regression: OK
```

Un run applicativo ha mantenuto 964,90 t/s medi sui sette chunk completi fra
16K e 73K e 923,09 t/s sui due chunk completi fra 82K e 98K. Il benchmark
riproducibile append, eseguito subito dopo con prompt fisso, ha confermato:

```text
Frontiera   Token prefill   Prefill
65536       65536           913,61 t/s
81920       16384           937,77 t/s
98304       16384           913,54 t/s
```

Il percorso lungo resta quindi sopra 900 t/s fino a 98K senza aumentare la
memoria persistente. Il `gen_tps` greedy breve di questo benchmark non e'
confrontabile direttamente con il server: 128 token includono il warm-up dei
CUDA Graph e il testo fisso ha prodotto soltanto 1,68-1,88 token accettati per
ciclo. Nel run HTTP, non modificato da questa patch, il decode ha mantenuto
23,92 t/s a 78K e 22,19 t/s a 100K.

#### Copia iniziale CUDA pipelined

Il caricamento del target da 80.76 GiB non usa piu' un unico `cudaMemcpy`
sincrono dalla mmap. Il percorso di produzione legge il GGUF in chunk da 64
MiB con `pread` e, quando disponibile, `O_DIRECT`; quattro buffer host pinned
permettono di sovrapporre la lettura del chunk successivo al
`cudaMemcpyAsync` corrente. Le pagine sorgente vengono scartate dopo che il
contenuto e' entrato nello staging pinned, quindi resta valido il risparmio RAM
del commit `bbaa42f`.

I buffer, gli eventi e lo stream di upload sono temporanei e vengono liberati
prima del caricamento del sidecar DSpark. Un errore di allocazione, lettura o
copia ripristina automaticamente il precedente `cudaMemcpy` monolitico. Il log
che dimostra l'attivazione e':

```text
ds4: CUDA pipelined model copy 80.76 GiB (chunk=64 MiB, stages=4, direct-io=1)
```

La metrica e' ora wall-clock e riporta anche i GiB/s. Il riferimento precedente
su Athena era 419.9-478.5 secondi per la sola copia primaria; l'obiettivo di
accettazione e' non oltre 210 secondi, senza variazioni nella memoria residente
a server pronto. Dopo il test di startup restano obbligatori la regression CUDA
e un controllo prefill/decode DSpark, perche' il layout dei pesi non cambia ma
la disponibilita' dei byte sul device deve restare completa.

Nel run del 17 luglio la copia primaria da 80,76 GiB ha impiegato 19,076
secondi, pari a 4,23 GiB/s. La copia separata del sidecar da 10,70 GiB ha
impiegato 69,999 secondi ed era diventata la parte dominante del caricamento.
Il sidecar usa ora lo stesso upload pipelined del target, mantenendo invariati
puntatore device, ownership e residenza finale. In caso di errore torna
automaticamente alla copia monolitica; `DS4_CUDA_SECONDARY_COPY_PIPELINED=0`
forza esplicitamente il percorso precedente per un confronto A/B o un rollback.
Il log che prova l'attivazione e':

```text
ds4: CUDA pipelined secondary model copy 10.70 GiB (chunk=64 MiB, stages=4, direct-io=1)
ds4: CUDA pipelined secondary model copy complete in ...s (10.70 GiB, ... GiB/s)
```

Questa patch modifica soltanto il trasferimento durante lo startup: formato
GGUF, pesi, cache DSpark e kernel di decode non cambiano. Il tempo effettivo e
la neutralita' del decode devono comunque essere confermati su Athena.

I numeri Entrpi su PRO 6000 non vanno trasferiti direttamente alla GB10. Per
accettare la patch servono sullo stesso prompt Athena: prefill superiore,
nessun aumento materiale della memoria residente dopo il chunk, decode DSpark
invariato rispetto al riferimento 18-20 t/s e nessuna anomalia qualitativa.

`DS4_CUDA_DSPARK_GRAPH=1` usa graph dedicati sia per il drafter sia per il
verifier: dieci famiglie K-aware (drafter K=1..5 e verifier con 2..6 righe),
ciascuna con quattro varianti di posizione e quattro per il passaggio del
boundary ratio-128. Gli 80 slot DSpark sono
separati dai graph del decode normale e dai 16 slot MTP; cambiare K o passare
dal draft alla verifica non ricostruisce quindi un eseguibile con una topologia
differente. Il primo uso di ogni famiglia resta intenzionalmente un warm-up
normale, così allocazioni lazy e preparazione cuBLAS avvengono fuori capture.

La KV disk cache è nuovamente attiva. L'ABI esterna e il payload sessione sono
stati incrementati: i vecchi checkpoint privi delle ring DSpark vengono
rifiutati automaticamente, mentre i nuovi ripristinano le tre ring in ordine
logico e validano blocchi, finestra, dimensione head e numero di righe.

In greedy e sampling probabilistico DSpark non decide il testo finale: propone
token e il target Q2 li accetta o sostituisce. Il verifier batch usa sempre i
pesi target, ma l'ordine delle riduzioni CUDA può non essere bit-identico al
decode monoriga quando due logits sono quasi pari. Per misurare questa eventuale
divergenza esiste `DS4_DSPARK_EXACT_VERIFY=1`, che usa la stessa macchina a
stati pending ma verifica il target in sequenza (corretto e volutamente lento).
Il criterio prestazionale resta un throughput pesato superiore al baseline
`14,689 t/s`. Il dato decisivo è ora il rapporto tra `emitted`, `target_rows` e
tempo verifier: il circuit breaker garantisce che una sequenza poco prevedibile
torni automaticamente al decode target.

Analisi del log:

```bash
./analyze-dspark-log.sh /tmp/ds4-dspark.log 1
```

Oltre a acceptance e throughput, il report mostra selezioni `K=0..5` e bypass
del circuit breaker. I token calcolati prima di una scelta `K=0` non vengono
più contati come draft verificati: il report separa fallback e verifier e
mostra costo/rate per ogni K. Per un benchmark finale usare `DS4_TELEMETRY=0`; i log
per ciclo sono diagnostici e non devono essere inclusi nella misura definitiva.

Nel primo test del verifier fuso, rianalizzato separando correttamente i
fallback, DSpark ha accettato il 76,32% dei 38 token realmente verificati. I
pochi campioni disponibili hanno misurato `16,96 t/s` per K=2 e `17,04 t/s`
per K=4, contro `14,47 t/s` pesati sull'intera richiesta. Il totale era peggiore
perché il vecchio scheduler ha eseguito appena 15 cicli verifier su 3004 e ha
attivato 2656 bypass. Questi numeri motivano la calibrazione obbligatoria e la
selezione basata sul throughput reale per K; il dato K=5 da `25,40 t/s` aveva
un solo campione e non va ancora considerato rappresentativo.

Controlli A/B dello scheduler:

```text
DS4_DSPARK_FIXED_VERIFY=1       forza sempre --dspark-draft
DS4_DSPARK_NO_AUTOTUNE=1       salta l'esplorazione iniziale K=1..N
DS4_DSPARK_NO_CIRCUIT_BREAKER=1 non sospende DSpark nei tratti sfavorevoli
DS4_DSPARK_AUTOTUNE_SAMPLES=... campioni iniziali per K (default 8)
DS4_DSPARK_PROBE_INTERVAL=...   cicli target tra due probe DSpark (default 64)
DS4_DSPARK_EXACT_VERIFY=1       oracle sequenziale target, solo A/B qualità
DS4_DSPARK_CONF_TEMPERATURE=... calibrazione logit confidence
DS4_DSPARK_STS_PROFILE=...      profilo STS offline versionato e immutabile
DS4_DSPARK_STS_CAPTURE=...      CSV held-out; forza K=5 nel solo run di raccolta
DS4_DSPARK_STS_TEMPERATURES=... fallback online: cinque temperature iniziali
DS4_DSPARK_STS_DISABLE=1        disabilita il vecchio adattamento online
DS4_DSPARK_SPS_PROFILE=...      curva SPS(B) offline completa e fingerprintata
DS4_DSPARK_CONF_BIAS=...        bias della sigmoid confidence
DS4_DSPARK_CONF_THRESHOLD=...   interrompe K quando confidence scende sotto soglia
DS4_DSPARK_CAUSAL_MIN_K=...     minimo K dello scheduler stabile precedente (default 2)
DS4_DSPARK_SCHEDULER_SHADOW=1   osserva Algorithm 1 + capacity production
DS4_DSPARK_SCHEDULER_DETERMINISTIC=1 applica il canary hardware-aware
DS4_DSPARK_K0_PATIENCE=...      K0 consecutivi prima del cooldown (default 4)
DS4_DSPARK_K0_COOLDOWN=...      decode ordinari dopo la patience (default 8)
DS4_DSPARK_CHAMPION_DISABLE=1   disabilita la scelta del miglior K misurato
DS4_DSPARK_CHAMPION_MARGIN=...  margine sul target ordinario (default 1.01)
DS4_DSPARK_CHAMPION_EXIT_MARGIN margine di uscita isteretico (default 0.98)
DS4_CUDA_MOE_TINY_DIRECT=1      percorso MoE diretto per verifier da 2..6 righe
DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY=1 limita il direct-MoE al sidecar Q4
DS4_CUDA_Q8_BATCH_REUSE=1       riusa i pesi Q8 tra 2..8 righe verifier
DS4_CUDA_NO_BATCHED_ARGMAX=1    rollback argmax parallelo multi-riga
```

I default non richiedono tuning manuale: ogni K raccoglie otto campioni, così
il primo warm-up CUDA/Graph non può disabilitare DSpark. Per ogni K viene poi
misurato anche il throughput end-to-end del ciclo (`emitted / (draft+verify)`).
Dopo la calibrazione questa misura hardware pesa per l'85% nella selezione,
mentre confidence e accettazioni osservate sul target Q2/imatrix adattano il
restante 15% al prefisso corrente. La telemetria espone i cinque EWMA in
`rate_tps=[K1 K2 K3 K4 K5]`.

Lo scheduler è diviso in due decisioni. Prima di lanciare DSpark confronta il
miglior K storico con il decode target e, se non è redditizio, esegue il Large
direttamente senza pagare il drafter; un probe ogni 64 cicli permette di
recuperare quando cambia il workload. Dopo che DSpark ha prodotto le proposte,
K=0 viene invece confrontato con il costo reale `draft + target`, perché il
tempo del drafter è ormai stato sostenuto. Questo evita il precedente ciclo
patologico di draft scartati seguito da cooldown.

Il sidecar genera sempre tutte le cinque posizioni previste dal checkpoint,
anche quando lo scheduler decide di verificare soltanto un prefisso K più
corto. L'attenzione DSpark tra gli slot non è causale: ridurre fisicamente il
blocco può cambiare anche le prime proposte e ridurne l'accettazione. Il limite
`--dspark-draft` controlla quindi il K massimo del verifier, non la forma a
cinque slot calcolata dal sidecar.

Dopo la calibrazione, quando il champion è per esempio K3, il Transformer
DSpark continua a elaborare tutte e cinque le righe non causali ma la catena
Markov autoregressiva, gli argmax e le confidence terminano a K3. Le posizioni
Markov 4 e 5 non possono influenzare i primi tre token e vengono eliminate dal
CUDA Graph del drafter. La telemetria distingue `block=5` da `proposed=K`.

Lo scheduler implementa inoltre la strategia descritta nel paper
[DSpark](https://arxiv.org/html/2607.05147v1): le cinque confidence
condizionali hanno una temperatura distinta (Sequential Temperature Scaling)
e la temperatura viene corretta online usando gli esiti del verifier sul target
Q2/imatrix. Il checkpoint pubblico non include il validation set usato dagli
autori, quindi l'adattamento online sostituisce il fitting offline senza
modificare logits, campionamento o token accettati dal modello target. Dopo la
calibrazione iniziale i candidati K vengono esaminati da sinistra a destra e la
ricerca si arresta alla prima diminuzione del rendimento previsto/misurato.
Sulla GB10 il minimo predefinito è K=2: il costo fisso del verifier rende K=1
non regolare, quindi K1, K2 e K3 vengono sempre esaminati prima che l'early-stop
possa interrompere la scansione. I test locali hanno mostrato il massimo
intorno a K=2/3.
La telemetria aggiunge `sts=[T1 ... T5]` ed `early_stop=K`.

Dopo la calibrazione, se un K ha dimostrato un throughput end-to-end superiore
di almeno l'1% al decode ordinario, diventa il `champion` empirico. In questo
caso il dato hardware prevale sulla confidence assoluta, che sul target
Q2/imatrix può essere calibrata diversamente dal checkpoint originale. Il
circuit breaker K-aware continua a osservare il champion e lo sospende se una
sequenza reale lo rende sfavorevole. La telemetria espone `champion=K` e
l'analizzatore conta quante volte è stato applicato.

L'attivazione del champion è isteretica: entra sopra `1.01x` la baseline,
rimane agganciato durante oscillazioni brevi e viene sganciato soltanto dopo
otto cicli consecutivi sotto `0.98x`. Questo impedisce a un singolo campione
EWMA debole di trasformarsi in migliaia di bypass pre-draft.

La selezione del champion usa ora una media stabile `token emessi / tempo`
separata dall'EWMA veloce del circuit breaker. I primi due cicli di ogni K
vengono esclusi dalla media per non incorporare warm-up di graph e kernel.
L'EWMA resta intenzionalmente sensibile e serve soltanto a riconoscere una
regressione recente; non può più sostituire K=4 con K=1 dopo un singolo ciclo
stocastico sfortunato. La telemetria mostra entrambe come `rate_tps` e
`stable_tps`.

Quando la history è disattivata, un probe esegue sempre il blocco DSpark
completo e lascia allo scheduler la possibilità di riscoprire qualunque K.
Soltanto dopo il nuovo aggancio il percorso Markov viene ristretto al champion.
Il comportamento precedente restringeva anche i probe al vecchio K: se la
EWMA aveva momentaneamente scelto K=1, K=2/K=4 non ricevevano più campioni e il
server rimaneva nel fallback storico. Nel run che ha evidenziato il problema,
K=4 raggiungeva `17.783 t/s`, ma veniva usato 33 volte su 1682 cicli mentre
1490 cicli erano esclusi prima ancora di eseguire DSpark.

Il circuit breaker è a sua volta **K-aware**: valuta soltanto l'EWMA del K
appena scelto, non la media globale contaminata dai K di esplorazione. I decode
ordinari di fallback aggiornano l'anchor ma non alimentano il breaker. Questo
permette al scheduler di mantenere K=2/K=3 quando K=1 o K=5 sono sfavorevoli.
Una scelta isolata `K=0` vale soltanto per il ciclo corrente e non apre più
automaticamente otto cicli di bypass. Quattro K0 consecutivi attivano invece
un cooldown separato di otto decode ordinari, evitando di pagare il drafter
all'infinito quando nessun K è conveniente. Il breaker prestazionale si attiva
soltanto dopo otto osservazioni sfavorevoli dello stesso K selezionato. Nei log
precedenti ogni singolo K0 produceva quasi esattamente `257 * 8 = 2056` bypass;
nel log successivo, senza cooldown, 1074 K0 hanno invece pagato inutilmente il
drafter a ogni ciclo. Entrambi gli estremi sono ora evitati.

`DS4_CUDA_MOE_TINY_DIRECT=1` è un A/B opt-in specifico per i micro-batch del
verifier DSpark. Per 2..6 righe evita count, prefix scan, scatter e costruzione
dei tile per esperto, operazioni progettate per batch di prefill molto più
grandi. I kernel diretti conservano l'indicizzazione `token x expert` e usano
la stessa quantizzazione, gli stessi pesi e la stessa riduzione finale. Il log
`CUDA MoE tiny-batch direct enabled` conferma che il percorso è attivo.
Con `DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY=1` il bypass è limitato al sidecar Q4:
il target Q2 del verifier conserva il raggruppamento per esperto, più adatto a
riusare i pesi su quattro righe.

`DS4_CUDA_Q8_BATCH_REUSE=1` attiva il kernel GB10 per le proiezioni Q8 non
residenti nella cache F16. Usa un blocco CUDA per riga di output e accumula
insieme 2..6 token, leggendo ogni blocco di pesi una sola volta invece di una
volta per token. Quantizzazione, DP4A e albero di riduzione per token restano
invariati. In `run-dspark-server.sh` è attivo di default; impostarlo a `0`
realizza il rollback A/B.

Il verifier K2..K5 non usa più il fallback top-k monothread per scegliere i
token target. Prima della patch, K3 lanciava tre scansioni del vocabolario da
129280 elementi con un solo thread CUDA per riga; soltanto K1 usava l'argmax
parallelo. Ora ogni riga ha un blocco da 1024 thread con la stessa regola di
tie-break sull'indice minore. `DS4_CUDA_NO_BATCHED_ARGMAX=1` ripristina il
percorso precedente esclusivamente per un confronto A/B.

### Speculative rejection sampling DSpark

Il percorso stocastico non usa più `argmax(draft) == sample(target)` come
criterio di accettazione. La correzione Markov produce la distribuzione draft
esatta `q`; un kernel CUDA da 1024 thread applica temperatura e min-p, campiona
il token draft e conserva il vettore normalizzato usato per l'estrazione. Il
verifier target costruisce `p` con la stessa sampling policy e accetta il token
con probabilità:

```text
min(1, p(token) / q(token))
```

Al primo rifiuto campiona il token correttivo dalla distribuzione normalizzata
`max(p-q, 0)`. Questo è il rejection sampling lossless descritto dal paper
DSpark: il target rimane statisticamente identico anche se la distribuzione del
drafter è diversa. Il token correttivo usa il contratto pending già presente:
viene emesso una sola volta e alimentato al target nel ciclo successivo.

La variante GB10 ora sposta anche la verifica `p/q` sul device: `spec_logits`,
`q`, uniformi di acceptance e uniformi residuali restano in CUDA e il CPU legge
soltanto token correttivi, flag di acceptance e la singola riga logits di
continuazione. In modalità rejection il verifier non calcola più i top token
ausiliari, perché non servono al criterio `p/q`. Questo riduce sync e traffico
host nel ciclo caldo K5.

Il percorso CUDA copre la policy usata da Athena e dal thinking DeepSeek:
`temperature > 0`, `top_k=0`, `top_p=1` e `min_p` configurabile (default
`0.05`). Policy con top-k o top-p troncato usano per ora il precedente fallback
target-authoritative e stampano un avviso esplicito. Il greedy resta invariato.
`DS4_DSPARK_REJECTION_DISABLE=1` ripristina l'exact-match per un A/B.

Il launcher di rilascio lascia attivo il path cuBLAS/Tensor Core tiny-batch:
`DS4_CUDA_DSPARK_TENSOR_CORES=1` e `DS4_CUDA_DSPARK_TENSOR_CORES_Q8=1`. Sulla
GB10 questa variante ha prodotto una traiettoria di output percepita come
migliore e mantiene throughput reale nell'area 18 token/s con K4 dominante. Il
rollback prestazionale resta immediato:
`DS4_CUDA_DSPARK_TENSOR_CORES=0` spegne tutto il percorso tiny-TC, mentre
`DS4_CUDA_DSPARK_TENSOR_CORES_Q8=0` conserva Tensor Core solo sui GEMM F16 e
lascia i Q8 al kernel nativo `DS4_CUDA_Q8_BATCH_REUSE=1`.

`run-dspark-server.sh` abilita inoltre `DS4_DSPARK_ALWAYS_DRAFT=1` e disabilita
il circuit breaker prestazionale sulla singola GB10. Lo scheduler continua a
calibrare e selezionare K1..K5, ma non può più entrare nel ciclo chiuso in cui
un bypass storico impedisce di raccogliere i campioni necessari a riattivare
K2/K4. Per ripristinare il gate precedente usare
`DS4_DSPARK_ALWAYS_DRAFT=0 DS4_DSPARK_CIRCUIT_BREAKER=1`.

La telemetria delle righe `dspark timing` espone `rejection=1` e `residual=1`;
`analyze-dspark-log.sh` riporta il numero di cicli p/q e di correzioni residue.
La build esegue anche un test statistico indipendente: per distribuzioni
artificiali `p=(0.7,0.3)` e `q=(0.2,0.8)`, draft + rejection devono ricostruire
`p`; quando `p=q`, ogni proposta deve essere accettata.

### Esperimento expert DSpark MXFP4 nativi: respinto

Il 22 luglio 2026 è stato provato un sidecar alternativo che conservava gli
expert FP4 ufficiali senza il passaggio `FP4 -> F32 -> Q4_K`. Il convertitore
copiava le scale E8M0 e riordinava soltanto le nibble nel layout GGUF MXFP4;
il target `ds4flash.gguf`, il sidecar Q4_K stabile e i pesi numerici FP4 non
venivano modificati. Il runtime collegava poi gate, up e down al MMQ MXFP4 con
attivazioni FP4 e MMA block-scaled native `sm_121a`.

Il test end-to-end sul server Athena ha però mostrato una regressione grave:
`5.94 t/s` nei primi 50 token e `4.86 t/s` nei successivi 50, con media
`5.34 t/s`, contro l'area precedente di circa 19-22 t/s. Durante il run sono
stati istanziati graph ausiliari da circa 4.300-4.800 nodi. Le righe denominate
`CUDA MTP graph` non indicavano un avvio accidentale di MTP: DSpark condivide lo
stesso gestore e le variant osservate `34..54` e `92..95` appartenevano alle
famiglie verifier/draft DSpark.

La conversione lossless aveva superato il self-test, quindi il collo di
bottiglia più probabile non era il formato dei pesi ma il kernel scelto. Il MMQ
generico aggiunge, nei micro-batch DSpark da 1-6 righe, costruzione delle mappe
expert, gather/quantizzazione delle attivazioni e matmul distinti gate/up/down.
Questi costi possono annullare il vantaggio delle Tensor Core FP4; il codice
ufficiale usa invece un `fp4_gemm`/MegaMoE specializzato. Questa spiegazione è
una diagnosi architetturale, non ancora un profilo kernel conclusivo.

Decisione: percorso MXFP4 rimosso dalla working tree, sidecar Q4_K confermato
come default e rollback al commit `04ee700`. Non reintrodurre MXFP4 tramite il
MMQ generico. Un eventuale nuovo tentativo richiede prima un kernel grouped-MoE
FP4 specifico per tiny batch, un microbenchmark isolato che batta Q4_K di almeno
il 5% e soltanto dopo l'integrazione nei CUDA Graph e il test end-to-end.

## Rollback

MMQ e token-tile usano guardie strutturali e non hanno flag runtime nel launcher.
Il rollback corretto è quindi conservare il binario stabile precedente oppure
ricompilare il commit stabile `04ee700` in un checkout separato. Non ripristinare
singoli file CUDA: MMQ coinvolge anche `cuda/mmq`, header e test di regressione.

Rollback e diagnostica ancora supportati:

1. `DS4_CUDA_DROP_COPIED_MODEL_PAGES=0` conserva le pagine sorgente GGUF;
2. `DS4_MEMORY_PROFILE=lean` riduce cache e chunk mantenendo il sidecar copiato;
3. `DS4_MEMORY_PROFILE=prefill-fast` non copia il sidecar, solo per A/B;
4. `DS4_DSPARK_REJECTION_DISABLE=1` ripristina l'exact-match speculativo;
5. `DS4_DSPARK_ALWAYS_DRAFT=0 DS4_DSPARK_CIRCUIT_BREAKER=1` ripristina gate e
   circuit breaker storici;
6. `DS4_CUDA_DSPARK_TENSOR_CORES=0` disabilita il tiny-batch Tensor Core;
7. avviare direttamente `ds4-server` senza `--dspark` esclude il sidecar;
8. `DS4_ADVERTISE_CONTEXT_PCT=100` rimuove il guard client, solo per diagnosi.

Target, sidecar e KV cache non vengono modificati dalla compilazione. Il deploy
rsync non tocca la directory `.git` remota.

## Prossimi interventi ad alto potenziale

L'integrazione DSpark è completa e token-tile HMMA ha superato il gate con
+25,88%, mantenendo 23,58 t/s di decode. Restano da registrare la memoria
residente e da estendere il binding post-tool alla rappresentazione senza
reasoning del client Athena. La protezione NVMe ha gia' eliminato il replay dal
vecchio anchor 24.576 senza regressioni di decode; il passo successivo puo'
rimuovere anche i circa 100-240 ms di caricamento del checkpoint. In seguito,
un nuovo profilo MoE/attention stabilira' quale costo residuo puo' giustificare
un altro incremento end-to-end significativo.

### Graph interamente persistente senza prepare in background

Il dato più importante è:

```text
launches=1000 updates=994 rebuilds=6
```

La pipeline implementata nasconde la ricattura dei circa 1.600 nodi sotto il
calcolo GPU del token precedente. Un progetto successivo potrebbe eliminarla
del tutto spostando anche posizione, righe KV e dimensioni dell'attenzione in
uno stato dinamico residente sulla GPU.

È un intervento più ampio, ma elimina il principale lavoro host rimasto nel
percorso CUDA Graph ed è più promettente di ulteriori micro-ottimizzazioni.

### Sampling probabilistico sulla GPU

Il percorso greedy ora legge soltanto l'ID del token. Le richieste con
temperature/top-p/min-p leggono ancora l'intero vettore per mantenere esatta la
semantica del sampler esistente. Un sampler CUDA completo potrebbe evitare
anche questo readback, ma richiede una validazione statistica separata.

### Kernel GB10 a bassa precisione su Tensor Core

La potenza di picco della GB10 riguarda formati a bassa precisione e Tensor
Core. Molti matmul di decode Q8/IQ2/Q2 usano ancora kernel custom DP4A o
dequantizzazione specializzata. Un kernel W8A8/FP8 o un percorso CUTLASS
specifico per `sm_121a` potrebbe fornire un guadagno maggiore, ma richiede
benchmark di qualità e banda e non è una patch minimale.

### Profondità MTP superiore a due

Il nuovo percorso parte prudentemente da `--mtp-draft 2`. Solo se la telemetria
mostra acceptance elevata conviene provare `DS4_MTP_DRAFT=3` o `4`: aumenta il
lavoro utile per verifier ma anche il costo dei draft rifiutati.

## File modificati

- `Makefile`: target CUDA Graph, build nativa `sm_121a` e oggetti MMQ Entrpi.
- `ds4.c`: DSpark, confidence scheduler, commit diretto, ring rollback, payload
  KV e carry F16 confinato al prefill largo.
- `ds4_cuda.cu`: residenza multi-GGUF, kernel DSpark, graph K-aware, routed-MoE
  MMQ, token-tile HMMA ed epilogue HC/RMS/RoPE fuse confinate al prefill target.
- `ds4_gpu.h`: API interne per token graph, ring backup, verifier DSpark e test
  di parità MMQ/token-tile/epilogue prefill.
- `cuda/mmq/`: kernel llama.cpp e adattatore CUDA importati dal fork Entrpi.
- `ds4_server.c`: speculative sampling DSpark anche con temperatura non nulla.
- `ds4_kvstore.c`: ABI payload 3 per rifiutare checkpoint incompleti.
- `run-mtp-tc-server.sh`: configurazione riproducibile su porta 30007.
- `analyze-mtp-log.sh`: acceptance, tempi MTP e token/s dai log.
- `build-dspark-sidecar.sh`: conversione riproducibile dei soli shard DSpark.
- `run-dspark-server.sh`: configurazione DSpark conservativa per Athena.
- `analyze-dspark-log.sh`: acceptance, K scelti, circuit breaker e throughput.
- `deploy-athena.sh`: rsync compatibile con macOS verso il checkout persistente
  di Athena, seguito da regressione, build e avvio.

Le modifiche sono deliberatamente circoscritte al backend CUDA. I percorsi
esistenti restano regolabili tramite variabili d'ambiente; MMQ usa invece il
guard strutturale descritto sopra. Gli altri backend restano utilizzabili.
