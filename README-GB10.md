# DS4 GB10 + DeepSeek V4 Flash + DSpark

Questa guida descrive la configurazione corrente del fork GB10 per eseguire
DeepSeek V4 Flash con il sidecar DSpark su una singola NVIDIA GB10. La priorità
è mantenere invariata la qualità del modello target, aumentare il prefill e
conservare il throughput DSpark sui contesti lunghi.

La sezione **Avvio rapido** è la procedura operativa aggiornata. Le sezioni
successive conservano la cronologia tecnica, comprese prove scartate e rollback.

## Stato corrente

Stato al 17 luglio 2026:

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

## Requisiti

- NVIDIA GB10 / DGX Spark con Linux ARM64;
- CUDA Toolkit 13 con `/usr/local/cuda/bin/nvcc` e supporto `sm_121`;
- toolchain C/C++ (`make`, `cc`, `git`);
- almeno 128 GB di memoria unificata per il profilo `balanced`;
- spazio disco per target GGUF, sidecar DSpark e KV cache;
- porta TCP 30007 libera.

Verifica minima:

```bash
/usr/local/cuda/bin/nvcc --version && nvidia-smi && git --version && make --version
```

## Avvio rapido

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
/home/athena/ds4/ds4flash.gguf
/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf
```

Il target Q2/imatrix può essere scaricato con:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && mkdir -p /home/athena/ds4 && DS4_GGUF_DIR=/home/athena/ds4 ./download_model.sh q2-imatrix
```

Dopo il download, creare o aggiornare il collegamento atteso dal launcher:

```bash
ln -sfn /home/athena/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf /home/athena/ds4/ds4flash.gguf
```

Per costruire il sidecar servono gli shard Hugging Face 46–48 e
`model.safetensors.index.json` in
`/home/athena/ds4/dspark-v4flash-hf`. Quando sono presenti:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./build-dspark-sidecar.sh 2>&1 | tee /tmp/ds4-dspark-convert.log
```

Percorsi differenti possono essere passati con `DS4_MODEL`,
`DS4_DSPARK_MODEL`, `DS4_DSPARK_HF_DIR` e `DS4_DSPARK_GGUF`.

### 3. Regressione CUDA obbligatoria

Fermare il server prima del test per evitare pressione di memoria, quindi:

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && make -B cuda-regression CUDA_ARCH=sm_121
```

Il comando compila anche gli oggetti `cuda/mmq`. L'esito valido termina con
`cuda long-context regression: OK`; warning, errori di parità, non-finite o
fallimenti precedenti a quella riga bloccano il deploy.

### 4. Build del server

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && make -B cuda-spark-graph-sm121
```

La riga NVCC deve contenere `-arch=sm_121`,
`--default-stream per-thread` e `-DDS4_CUDA_TOKEN_GRAPH_BUILD`.

### 5. Avvio

```bash
cd ~/DS4-GB10-GX10-DSpark-CUDA && ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-dspark-server.log
```

Il launcher usa per default `balanced`, chunk 8192, context 131072, advertise
85%, cache Q8→F16 da 12 GiB, porta 30007 e KV cache in
`/tmp/ds4-gb10-dspark-kv`.

Verifica:

```bash
curl -fsS http://127.0.0.1:30007/v1/models
```

Log principali attesi durante startup e primo prompt:

```text
ds4: CUDA pipelined model copy ...
ds4: CUDA Entrpi batched MMQ MoE prefill enabled (... token-bound stream-K; decode excluded)
ds4: CUDA token-tile HMMA raw/mixed prefill enabled ...
ds4: CUDA token-tile HMMA indexed prefill enabled ...
```

Gli ultimi due messaggi appartengono all'intervento in validazione e compaiono
solo quando una forma eleggibile viene realmente eseguita.

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

## Configurazione corrente

`run-dspark-server.sh` è la fonte autorevole dei default. Il profilo
`balanced` imposta:

```text
DS4_CTX=131072
DS4_ADVERTISE_CONTEXT_PCT=85
DS4_MEMORY_PROFILE=balanced
DS4_PREFILL_CHUNK=8192
DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=112
DS4_CUDA_Q8_F16_CACHE_MB=12288
DS4_CUDA_COPY_SECONDARY_MODEL=1
DS4_CUDA_DROP_COPIED_MODEL_PAGES=1
DS4_CUDA_DSPARK_CACHE_PRIORITY=1
DS4_CUDA_TOKEN_GRAPH=1
DS4_CUDA_DSPARK_GRAPH=1
DS4_DSPARK_ALWAYS_DRAFT=1
DS4_DSPARK_NO_CIRCUIT_BREAKER=1
DS4_PREFILL_FINAL_LOGITS_ONLY=1
DS4_KV_PREFILL_CHECKPOINT_POLICY=canonical-only
DS4_KV_CACHE_COLD_MAX_TOKENS=131072
DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS=65536
DS4_KV_LONG_COLD_ANCHOR_TRIM_TOKENS=8192
```

Il context guard pubblica circa 111411 token e sottrae anche il budget di
output, 2200 token per default. Questo lascia margine per la compaction del
client prima del limite fisico.

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

## Cronologia tecnica

Le sezioni seguenti documentano l'evoluzione del lab. Quando un valore storico
contrasta con **Avvio rapido**, prevalgono sempre i default correnti riportati
all'inizio del documento e in `run-dspark-server.sh`.

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
poi la frontiera completa del prompt viene salvata una sola volta con la chiave
testuale canonica della richiesta, e infine inizia il decode. Non vengono
serializzate frontiere intermedie `continued` durante il prefill.

Se una scrittura SSE fallisce, i token già prodotti ma non consegnati al client
non vengono salvati come nuova risposta e il server non tenta un rewind CUDA nel
job fallito. Viene armato un guard one-shot; soltanto il retry identico può
caricare il checkpoint canonico completo e ripartire senza ripetere il prefill.
Il guard confronta API, tipo richiesta, lunghezza token e SHA-1 del prompt, e
viene consumato anche da una richiesta non corrispondente.

Log attesi:

```text
kv canonical prefill checkpoint tokens=... persisted=1
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
non coincide ancora con quella rappresentazione e compare `live kv cache miss`.
Entra quindi in funzione la seconda rete di sicurezza: il salvataggio
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
retention del prefisso lungo contro un anchor favorito dall'euristica. Rimane
un'ottimizzazione separata: accettare anche la rappresentazione senza reasoning
usata dal client e trasformare il fallback NVMe nel vero hit RAM
`tool live continuation`.

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
9. **Build nativa GB10.** `make cuda-spark-mtp-tc` usa `-arch=sm_121` oltre al
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
make -B cuda-regression CUDA_ARCH=sm_121
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
impiegato 69,999 secondi ed e' ora la parte dominante del caricamento modelli;
non influisce sul risultato token-tile ma identifica il prossimo costo di
startup misurabile.

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
DS4_DSPARK_STS_TEMPERATURES=... cinque temperature iniziali, separate da virgole
DS4_DSPARK_STS_DISABLE=1        disabilita l'adattamento STS online
DS4_DSPARK_CONF_BIAS=...        bias della sigmoid confidence
DS4_DSPARK_CONF_THRESHOLD=...   interrompe K quando confidence scende sotto soglia
DS4_DSPARK_CAUSAL_MIN_K=...     minimo K prima dell'early-stop (GB10 default 2)
DS4_DSPARK_K0_PATIENCE=...      K0 consecutivi prima del cooldown (default 4)
DS4_DSPARK_K0_COOLDOWN=...      decode ordinari dopo la patience (default 8)
DS4_DSPARK_CHAMPION_DISABLE=1   disabilita la scelta del miglior K misurato
DS4_DSPARK_CHAMPION_MARGIN=...  margine sul target ordinario (default 1.01)
DS4_DSPARK_CHAMPION_EXIT_MARGIN margine di uscita isteretico (default 0.98)
DS4_CUDA_MOE_TINY_DIRECT=1      percorso MoE diretto per verifier da 2..6 righe
DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY=1 limita il direct-MoE al sidecar Q4
DS4_CUDA_Q8_BATCH_REUSE=1       riusa i pesi Q8 tra 2..6 righe verifier
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

## Rollback

MMQ e token-tile usano guardie strutturali e non hanno flag runtime nel launcher.
Il rollback corretto è quindi conservare il binario stabile precedente oppure
ricompilare il commit stabile `59a5614` in un checkout separato. Non ripristinare
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
specifico per `sm_121` potrebbe fornire un guadagno maggiore, ma richiede
benchmark di qualità e banda e non è una patch minimale.

### Profondità MTP superiore a due

Il nuovo percorso parte prudentemente da `--mtp-draft 2`. Solo se la telemetria
mostra acceptance elevata conviene provare `DS4_MTP_DRAFT=3` o `4`: aumenta il
lavoro utile per verifier ma anche il costo dei draft rifiutati.

## File modificati

- `Makefile`: target CUDA Graph, build nativa `sm_121` e oggetti MMQ Entrpi.
- `ds4.c`: DSpark, confidence scheduler, commit diretto, ring rollback e payload KV.
- `ds4_cuda.cu`: residenza multi-GGUF, kernel DSpark, graph K-aware, routed-MoE
  MMQ e token-tile HMMA confinati al prefill target.
- `ds4_gpu.h`: API interne per token graph, ring backup, verifier DSpark e test
  di parità MMQ/token-tile.
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
