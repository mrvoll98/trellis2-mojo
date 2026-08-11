# WP10 — Benchmark-resultater

> **Historisk resultatarkiv:** Målt 2026-07-08 med den daværende
> sammenligningsharnessen. Referansedriveren og `pixi run bench` ble fjernet
> 2026-08-11 sammen med framework-avhengigheten. Mojo-mikrobenchmarkene i
> `benchmarks/` kan fortsatt kjøres direkte.

**Miljø:** macOS 27.0 arm64 (Apple Silicon, 14 kjerner), Python 3.14.6,
torch 2.12.0 (CPU), Mojo 1.0.0b2.

## Metodikk

- Casene genereres seedet på Python-siden; interop-konvertering måles IKKE.
- 1 warmup-kjøring per side (fyller også layout-/partisjon-/nabokart-cachene
  → steady-state hot-path-tall), deretter min av 3 målte kjøringer,
  `torch.no_grad()` på torch-siden.
- Torch-baseline for full sparse attention er den **semantisk korrekte**
  per-batch-SDPA-løkka (samme som paritetsfasiten) — originalens naive
  CPU-fallback er feil (attenderer til padding) men vektorisert; dens tall
  står som egen linje i bench-utskriften.
- Windowed-baseline er per-vindu-torch-SDPA-referansen fra paritetstestene —
  originalen har ingen CPU-backend, så OGSÅ torch-tallet der er en naiv
  løkke over vinduer (ikke en produksjonsbackend).
- `slat-sampler` er hele FlowEuler CFG-intervall-trajektorien (8 steg) med en
  liten SLat-flow-modell (rope + share_mod + qk_rms, som ekte sjekkpunkt).

## Sluttresultat etter WP10 pass 1–6 (SIMD + parallellisme + tiling + registerblokker + GEMM/glue + normer)

**Mojo slår torch-originalen på ALLE caser, mot både default-tråder og
1 tråd.**

Torch med default tråder (10 — det en torch-CPU-bruker faktisk får):

| case | størrelse | torch ms | mojo ms | mojo/torch |
|---|---|---:|---:|---:|
| attn-full S | 2×256 tok, H4 D32 | 2.0 | 0.4 | **0.2x** |
| attn-full M | 2×512 tok, H4 D64 | 4.8 | 2.0 | **0.4x** |
| attn-full L | 2×1024 tok, H8 D64 | 16.4 | 15.8 | **0.96x** |
| attn-windowed | 2×2048 tok, H4 D32, vindu 4 | 19.4 | 1.7 | **0.09x** |
| conv3d 3³ | 2×2048 tok, 32→32 | 13.8 | 1.4 | **0.1x** |
| mod-block | 2×512 tok, C128 H4, ffn ×4 | 5.8 | 3.1 | **0.5x** |
| slat-sampler | 2×128 tok, C64 ×2 blk, 8 steg | 76.2 | 20.6 | **0.27x** |

Torch pinnet til 1 tråd:

| case | størrelse | torch ms | mojo ms | mojo/torch |
|---|---|---:|---:|---:|
| attn-full S | 2×256 tok, H4 D32 | 2.1 | 0.5 | **0.2x** |
| attn-full M | 2×512 tok, H4 D64 | 10.2 | 2.1 | **0.2x** |
| attn-full L | 2×1024 tok, H8 D64 | 75.6 | 15.6 | **0.2x** |
| attn-windowed | 2×2048 tok, H4 D32, vindu 4 | 14.4 | 1.7 | **0.1x** |
| conv3d 3³ | 2×2048 tok, 32→32 | 6.3 | 1.4 | **0.2x** |
| mod-block | 2×512 tok, C128 H4, ffn ×4 | 13.1 | 3.1 | **0.2x** |
| slat-sampler | 2×128 tok, C64 ×2 blk, 8 steg | 38.3 | 22.4 | **0.6x** |

(attn-L-ratioen mot default-torch varierer ~0.9–1.05 mellom kjøringer —
i praksis paritet. Merk at torch ofte er RASKERE med 1 tråd enn 10 på små
ops (conv 6.3 vs 13.8 ms, sampler 38 vs 75 ms) — trådpool-overhead dominerer
der; flertrådsgevinsten er reell kun for store SDPA/matmul.)

## Historikk

**Naiv skalar v1** (mot 1-tråds torch): attn-full S/M/L 19.1/173/1429 ms
(8.9x/16x/20x), windowed 21.1 ms (1.5x), conv 25.9 ms (4.3x), mod-block
208 ms (16x), slat-sampler 323 ms (9.4x). Mot torch default-tråder var
attn-L 92x og mod-block 37x.

**Pass 1 — SIMD (paritetssuiten grønn):**
1. `varlen_sdpa` (`sparse/attention/full_attn.mojo`): SIMD-dot (bredde 8)
   over head-dim i qk, av-akkumulering ombyttet til per-nøkkel-axpy inn i
   kontiguøs out-rad, pekertilgang, scores-buffer allokert én gang.
   → attention 5–7.5x raskere (L: 1429→191 ms).
2. `linear` (`modules/nn.mojo`): SIMD-dot over inn-dim.
   → mod-block 208→33 ms, sampler 323→72 ms (sammen med 1).

**Pass 2 — parallellisme + conv (paritetssuiten grønn):**
3. `varlen_sdpa`: `std.algorithm.parallelize` over (segment, head) —
   disjunkte out-regioner og egen scores-scratch per item → bit-identisk
   med seriell sti. → attn-L 191→29 ms.
4. `linear`: parallelize over rader. → mod-block 33→10 ms.
5. `conv3d` (`sparse/conv.mojo`): SIMD-dot over Ci + parallelize over
   utkanal-blokker (hele kantlisten per item, disjunkte kolonner →
   bit-identisk). → 27→1.4 ms.
6. Terskler (flops-proxy) for seriell fallback: parallelisering av små ops
   TAPTE (sampleren 72→81 ms før tuning) fordi spawn/join-overhead ×
   ~400 kall/trajektorie dominerer; attention < 2^19, linear < 2^21 går
   serielt → sampler 65 ms.

**Pass 3 — q-tiling + SIMD i småoperasjonene (paritetssuiten grønn):**
7. `varlen_sdpa`: q-rader prosesseres i tiler på 8 med nøkkel-løkka
   ytterst i både qk- og av-passet — hver k/v-rad gjenbrukes 8× mens den
   er varm i L1; matte og akkumuleringsrekkefølge per (qi, kj) uendret →
   bit-identisk. → attn-M 3.9→3.8, mod-block-attention-delen ned.
8. `LayerNorm32`, `activation` (relu/silu/gelu/gelu-tanh) og `modulate`
   (`modules/nn.mojo`): SIMD over kanal-/flatdimensjonen, samme formler
   lanewise. → sampler 65→57.5 ms, mod-block 10.3→9.3 ms.

**Pass 4 — registerblokker + finere parallel-granulering (paritetssuiten
grønn, alle stier fortsatt bit-identiske):**
9. `varlen_sdpa` qk: KU=2 nøkler × TU=4 q-rader per registerblokk → 8
   uavhengige FMA-akkumulatorkjeder som deler k/q-lastene (én kjede er
   FMA-latensbundet); per-(qi, kj)-matten (SIMD-chunks + reduce_add +
   skalar-rest) er uendret → bit-identisk.
10. `varlen_sdpa` av: hver out-rad akkumuleres i registre over ALLE nøkler
    (chunks på 8/4/1×W lanes) med denominator-divisjonen fusjonert i
    lagringen — per-lane addisjonsrekkefølgen over kj og divisjon-etter-
    full-sum er uendret → bit-identisk. Fjerner load/store av out-raden
    per nøkkel og hele det separate divisjonspasset.
11. `varlen_sdpa` arbeidsdeling: lange q-segmenter deles i chunks på
    QC=64 rader per work-item (samme matte uansett hvilket item som eier
    raden). attn-L hadde bare 2 seg × 8 hoder = 16 items på 14 kjerner
    (span = 2 items ⇒ ~8× effektiv parallellisme); nå 256 items → jevn
    lastbalanse. (Dette var kø-punkt «finere attention-granulering».)
12. `linear` (`modules/nn.mojo`): RU=4 rader × OU=2 utkanaler per
    registerblokk (8 kjeder, delte x/w-laster), parallelisert over
    radblokker i stedet for enkeltrader; per-(rad, utkanal)-matten
    uendret → bit-identisk.
→ attn-L 28.9→16.4 ms, attn-M 3.8→2.2 ms, mod-block 9.3→7.7 ms,
sampler 57.5→48.7 ms.

**Pass 5 — pakket GEMM i `linear` + SIMD i tensor-primitivene
(paritetssuiten grønn):**
13. `linear` (`modules/nn.mojo`): pakket-GEMM-sti for store input — vekten
    pakkes per kall til [k][16]-major-paneler, hver 4-raders x-blokk til
    [k][4], og 4×16-utgangsflisen ligger i 8 SIMD-registre gjennom HELE
    k-løkka (ytre-produkt-formulering: 3 laster per 8 vektor-FMA →
    FMA-bundet). 270–775 GF/s på modellformene (`microbench_linear.mojo`).
    MERK: akkumuleringsrekkefølgen per utgangselement er nå en ren
    sekvensiell sum over k — IKKE bit-identisk med dot-stien (innenfor
    paritetstoleransen; deterministisk, seriell == parallell). Haler i
    rader/kolonner og små input tar fortsatt dot-stien fra pass 4.
14. Profilering med `microbench_block.mojo` viste at mod-block-tiden IKKE
    lå i matmulene (~1 ms) men i lim-oppene: `elemwise_batch`
    (shift/scale/gate) 1.8 ms, reshape/unbind/residual-add ~0.9 ms —
    alle skalare element-løkker, og `_binop_rows` kalte `at()` (som
    rekomputerer `row_size()`) PER ELEMENT.
15. `sparse/tensor.mojo`: SIMD-span-hjelpere (`_copy_span`/`_op_span`/
    `_opscalar_span`, op-grenen løftet ut av løkkene) og omskriving av
    reshape_rows/flatten_leading/_binop_flat/_binop_scalar/_binop_rows/
    select_rows/slice_rows/cat_rows/cat_dim/slice_dim/stack_dim1/unbind —
    verdibevarende kopier/elementvise ops → bit-identisk per konstruksjon.
    → shift/scale 1.16→0.11 ms, gate 0.62→0.06 ms; windowed attention
    (gather/scatter-tung) 6.1→1.7 ms.
16. `activation`: W-alignede chunks parallelisert for store buffere
    (per-element-verdier uendret → bit-identisk); gelu-tanh 0.33→0.17 ms.
→ mod-block 7.7→3.2 ms, sampler 48.7→23.5 ms, attn-windowed 6.1→1.7 ms.

**Pass 6 — normene og rope (paritetssuiten grønn):** profilert med
`microbench_norms.mojo` på modellrealistiske former (4096 tok H16 D64 for
qk_rms/rope-stien de ekte sjekkpunktene bruker; [1,512,16³]/[1,128,32³]
for SS-VAE-dekoderstien):
17. `MultiHeadRMSNorm.forward` (qk_rms, kjøres 2× per blokk): SIMD over
    head-dim (kvadratsum via W-laner + reduce_add — omordnet sum, innenfor
    toleranse), radchunks parallelisert. → 6.3→1.1 ms.
18. rope `_rotate`: (re, im)-parene og (cos, sin)-fasene deinterleaves til
    SIMD-laner, roteres med eksakt samme per-par-formel (bit-identisk) og
    interleaves tilbake; radchunks parallelisert. → embed q+k 5.1→2.3 ms.
19. `GroupNorm32`: (batch, gruppe)-regionen er kontiguøs → mean/varians som
    SIMD-reduksjoner + SIMD normaliseringspass per kanal; (b, g)-items
    parallelisert. → 4.3–8.8→0.6–1.1 ms.
20. `ChannelLayerNorm32` (den klart verste: 20–33 ms — skalar per-posisjon-
    vandring med 16 KB stride): vektorisert over romposisjoner (hver lane
    én posisjon, kanal-akkumulering i samme rekkefølge som skalarsløyfen;
    inv_std via SIMD-sqrt i stedet for `** 0.5` — ULP-nivå), romchunks
    parallelisert. → 20.5/32.9→1.7 ms (12–19x).
→ sampler 23.5→20.6 ms (0.27x; rope+qk_rms ligger i sampler-modellen).
SS-VAE-dekode-stien (WP9 decode_shape/decode_tex) er hovedmottakeren —
gevinsten der synes ikke i bench-casene, men chan-ln/groupnorm kjøres
2× per res-blokk gjennom hele dekoderen.

## Analyse (sluttstatus)

- **Måltallet i `target_metrics.md` («< original i hot paths») er nådd på
  ALLE caser mot BÅDE 1-tråds og default-torch (10 tråder).** Eneste case
  nær paritet er attn-L (0.96x mot default, varierer 0.9–1.05); alt annet
  er 0.09–0.6x.
- Sampling-løkka — det reelle arbeidsflyt-målet — er ~21 ms mot torch'
  ~75 ms (default) / ~38 ms (1 tråd), dvs. 1.8–3.7× raskere enn
  originalen på CPU.
- Lærdom fra pass 5 (profil før optimalisering): mod-block-tiden lå IKKE
  i matmulene men i det skalare tensor-limet (broadcast/copy-ops). Etter
  SIMD-ifisering av primitivene er blokken glue-lett: sdpa ~1.1 ms,
  linears ~1.0 ms, alt annet ~1.0 ms. `benchmarks/microbench_linear.mojo`,
  `microbench_block.mojo` og `microbench_norms.mojo` er beholdt som
  profileringsverktøy.
- Kernel-throughput: `varlen_sdpa` (attn-L) ~3 GFLOP/s naiv → ~260 nå
  (torch-SDPA: ~57 én tråd / ~280 ti tråder). `linear` pakket GEMM:
  270–775 GF/s avhengig av form (torch/Accelerate sgemm er fortsatt
  raskere per GEMM isolert, men matmulene er ikke lenger flaskehalsen).
- Numerikk: alle parallel-/tile-/registerblokk-stier og tensor-primitivene
  er bit-identiske med de serielle; unntaket er `linear`s GEMM-sti
  (pass 5), som endrer akkumuleringsrekkefølge per utgangselement (ren
  sekvensiell k-sum) — deterministisk og innenfor paritetstoleransene
  (hele suiten, inkl. den terskel-følsomme WP9-integrasjonen, er grønn).

**Pass 7 — dense Conv3d (2026-07-09, paritetssuiten grønn):** e2e-smoken
viste at ss-stadiet (519 s ved steps=2) IKKE lå i DiT-en: `linear` måler
700–900 GF/s på de ekte DiT-formene (`microbench_linear.mojo`, nye
real-shape-caser) og én hel dense cross-blokk @ [1, 4096, 1024] tar
~550 ms (`microbench_dit_block.mojo`, ny) → ~13 s/forward, ~55 s
sampling. Lasting tok 1 s (`microbench_load.mojo`). Synderen var dense
`Conv3d` (`modules/conv.mojo`) — SS-VAE-dekoderens hele kostnad — som
fortsatt var WP8-naiv: skalar 7-nivås løkke, én tråd, ~2–3 GF/s (~15–23 s
PER 512-kanals res-conv @16³; dekoderen har mange).
21. `conv3d` vektorisert + parallelisert (`microbench_conv3d.mojo`, ny):
    SIMD-laner over innerste zd-dim (stride 1; interiør-span der alle
    kd-taps er i bounds, W8-chunks + W4-rung så 16³-interiøret på 14
    posisjoner ikke faller til skalar), registerblokk OU=4 utkanaler som
    deler hver x-last (pass 4-oppskriften; partial-blokker får
    én-kanals W8-rung), kant/rest/stride≠1 tar den originale skalarstien.
    Per-element-akkumuleringsrekkefølgen (bias, så c/kh/kw/kd) er identisk
    i alle stier → bit-identisk. (b, o-blokk)-items parallelisert med
    flops-terskel. → res 512→512 @16³ 67 630→750 ms (MÅLT naiv baseline:
    0.86 GF/s → 77; 90x), upsample 512→1024 @16³ →1 541 ms, res 128→128
    @32³ →274 ms, res 32→32 @64³ →96 ms (72–150 GF/s). e2e-smoke
    (steps 2, --no-tex, shoe_3q seed 42, ren kjøring): ss-stadiet
    519→162 s (3.2x), TOTALT 627→271 s — og OBJ-en er BYTE-identisk med
    før-pass-7-referansen (bit-identisk kjerne). Resten av ss-stadiet er
    nå sampling (~4 CFG-forwards à ~13 s DiT-blokker + sdpa, jf.
    microbench_dit_block) + SS-VAE-dekoding ~10 s.

**Pass 8 — flash-sti i `varlen_sdpa` (2026-07-09, suiten grønn):**
`microbench_sdpa.mojo` (ny) med fase-attribusjon via degenererte dimensjoner
(ci=1 isolerer softmax+overhead, co=1 isolerer qk) viste at self-attention
@4096 tokens (284 ms) IKKE var FMA-bundet: qk ~27 ms, av ~126 ms (v
re-streamet per q-rad fra L2), «softmax+overhead» ~122 ms — dominert av
materialiseringen av scores-bufferen ([TQ, 4096], 1 GB skrevet + 3×lest
per forward). Mislykkede hypoteser først (ærlig logg): SIMD-softmax
(exp/max var ikke flaskehalsen), KU 2→4-registerblokk (kjernen var ikke
FMA-bundet), uinit per-item-scratch + TQ 8→32 (bare ~7 %).
22. Flash-sti for segmenter med kv_len ≥ 1024 (`FLASH_KV`): online
    softmax over kv-blokker på KB=128 — scores-flisen (16 KB) og
    akkumulatorene bor i L1, v-blokken gjenbrukes over hele 32-raders
    q-flisen, qk beholder 4×4-registerblokkene og av kjører 2 q-rader
    per blokk med delte v-laster. Numerikk: samme max-subtraherte
    softmax, men løpende maks utløser exp-reskalering av akkumulatoren
    og nevnersummen er blokkvis — deterministisk, IKKE bit-identisk
    (pass 5-GEMM-presedens). ALLE paritetstest-former (< 1024 kv) tar
    fortsatt den eksakte kjernen (som beholdt uinit-scratch/TQ32/KU4 —
    bit-identisk); flash-stien er paritetsverifisert med ekte vekter:
    test-cond cond(512) 3.4e-5→4.1e-5 (1029 tokens; cond(128) UENDRET
    3.147e-5 = eksakt sti bevist bit-identisk), test-real ss_flow
    1.1e-4 (atol 2e-3).
    → self 4096×4096 H16 D64: 284→197 ms (1.44x, 349 GF/s);
    cross 4096×1029: 58→52 ms; DINOv3-self 1029: 15.9→15.1 ms.
    e2e-smoke (steps 2, ren kjøring): ss-stadiet 162→145 s, totalt
    271→252 s; output strukturelt uendret (948 566 V mot 948 568 —
    2 borderline-voxels av ~950k flippet av flash-numerikken, identisk
    32³-okkupans og bbox). WP10-benchen grønn på alle caser etter
    passet (attn-L 0.9x uendret — 2×1024-casen ligger akkurat på
    FLASH_KV-terskelen).
    Gjenværende: flash-qk lider ~40 ms av prefetch-avbrudd per
    kv-blokk (eksakt-stiens qk var 27 ms); ytterligere KB-tuning eller
    k-pakking mulig.

## Gjenstående optimaliseringskø (alt valgfritt — målet er nådd)

1. Tensor-allokering zero-fyller alltid (`List(length=n, fill=0)`) —
   en uinitialisert variant ville kutte ~halvparten av gjenværende
   reshape/copy-kostnad (~0.1 ms per stor op).
2. Større GEMM-registerkjerner / parallell vektpakking hvis ekte
   modellformer (C=1024, ffn 4096) viser behov (per microbench pass 7:
   700–900 GF/s — neppe verdt det).
3. rope `embed` kopierer de cachede fasene per kall (~1 MB) og
   qk_rms/rope-grenen i MHA.forward gjør unbind+stack-kopier — kan
   elimineres hvis profilen på ekte modeller viser dem
   (`microbench_dit_block.mojo`: reshape+unbind ~20 ms av ~550 ms/blokk).
4. ~~Dense self-SDPA @4096 tokens flash-tiling~~ GJORT i pass 8 (1.44x);
   gjenværende sdpa-kø: k-pakking/KB-tuning mot flash-qk-prefetch-tapet,
   evt. GEMM-formulering av qk/av (unngår reduce_add helt).
5. ~~WP11: tiled GPU-GEMM bak `linear`~~ GJORT 2026-07-10 (se under).
   Gjenværende WP11-kø: kjede ops på GPU-en (sdpa → hele DiT-blokken
   resident i device-minne) — transfer-skatten per enkelt-op-kall er
   nå den bindende begrensningen.

## WP11 steg 2 — GPU-linear (2026-07-10, suiten grønn, 13 testfiler)

`trellis2_mojo/gpu/linear.mojo` wirer den tilede Metal-GEMM-en bak
`SparseLinear.forward` (env-flagg `TRELLIS2_GPU=1`, CPU-fallback som
default; se 06_MASTER_PLAN WP11 for arkitektur og de fire nye
API-fellene — bindingsgrense, commit-semantikk, WC-minne, Span-lengde).

`benchmarks/microbench_gpu_linear.mojo` (hele forward inkl. pack,
transfer, bias-readback; min av 3):

| Form (rows × co × ci) | GPU ms | CPU ms | Speedup |
|---|---|---|---|
| 4096 × 3072 × 1024 (qkv) | 20.6 | 32.4 | **1.57x** |
| 4096 × 1024 × 1024 (to_out) | 9.8 | 10.4 | 1.06x |
| 4096 × 4096 × 1024 (mlp-opp) | 25.6 | 42.3 | **1.65x** |
| 4096 × 1024 × 4096 (mlp-ned) | 23.7 | 40.1 | **1.70x** |
| 1029 × 3072 × 1024 (DINOv3) | 10.8 | 9.4 | 0.87x → CPU via terskel |
| 2369 × 4096 × 1024 (slat-mlp) | 19.6 | 26.6 | **1.36x** |

Kjernen alene (`microbench_gpu_gemm.mojo`, remålt med ærlig
enqueue→flush-timing): 2.2–2.7 TF/s på DiT-formene. Gapet ned til
forward-tallene er transfer-skatten: A-opplasting (map-skriv ~8 GB/s)
+ C-readback (WC-les, 9.2 GB/s med 16 parallelle chunks — 2.2 GB/s
seriell!). Dispatch-terskler målt: rows ≥ 512 OG rows·co·ci ≥ 2^32
(4096×1024×1024 er break-even 1.06x; 1029×3072×1024 taper 0.87x).

e2e-smoke steps 2 med `TRELLIS2_GPU=1`: 252→230 s (ss 145→128 s,
shape-slat ~65→55 s); strukturelt identisk output (2369 voxels @32³
eksakt, 948 567 vs 948 568 @512³). Neste sprang krever at ops KJEDES
på GPU-en (sdpa → hel DiT-blokk device-resident) så transfer-skatten
amortiseres.

## WP11 steg 3 — GPU dense SDPA (2026-07-10, suiten grønn, 14 testfiler)

`trellis2_mojo/gpu/attention.mojo`: GEMM-komposisjon med
device-resident scores (se 06_MASTER_PLAN WP11) — qk grid-z-batched
over hoder, maskert softmax med radsummer, av, 1/sum i CPU-readbacken.
`benchmarks/microbench_gpu_attn.mojo` (hele kallet inkl. pack/transfer;
CPU = flash-stien fra pass 8):

| Form | GPU ms | CPU ms | Speedup |
|---|---|---|---|
| self 4096, H16 D64 (ss_flow) | 57.4 | 207.7 | **3.62x** |
| cross 4096×1029, H16 D64 | 20.1 | 58.3 | **2.91x** |
| self 2048, H16 D64 | 19.2 | 55.0 | 2.87x |
| single-seg self 2369, H16 (slat, steg 4) | 23.0 | 67.2 | **2.92x** |
| single-seg cross 2369×1029, H16 (steg 4) | 14.3 | 30.0 | **2.10x** |

Scores-trafikken (1 GB/forward @4096 self — flaskehalsen som motiverte
CPU-flash i pass 8) skjer nå på GPU-ens interne båndbredde. Attention
per ss_flow-blokk: ~266 ms → ~78 ms.

e2e-smoke steps 2 med `TRELLIS2_GPU=1` (steg 2+3): totalt **171 s**
mot 230 s (steg 2 alene) / 252 s (ren CPU) = 1.47x; ss-stadiet
128→**71 s**. OBJ byte-identisk med steg 2-smoken (okkupans-
binariseringen skjuler sdpa-driften).

Med steg 4 (varlen/sparse sdpa, q-padding): totalt **156 s** = 1.62x
mot ren CPU; shape-slat-stadiet 55→**40 s**. Strukturelt identisk
(948 566 vs 948 567 voxels — 1 borderline-flip).

## WP11 steg 5 — device-resident mlp-kjeding (2026-07-10, suiten grønn)

`gpu_mlp_forward`: lin0-GEMM → bias+gelu-tanh-kjerne (tanh via exp —
GPU-bibliotekets tanh avviker ~2e-3!) → lin2-GEMM, intermediatet
forlater aldri GPU-en. `microbench_gpu_linear` (min av 3):

| mlp-form | chain | ukjedet GPU | CPU | chain vs CPU |
|---|---|---|---|---|
| 4096×1536→8192→1536 (ss) | 84.2 ms | 131.8 ms | 279.1 ms | **3.31x** |
| 2369×1536→8192→1536 (slat) | 59.1 ms | 87.9 ms | 178.4 ms | **3.02x** |

e2e-smoke steps 2: totalt **136 s** = **1.85x** mot ren CPU
(ss-sampling 66→59 s, slat-sampling 36→28 s); strukturelt identisk
(12 borderline-flips av ~950k — gelu-numerikk i alle mlp-er).

## WP11 steg 6 — sparse conv på GPU (2026-07-10, suiten grønn, 15 filer)

Decode-instrumentering viste conv-dominans: 23 s ConvNeXt-convs +
mesteparten av 17 s upsample-blokker av 43 s decode. `gpu/conv.mojo`:
CSR-sortert gather-kjerne (tråd = rad × 8 co-laner, x-broadcast delt
over to vec4-akkumulatorer), vekt [K, Ci, Co] på device én gang per
modell, alle dims i edges-headeren (4-peker-marshalling).
`microbench_gpu_conv` (min av 3, dense blobber):

| Form | GPU ms | CPU ms | Speedup |
|---|---|---|---|
| 512ch @ 12k tokens | 216.7 | 657.0 | **3.03x** |
| 256ch @ 55k | 230.4 | 817.6 | **3.55x** |
| 128ch @ 216k | 266.6 | 826.1 | **3.10x** |
| 512→2048 @ 12k (up-conv1) | 994.9 | 2657.7 | 2.67x |

e2e-smoke steps 2: **decode 43→19 s** (ConvNeXt-sum 23→10.9 s,
up-blokker 17→8.4 s), totalt **116 s** = **2.17x** mot ren CPU.
Strukturelt identisk (948 578 voxels begge, bbox lik). Fordeling nå:
ss 61 s, slat 30 s, decode 19 s. Gjenstående GPU-kø:
qkv→(rms/rope)→sdpa→out-kjeding, windowed-batching, fp16,
conv-kjerne-registerblokking (rad-par som deler vektlinjer).

## WP11 steg 7 — device-resident attention-kjeding (2026-07-11)

`gpu_attn_self_chain`: qkv-GEMM → fused bias+rms+rope → head-major
pack → sdpa-komposisjonen → unpack (1/sum fusjonert) → out-GEMM, kun
x opp og [T, C] ned. `microbench_gpu_attn` (hel MHA-forward inkl.
rms/rope, min av 3; «ukjedet» = steg 2-linears + steg 3/4-sdpa med
CPU-rundturer mellom):

| MHA-form (ekte geometri) | chain | ukjedet GPU | CPU | chain vs ukjedet |
|---|---|---|---|---|
| dense 4096×1536, H12 D128 (ss_flow) | 90.3 ms | 171.8 ms | 540.0 ms | **1.90x** (5.98x vs CPU) |
| sparse 2369×1024, H16 D64 (slat) | 28.2 ms | 66.4 ms | 118.3 ms | **2.35x** (4.19x vs CPU) |

e2e-smoke steps 2: totalt **94 s** = **2.68x** mot ren CPU (steg 6:
116 s); ss-stadiet 61→**48 s**, slat 30→**23 s**, decode 18 s.
Strukturelt identisk: EKSAKT samme 948 578 voxels @512³ og 2369 @32³,
2 055 490 vs 2 055 492 triangler (2 borderline-flips av ~2M —
flash-presedens).

## WP11 steg 8 — cross-attention-kjeding (2026-07-11)

Blokk-profilering med GPU på (`microbench_gpu_block.mojo`, ekte
ss_flow-geometri [1, 4096, 1536] H12 D128 ffn 8192) etter steg 7:
self 87.2 ms (kjedet), **cross 92.6 ms (største post)**, mlp 80.6 ms,
glue ~28 ms → 289.6 ms/blokk. `gpu_attn_cross_chain` kjeder q-siden
device-resident (kv + k-rms på CPU, host-pakket). `microbench_gpu_attn`
(hel cross-MHA inkl. kv-linear/k-rms; min av 3):

| Cross-form (ekte geometri) | chain | ukjedet GPU | CPU | chain vs ukjedet |
|---|---|---|---|---|
| dense 4096×1029, C1536 H12 (ss_flow) | 51.4 ms | 86.3 ms | 155.2 ms | **1.68x** (3.02x vs CPU) |
| dense 2369×1029, C1024 H16 (slat-geom.) | 28.1 ms | 44.0 ms | 61.3 ms | **1.56x** (2.18x vs CPU) |

Blokk-totalen (ss-geometri): 289.6→**244.7 ms**. Gate-notat:
`wants_cross` er KUN sdpa-gaten — slat-cross vinner 1.56x selv om
q-linearen alene er under proxy-terskelen (den rir på opplastingen
sdpa-en trenger uansett).

e2e-smoke steps 2: totalt **88 s** = **2.86x** mot ren CPU (steg 7:
94 s); ss-stadiet 48→**45 s**, slat 23→**20 s**, decode 19 s.
Strukturelt identisk: 948 577 vs 948 578 voxels (1 borderline-flip),
bbox lik.

## WP11 steg 9 — conv-registerblokking, rad-par (2026-07-11)

`sparse_conv_gather`: 2 rader × 8 co-laner per tråd, kantlistene
merge-vandret på kidx → vektlinjer lastes én gang for begge rader der
offsetet finnes i begge. Bit-identisk per rad (samme kantrekkefølge og
akkumulator-matte). `microbench_gpu_conv` (inkl. CSR-bygg + transfer,
min av 3; «steg 6» = enkeltrad-kjernen):

| Form | steg 9 | steg 6 | CPU | steg 9 vs CPU |
|---|---|---|---|---|
| 512ch @ 12k | 152.4 ms | 216.7 ms | 657.0 ms | **4.31x** |
| 256ch @ 55k | 146.6 ms | 230.4 ms | 796.3 ms | **5.43x** |
| 128ch @ 216k | 174.3 ms | 266.6 ms | 883.4 ms | **5.07x** |
| 512→2048 @ 12k (up-conv1) | 644.9 ms | 994.9 ms | 2521.4 ms | **3.91x** |

e2e-smoke steps 2: totalt **86 s** = **2.93x** mot ren CPU (steg 8:
88 s); decode-stadiet 19→**17 s** (resten av decode er norm/mlp-glue,
CSR-bygg og transfers). OBJ-en er BYTE-IDENTISK med steg 8-smoken
(`cmp`) — kjernen er bit-identisk per rad som designet.

KØ-NOTAT (2026-07-11): «windowed attention-batching» er STRØKET —
slat-DiT-en (structured_latent_flow.py:71) bruker `attn_mode='full'`
for alle blokker, så windowed attention er ikke i 512-runner-stien
overhodet.

## WP11 steg 10 — hel-blokk-residens (2026-07-11)

`gpu/block.mojo`: hele cross-blokken (dense + sparse deler
orkestratoren) i ÉN kø med én x-opplasting/readback; glue
(ln+modulate, gate+add med out-bias-fold) som GPU-kjerner; kjedene
refaktorert til enqueue-deler. `microbench_gpu_block` (ss-geometri
[1, 4096, 1536] H12 D128 ffn 8192, min av 3):

| | per-op (steg 7–9) | hel-blokk (steg 10) |
|---|---|---|
| glue (CPU → GPU) | 27.9 ms | i køen |
| self / cross / mlp | 88.0 / 51.2 / 81.2 ms | i køen |
| **blokk totalt** | **249.2 ms** | **211.4 ms (1.18x)** |

e2e-smoke steps 2: totalt **77 s** = **3.27x** mot ren CPU (steg 9:
86 s); ss-stadiet 45→**40 s**, slat 20→**16 s**, decode 17 s.
Strukturelt identisk: 948 565 vs 948 577 voxels (12 borderline-flips
av ~950k — LN-numerikken endret seg i alle blokker, samme klasse som
gelu-passet), samme 2369 @32³, bbox lik.

**Målte NEGATIVE resultater (2026-07-11, microbench_gpu_gemm på ekte
DiT-former)** — to kø-punkter LUKKET uten wiring:
- GEMM-registerblokking: 64×128/4×8-variant ga bare +6 % (2.87→3.06
  TF/s på qkv-formen), 128×128/8×8 +3 % — kjernen er ikke
  register-bundet. Ikke verdt padding-endringene.
- bf16-lagret B-vekt (u16<<16-konvertering på shared-fill; EKSAKT for
  DiT-ene siden sjekkpunktene ER bf16): FLAT (±1 % — 2889 vs 2867
  GF/s). Kjernen er ikke B-båndbredde-bundet: B-flisene L2-caches.
  fp16/bf16-vektlagring gir altså KUN lastetid/minne, ikke kjernetid.
Kjernen ligger på ~2.9 TF/s av ~9 teoretisk — begrensningen er
shared-memory-/issue-rate, og uten simdgroup_matrix-tilgang i 1.0.0b2
er videre GEMM-tuning lav-ROI.

## WP11 steg 11 — CSR-caching (2026-07-11)

Conv-CSR-sorten (avhenger kun av kantene) spatial-caches nå per
coords/kernel/dilation i `SparseConv3d.forward`; int32-packen fylles
chunk-parallelt i `GpuSparseConv.forward`. Bit-identisk — steg
11-smoke-OBJ-en er BYTE-identisk med steg 10 (cmp). Microbench:
sort-kostnaden borte på cache-treff (128@216k: 174→161 ms); e2e-smoke
77 s uendret (besparelsen ~0,5–1 s er under utskriftsgranulariteten).

## WP11 steg 12 — modellnivå-residens (2026-07-11)

`gpu_cross_block_forward` splittet i upload/enqueue/readback-
primitiver; begge DiT-forwardene holder x device-resident over ALLE
30 blokker (én opplasting/readback per forward, faser én gang;
bk/kv-uploadene er inter-blokk-syncer). BIT-identisk med per-blokk-
stien (2-blokks residens-driver vs sekvensiell: diff == 0.0 EKSAKT;
smoke-OBJ BYTE-identisk med steg 11). e2e-smoke steps 2: totalt
**72 s** = **3.50x** mot ren CPU (steg 10/11: 77 s); ss-stadiet
40→**37 s**, slat 16→**14 s**, decode 17 s.

## WP11 steg 13 — sdpa-gate-gulv 1024 + golden GPU-verifisert (2026-07-11)

Golden 12-stegs-kjøringen landet på 1857 slat-tokens — UNDER det gamle
2048-gulvet, så hele slat-DiT-en falt til CPU (177 s). Målt
(microbench_gpu_attn, H12 D128): 1857 self **3.35x**, 1280 2.14x,
1024 fortsatt **1.81x** på GPU → `GPU_SDPA_MIN_Q` senket 2048→1024
(gate-sjekkene i testen flyttet til 512; under 1024 er umålt).

**Golden GPU-kjøring** (12 steg + tekstur, shoe_3q seed 42):
totalt **244 s = 4.1 min** mot CPU-goldenens 27.4 min = **6.74x**
(gammelt gulv: 449 s — slat 177→48 s, tex-slat 103→30 s).
Strukturelt mot CPU-goldenen (cKDTree/trimesh, samme metode):
514 604 vs 514 603 voxels @512³ (1857 @32³ EKSAKT likt), V-avvik
0.000 % / F-avvik 0.001 %, bbox-diff 0.0, NN-avstander mean 2.9e-08 /
p95 4.9e-08 / max 8.8e-04 (< ½ voxel; CPU-vs-mac-referansen lå på
mean 2.1e-3), 0 degenererte flater, tex-npz [514 604 × 6] i
[-0.011, 1.002]. Artefakter: `outputs/shoe_3q_mojo_gpu_seed42.obj`
+ `_texvoxels.npz`.

## WP11 steg 14 (2026-07-11): 16-bits W^T-lagring på device

bf16-vektlagringen fra negativ-resultatene er wiret (`GpuLinear`
lagrer W^T som u16-bits — bf16 for DiT-ene, f16 for fp16-dekoderne —
når hver vekt er bit-eksakt representerbar; ekspansjon på shared-fill,
dispatch via `enqueue_gemm` på alle vekt-GEMM-kallstedene). Som målt i
negativene er kjernetiden FLAT; gevinsten er minne/transfer:
device-W^T per 1.3B-DiT ~4.8→2.4 GB (unified memory). e2e-smoke
(steps 2): 71–72 s uendret, stage-tider identiske, OBJ BYTE-identisk
(ekspansjonen er eksakt — ingen numerikk-variant); klassifiserings-
skannet koster ~det halverte WC-pakket sparer (+9 s user-tid, usynlig
i veggtid). MAX-RSS uendret ~11–12 GB — prosess-peaken ligger i
decode-stadiets aktiveringer + A/C-scratchene, ikke i DiT-vektene.

## WP11 steg 15 (2026-07-11): f16-lagret sparse-conv-vekt

Steg 14-mønsteret på GpuSparseConv (f16-bits, hardware-cast per
vektlinje-load; delt `wfmt_scan`). MERK ulikt GEMM-en: gather-kjernen
STREAMER vektlinjene i stedet for å L2-cache dem som B-flisene, så
halvert vekttrafikk gir reell ytelse på de vekt-tunge formene
(microbench_gpu_conv, nå f16- vs f32-lagret W på SAMME f16-kvantiserte
verdier — decoder-vektene ER fp16):

| form | gpu f16-W | gpu f32-W | f16-gevinst | vs CPU |
|---|---|---|---|---|
| 512ch @ 12k | 131.7 ms | 150.7 ms | **1.14x** | 5.2x |
| 256ch @ 55k | 138.9 ms | 142.6 ms | 1.03x | 6.6x |
| 128ch @ 216k | 160.1 ms | 161.5 ms | 1.01x | 5.5x |
| up-conv1 512→2048 | 562.3 ms | 636.1 ms | **1.13x** | 5.2x |

e2e-smoke: decode **17→14–15 s**, totalt 71 s; OBJ BYTE-identisk
(f16-ekspansjonen er eksakt). Devicevekt-avtrykket for dekoderne
halvert på toppen av steg 14. GOLDEN re-verifisert etter steg 14+15
(12 steg + tekstur): OBJ OG tex-npz BYTE-identiske med steg
13-artefaktene — inkluderer tekstur-stien som smokene aldri kjører;
247 s totalt (244 s i steg 13 — støy; decode 10+11→9+10 s).

## WP17 (2026-07-11): hodegruppert sdpa + 1024-kaskaden

SJETTE b2-Metal-felle (probet): kjerneskriv forbi 4 GiB byte-offset i
én binding tapes stille → full-H scores for 1024-kaskadens HR-slat
(T≈12k → 9 GB) er umulig. Løsning: `_enqueue_sdpa_groups` kjører
qk→softmax→av i HODEGRUPPER mot én scores-scratch ≤ 2^28 floats
(uendret tak); gaten ble per-hode. hg == h er identisk med gammel
sti — alle paritetstall uendret; ny 15+1-gruppers-case ≤ 2e-7.

| kjøring @1024 | CPU-attention | GPU (hodegruppert) |
|---|---|---|
| HR-slat-sampling (steps 2, 11.9k tokens) | 698 s | **120 s (5.8x)** |
| smoke totalt (steps 2, --no-tex) | 836 s | **256 s** |

Golden @1024 (12 steg + tekstur, seed 42): **836 s = 13.9 min**
(512-golden: 244 s) — 7545 tokens @64³ → **2 058 563 voxels @1024³ =
4.00x** 512-goldenen → 2.06M V / 4.16M F; fordeling ss 138 s,
LR-slat 49 s, HR-slat 346 s, tex-slat 200 s, decode 33+38 s,
hullfylling+GLB 2 s (2080 hull); peak RSS 16.6 GB. GLB-sjekk grønn
(COLOR_0 3e-7 mot sparse numpy-referanse).

Gjenstående GPU-kø: TOM — WP11-køen er helt høstet.
Fordeling smoke @512 (steps 2): ss 37 s, slat 14 s, decode 14–15 s,
cond 2 s; full kjøring @512 (steps 12): ss 138 s, slat 48+30 s,
decode 21 s (før steg 15).
