# 06 — Masterplan: TRELLIS.2 → Mojo

**Dette er den operative planleggeren for porteringen.** `03_ROADMAP_PHASES.md` gir
grovfasene; dette dokumentet bryter dem ned i arbeidspakker (WP) med avhengigheter,
konkrete filer, akseptkriterier og verifiseringsporter. Fremdrift per fil spores i
[07_PORT_TRACKER.md](07_PORT_TRACKER.md).

> **Arkivmerknad 2026-08-11:** Planen under beskriver den opprinnelige
> hybrid-/paritetsfasen. Inferensstien er senere fullført som native Mojo, og
> interop- og referanseharnessen er fjernet. Se README og
> [PURE_MOJO_RUNTIME.md](PURE_MOJO_RUNTIME.md) for gjeldende løsning.

---

## 1. Omfang (fra ADR-ene)

| Beslutning | Konsekvens for planen |
|---|---|
| Hybrid, ikke full port (ADR 0001) | Python beholder orkestrering (pipelines, `from_pretrained`, device-håndtering). Mojo tar kjerne-datastrukturer og kernels. |
| Inferens først (ADR 0002) | `trainers/`, `datasets/`, `train.py`, `data_toolkit/` er **utenfor omfang** for v1. |
| Egen SparseTensor-design (ADR 0003) | `modules/sparse/basic.py` porteres som Mojo-structs, ikke som wrapper rundt torch. |
| o-voxel beholdes som FFI (ADR 0004) | `o-voxel/src/**` (C++/CUDA) porteres **ikke**. Kalles via eksisterende Python-bindinger. *(Delvis revidert av ADR 0007: kun mesh-ekstraksjonsstien porteres til Mojo.)* |
| Eksterne modeller forblir Python | DINOv2 (bildefeatures), BiRefNet/rembg — kalles via Python-interop. *(Revidert av ADR 0007: DINOv3 porteres i WP13; rembg omgås med RGBA-krav.)* |

**Målbilde v1:** `Trellis2ImageTo3DPipeline` kjører ende-til-ende der sampling-løkken,
transformer-blokkene og sparse-operasjonene kjører i Mojo, med numerisk paritet mot
golden outputs fra original.

---

## 2. Arbeidspakker

Avhengighetsgraf (piler = «krever»):

```
WP0 baseline ──► WP1 interop ──► WP2 flow_euler (quick win)
                     │
                     ├──► WP3 SparseTensor/VarLenTensor
                     │         │
                     │         ├──► WP4 småopps (linear/norm/nonlin/RoPE/utils)
                     │         │        │
                     │         │        ├──► WP5 attention (full/windowed/serialized)
                     │         │        └──► WP6 spatial + serialize
                     │         └──► WP7 conv ("none"-backend)
                     │
                     └──► (WP4–WP7) ──► WP8 transformer-blokker ──► WP9 modeller
                                                                        │
                                                          WP2 ─────────►│
                                                                        ▼
                                                              WP10 pipeline E2E
```

Rekkefølgen er valgt slik at hver pakke kan paritetstestes isolert før neste starter,
og slik at WP2 gir en tidlig, liten suksess som validerer hele interop-oppsettet.

---

### WP0 — Baseline og golden outputs
**Mål:** Fasit å måle paritet mot. *Blokkerer alt annet verifiseringsarbeid.*

- Kjør original `example.py` (og `example_texturing.py`) i fungerende CUDA-miljø.
- `baselines/capture_baseline.sh` finnes — utvid til å dumpe **mellomresultater**, ikke
  bare sluttresultat: (a) DINO-features, (b) SS-flow latent per sampler-steg,
  (c) SLat etter dekoding, (d) endelig mesh/voxel. Lagres som `.npz` under `baselines/golden/`.
- Dokumentér seeds, dtype (fp16/fp32), GPU og versjonslåser i `baselines/ENVIRONMENT.md`.

**Ferdig når:** `baselines/golden/` inneholder daterte dumps + miljønotat, og
`05_VERIFICATION_PLAN.md` peker på dem.

---

### WP1 — Verktøykjede og interop-skjelett
**Mål:** Ett kommando-drevet oppsett der Mojo-kode kan kalles fra Python-testene.

- Pixi/Magic-miljø med Mojo + MAX; lås versjon i `pixi.toml`.
- Opprett `trellis2_mojo/` (Mojo-pakke) med `__init__.mojo` og build-oppsett.
- Velg og dokumentér interop-mekanisme (ADR 0006): Mojo→Python via
  `PythonObject` for orkestrering, Python→Mojo via MAX custom ops eller kompilert
  shared lib for kernels. **Dette er en reell designbeslutning — skriv ADR før koding.**
- Paritetstest-harness: `tests/parity/` med hjelper som laster golden `.npz`, kjører
  Mojo-motpart, sammenligner med toleranse (`rtol`/`atol` per dtype, definert i
  `conversion/numerical_parity.md`).
- Smoke-test: Mojo-funksjon som tar imot numpy-array, dobler den, returnerer.

**Ferdig når:** `pixi run test-parity` kjører smoke-testen grønt.

---

### WP2 — FlowEuler-sampler (quick win)
**Mål:** Første ekte port; ren matematikk uten sparse-avhengigheter.

| Kilde | Linjer | Mojo-mål |
|---|---|---|
| `pipelines/samplers/base.py` | 18 | `trellis2_mojo/samplers/base.mojo` |
| `pipelines/samplers/flow_euler.py` | 208 | `trellis2_mojo/samplers/flow_euler.mojo` |
| `pipelines/samplers/classifier_free_guidance_mixin.py` | ~20 | inlines i flow_euler |
| `pipelines/samplers/guidance_interval_mixin.py` | ~25 | inlines i flow_euler |

Merk: mixins/arv finnes ikke i Mojo — kollaps CFG/guidance-interval til parametre
eller egne structs (se `conversion/mojo_idioms_mapping.md`).
Modellkallet abstraheres som callback/trait slik at sampleren kan testes mot en
dummy-modell i Python.

**Ferdig når:** Sampler-steg-for-steg-output matcher golden SS-flow-latents (WP0-b)
innen toleranse, med original PyTorch-modell som «modell» via interop.

---

### WP3 — SparseTensor / VarLenTensor
**Mål:** Kjernedatastrukturen. **Størst enkeltrisiko i hele porten** (836 linjer,
alt annet avhenger av den).

- Kilde: `modules/sparse/basic.py` → `trellis2_mojo/sparse/basic.mojo`.
- Følg design i `decisions/0003` og `conversion/sparse_tensor_in_mojo.md`:
  coords (int32 N×4) + feats (N×C) + layout/offsets, ingen torch-avhengighet.
- Port i denne rekkefølgen: konstruksjon → indeksering/`__getitem__` → elementvise
  ops → `sparse_cat`/`sparse_unbind` → `varlen_*` → dtype/device-håndtering.
- `modules/sparse/config.py` (backend-valg) forblir Python inntil WP6.

**Ferdig når:** Egen enhetstestfil kjører alle ops mot torch-referanse på tilfeldige
sparse-tensorer (fuzz med faste seeds), 0 avvik utover dtype-toleranse.

---

### WP4 — Små moduler: linear, norm, nonlinearity, RoPE, utils
**Mål:** Byggeklossene attention og blokker trenger. Alle er små og uavhengige —
kan porteres parallelt etter WP3.

| Kilde | Linjer | Merknad |
|---|---|---|
| `modules/utils.py` | 87 | `manual_cast`, `modulate`, `str_to_dtype` |
| `modules/sparse/linear.py` | ~20 | matmul over feats |
| `modules/sparse/norm.py` | 64 | Group/LayerNorm over layout-grupper |
| `modules/sparse/nonlinearity.py` | ~30 | ReLU/SiLU/GELU elementvis |
| `modules/norm.py`, `modules/spatial.py` | små | dense motparter |
| RoPE (`SparseRotaryPositionEmbedder` i `attention/modules.py`) | del av 141 | ren matematikk, port tidlig |

**Ferdig når:** Hver modul har paritetstest mot torch-motpart (tilfeldig input, fast seed).

---

### WP5 — Attention
**Mål:** De tyngste kernels. Startes først når WP3+WP4 er grønne.

| Kilde | Linjer | Mojo-mål |
|---|---|---|
| `modules/attention/full_attn.py` (dense) | 145 | naiv SDPA først, optimaliser etterpå |
| `modules/sparse/attention/full_attn.py` | 255 | varlen-SDPA over layout |
| `modules/sparse/attention/windowed_attn.py` | 190 | vindu-partisjonering + SDPA |
| `modules/sparse/attention/serialized_attn.py` | — | krever encode/decode_seq (WP6) |
| `modules/sparse/attention/modules.py` | 141 | `SparseMultiHeadAttention` (limet) |

Strategi: korrekthet før ytelse. Første versjon er naiv O(N²)-SDPA i Mojo;
flash-style kernel er egen oppfølgingsoppgave i WP11 (ytelse), ikke blokkerende.

**Ferdig når:** Alle attention-varianter matcher `xformers`/torch-referanse på
golden-input innen fp16-toleranse.

---

### WP6 — Spatial, serialize og conv
**Mål:** Resten av sparse-operasjonene.

- `modules/sparse/spatial/basic.py` (108), `spatial2channel.py` (93):
  down/upsample, subdivide, interpolering.
- `modules/sparse/serialize.py`: z-order/Hilbert-koding. Merk: kan låne logikk fra
  `o-voxel/src/serialize/*.cu` som referanse, men porteres til Mojo (brukes av
  serialized attention).
- `modules/sparse/conv/conv_none.py` (133): ren gather/scatter-conv — **eneste
  conv-backend som porteres.** `conv_spconv.py`/`conv_flex_gemm.py` forblir
  Python-fallbacks bak `config.py`-flagget.

**Ferdig når:** Paritetstester grønne; backend-velger kan velge `mojo`-backend.

---

### WP7 — Transformer-blokker
**Mål:** Komponer WP4–WP6 til blokker.

| Kilde | Linjer |
|---|---|
| `modules/transformer/blocks.py` + `modulated.py` | 185 + 164 |
| `modules/sparse/transformer/blocks.py` + `modulated.py` | 145 + 166 |

nn.Module-hierarkiet flates til structs med eksplisitt `forward`; vekter lastes inn
via safetensors-plan (`conversion/safetensors_loading.md`).

**Ferdig når:** Én enkelt blokk med ekte vekter fra sjekkpunkt matcher torch-blokk
på golden aktivering.

---

### WP8 — Modeller
**Mål:** Fulle nettverk, i stigende vanskelighetsgrad:

1. `models/sparse_structure_flow.py` (247) — **dense** DiT, enklest.
2. `models/sparse_structure_vae.py` (306) — dense VAE (kun decoder trengs for inferens).
3. `models/structured_latent_flow.py` (207) — sparse DiT.
4. `models/sc_vaes/sparse_unet_vae.py` (521) + `fdg_vae.py` (125) — sparse VAE-dekodere.
5. `models/sparse_elastic_mixin.py` — **droppes i v1** (elastisk minne er treningsoptimalisering; noter i ADR).
6. `modules/image_feature_extractor.py` (127) — forblir Python (DINOv2-kall).
   *(Revidert av ADR 0007: DINOv3-ekstraktoren porteres i WP13.)*

**Ferdig når:** Hver modell, lastet med ekte vekter, matcher golden mellomresultat
(WP0-b/c) på fast seed.

---

### WP9 — Pipeline ende-til-ende
**Mål:** `Trellis2ImageTo3DPipeline` (596 linjer) kjører med Mojo-komponenter.

- `pipelines/base.py` + `trellis2_image_to_3d.py` forblir Python (per ADR 0001), men
  bytter komponent-for-komponent til Mojo via feature-flagg
  (`TRELLIS_BACKEND=mojo|torch` per delkomponent — muliggjør bisect ved avvik).
- `representations/` (mesh/voxel) og `renderers/` forblir Python i v1.
- o-voxel `postprocess.to_glb` via eksisterende FFI (ADR 0004).
- `trellis2_texturing.py` (408) er **fase 2** — etter image-to-3d er i mål.

**Ferdig når:** `example.py` mot Mojo-backend produserer GLB som matcher golden
(mesh-metrikker fra `benchmarks/comparison_plan.md`), og ett komplett kjøreeksempel
er dokumentert i README_MOJO.md.

---

### WP10 — Herding og benchmarks
- Benchmark sampling-løkke og attention mot torch-baseline (`benchmarks/target_metrics.md`).
- Ytelsesarbeid (flash-attention-kernel, fusjonering) som egne oppgaver — kun
  etter paritet.
- App-er (`app.py`) og tester poleres.

---

### WP11 — fp16/GPU (kartlagt 2026-07-09, implementasjon gjenstår)
Reservert som notert i 08_HANDOVER/tracker: dtype er strukturparameter i
Mojo-porten; fp16 og GPU-backends tas etter at ren-Mojo-sporet er i mål.

**Kartlegging (2026-07-09, pinnet modular 26.4 / Mojo 1.0.0b2):**
- **Metal-GPU-en er tilgjengelig fra ren Mojo på denne maskinen**:
  `std.gpu.host.DeviceContext()` gir `api: metal, name: Apple M4 Pro`;
  `enqueue_create_buffer` + `map_to_host` + `enqueue_function[kernel]`
  (IKKE `enqueue_function_checked` — finnes ikke i 1.0.0b2) med
  `thread_idx/block_idx/block_dim` fra `std.gpu` er verifisert med
  kjørende kjerner.
- **PoC-tak**: naiv GEMM (én tråd per element, ingen threadgroup-tiling)
  på 4096×1024×1024 f32: 147 GF/s — TAPER mot CPU-pakket-GEMM (830
  GF/s). Threadgroup-/shared-memory-tiling er nødvendig; M4 Pro-taket er
  ~9 TF/s f32, så potensialet er ~10x over CPU for GEMM-tunge stier
  (DiT-forwards).
- **Steg 1 UTFØRT (2026-07-09): tiled GEMM-kjerne skrevet og målt** —
  `benchmarks/microbench_gpu_gemm.mojo`: 64×64-threadgroup-fliser (BK=16)
  i shared memory (`std.memory.stack_allocation[...,
  address_space=AddressSpace.SHARED]` + `std.gpu.barrier` — begge
  verifisert på Metal), 4×4-registerblokk per tråd, kooperative laster.
  **2.7–3.0 TF/s på DiT-formene (3.2–3.6x over CPU-ens 830 GF/s)**,
  korrekthets-spot-check grønn. Krever M%64 == N%64 == K%16 == 0 —
  wiring må padde M (token-antall er vilkårlig: 1029, 2369, ...).
- **API-feller (verifisert)**: `enqueue_function_checked` finnes ikke i
  1.0.0b2 (bruk `enqueue_function`); `enqueue_create_host_buffer` finnes
  og kjernel-launch mot den feiler IKKE, men skrivene når aldri dataene
  (STILLE no-op på Metal!) — bruk device-buffere + `map_to_host`.
- **Steg 2 UTFØRT (2026-07-10): wiring bak `linear`** —
  `trellis2_mojo/gpu/linear.mojo`: delt `GpuContext` (DeviceContext +
  ArcPointer-delt grow-only A/C-scratch + 1-elements fence-buffer),
  skapt av `TRELLIS2_GPU=1` (`gpu_context_from_env`, CPU-fallback
  ellers) i runneren og sendt inn i checkpoint-loaderne; konteksten RIR
  PÅ StateDict (`sd.gpu`) så ingen mellomliggende loader-signatur
  endres — `lin_from`/dinov3s `_lin_from` laster opp W^T én gang per
  modell-lasting (`GpuLinear.try_build`) og `SparseLinear.forward`
  dispatcher på form (co%64==0, ci%16==0, vekt ≥ 2^19, rows ≥ 512 og
  rows·co·ci ≥ 2^32 — målt break-even). Bias adderes på CPU i
  readback-passet (marshalling-grensen under gjør en bias-peker
  umulig, og GEMM-folding kostet en per-rad-repack). Paritet:
  `pixi run test-wp11` (I test-all) — 5 DiT-former + 3D-dispatch +
  declines + StateDict-ride-along, max|diff| ≤ 4.3e-6 (atol 2e-4;
  GPU-en er bit-identisk med NAIV seriell CPU-dot, toleranse mot
  SIMD-stiene).
- **Målt (microbench_gpu_linear.mojo, hele forward inkl. transfer)**:
  qkv 4096×3072×1024 1.57x, mlp-opp 1.65x, mlp-ned 1.70x, 2369-tokens
  mlp 1.36x; to_out @4096 1.06x (break-even), DINOv3-qkv @1029 0.87x —
  derav flops-proxy-terskelen. Kjernen alene: 2.2–2.7 TF/s (remålt med
  korrekt flush-timing).
- **NYE API-feller (verifisert empirisk 2026-07-10, full liste i
  gpu/linear.mojo-headeren)**: (1) arg-marshalling ryker over 4
  bindinger totalt (alle skalarer teller som ÉN binding): 3 ptr +
  skalarer OK, 4 ptr + 0 skalarer OK, 4 ptr + skalarer eller 5+ ptr
  gir søppel-adresser i kjernen. (2) INGENTING committes før en
  map_to_host av en HOST-SKREVET buffer — `ctx.synchronize()`
  committer IKKE pending arbeid, og å mappe en aldri-host-skrevet
  buffer returnerer stale data for alltid; fence-bufferen (re-skrevet i
  hver barriere) er commit+vent-primitivet. Mikrobenchens gamle
  per-iterasjons-timing målte et én-iterasjon-forskjøvet vindu (hver
  enqueue committer forgjengeren) — steg 1-tallene var derfor omtrent
  riktige; benchen er skrevet om til ærlig enqueue→flush-vindu.
  (3) Mappet host-peker er write-combined minne: skriv ~8 GB/s,
  enkelttråds LES ~2.2 GB/s (enqueue_copy d2h bruker samme trege sti
  synkront); parallelle les skalerer 4.2x til ~9.2 GB/s — readback er
  derfor chunket-parallell fusjonert bias+kopi rett fra mappet peker.
  (4) enqueue_copy(Span, buf)-lengden IGNORERES — kopierer alltid hele
  bufferlengden (heap-korrupsjon om target er mindre).
- **Steg 3 UTFØRT (2026-07-10): dense SDPA på GPU** —
  `trellis2_mojo/gpu/attention.mojo`: GEMM-komposisjon med
  device-residente intermediater (scores-matrisen — 1 GB/forward-
  trafikken som motiverte CPU-flash i pass 8 — forlater aldri GPU-en):
  qk-GEMM batched over hoder via grid-z (`gemm_z`, den tilede kjernen
  med per-z-strides), maskert radsoftmax med sidebuffer for
  radsummer (`softmax_rows_z`; scale er forhåndsbakt inn i q ved
  pakking så kjernen slipper float-skalarer), av-GEMM, og 1/sum
  fusjonert i den parallelle CPU-readbacken. kv padder til 64 med
  maskering (vilkårlige kryss-kontekstlengder); q-siden krever %64
  (DiT-ens 4096) ellers CPU-fallback. Wiret i dense
  `MultiHeadAttention` (self/rope/cross) via `dense_mha_from` +
  `sd.gpu`; gate `gpu_sdpa_wants` (L ≥ 2048, Lkv ≥ 512, D %64,
  scores-tak 2^28). Kontekst/scratch refaktorert til
  `gpu/context.mojo` (GpuContext gjenbrukes av linear + attention;
  re-eksport fra gpu/linear beholder gamle imports).
  Paritet: `pixi run test-wp11-attn` (I test-all → 14 testfiler),
  max|diff| ≤ 2.3e-7 mot CPU-stien (inkl. maskert odde-kv og
  MHA-dispatch). Målt (microbench_gpu_attn): self 4096 H16 D64
  207.7→57.4 ms (**3.62x**), cross 4096×1029 58.3→20.1 ms (**2.91x**),
  2048 2.87x.
- **FEMTE b2-Metal-felle (probet 2026-07-10)**: den FØRSTE komplette
  map-skriv→kjerne→map-les-syklusen i en prosess leverer korrupte les
  ved 256-byte-grenser — uavhengig av fencing og per-kjerne-warm-up
  (kjernekompilering er IKKE årsaken); eneste mitigering som holdt er
  å brenne én hel syklus. `GpuContext.__init__` kjører nå en
  offer-syklus + en VERIFISERT selvtest og kaster (→ CPU-fallback i
  gpu_context_from_env) hvis syklus to også feiler.
- **Steg 4 UTFØRT (2026-07-10): varlen/sparse sdpa via q-padding** —
  `_sdpa_core` padder nå BEGGE sider til 64: q får null-rader hvis
  utganger droppes i readbacken (komposisjonen er q-rad-uavhengig), så
  vilkårlige lengder virker. Ny inngang `gpu_varlen_sdpa_single`
  ([T,H,D]-layouten til varlen_sdpa) dekker B=1-enkeltsegmentet;
  `SparseMultiHeadAttention` (full self + cross, IKKE windowed/multi-
  segment) dispatcher via `sparse_mha_from` + `sd.gpu` når
  len(offsets)==2 og gaten holder (q %64-kravet er fjernet fra
  `gpu_sdpa_wants`). Paritet i test-wp11-attn utvidet: 7 former + begge
  MHA-dispatchene, max|diff| ≤ 2.3e-7 (grønn på første kompilering
  IGJEN — selvtest + lover). Målt: slat-self @2369 H16 67.2→23.0 ms
  (**2.92x**), slat-cross @2369×1029 30.0→14.3 ms (**2.10x**).
- **Steg 5 UTFØRT (2026-07-10): device-resident mlp-kjeding** —
  `gpu_mlp_forward` i gpu/linear.mojo kjeder lin2(gelu_tanh(lin0(x)))
  med [rows, hidden]-intermediatet PÅ GPU-en (134 MB WC-rundtur per
  ss-blokk eliminert). lin0-bias adderes på GPU før gelu (ny
  `bias_dev`-buffer i GpuLinear); gelu-tanh-kjernen regner tanh VIA
  EXP (GPU-bibliotekets tanh er en rask approksimasjon ~2e-3, exp er
  presis — softmax beviste 2.3e-7); A-scratchen gjenbrukes som
  utgangsbuffer for andre GEMM. Wiret i SparseFeedForwardNet + dense
  FeedForwardNet (gate: begge linears GPU-kvalifisert). Paritet i
  test-wp11: mlp-kjede + FFN-dispatch, max|diff| 1.6e-7. Målt:
  ss-mlp 4096×1536→8192 chain 84.2 ms vs 131.8 ukjedet-GPU vs 279 CPU
  (**3.31x**); slat-mlp @2369 3.02x. MERK for fremtidige GPU-prober:
  bruk GpuContext (verifisert selvtest) — håndrullede offer-sykluser
  er IKKE pålitelig nok (256-byte-korrupsjonen kan overleve dem;
  kostet en feilsporingsrunde der «tanh-upresisjon» viste seg å være
  én korrupt celle i proben).
- **Steg 6 UTFØRT (2026-07-10): sparse conv på GPU** —
  decode-instrumentering viste at conv dominerte (23 s ConvNeXt-convs +
  mesteparten av 17 s upsample-convs av 43 s decode; normer/mlp < 3.5 s).
  `trellis2_mojo/gpu/conv.mojo`: CPU-edge-listene (naboskapskartet
  SparseConv3d allerede cacher) counting-sorteres STABILT til CSR per
  target på host; ÉN gather-kjerne beregner hver utgangsrad ved å vandre
  radens edge-range: tråd = (rad, 8 co-laner — to vec4-akkumulatorer
  deler x-broadcasten; naborad-tråder leser koalescerte vektlinjer).
  Vekten lastes opp én gang per modell som [K, Ci, Co]
  (`GpuSparseConv.try_build` i sparse_conv3d_from via sd.gpu); x/edges
  per kall, bias i CPU-readbacken. MARSHALLING: 4 pekere = null
  skalarer → ALLE dimensjoner rir i edges-headeren (int32: n, ci, co,
  E + row_start + src + kidx). Gate: co%64==0 (try_build) og
  E·ci·co ≥ 2^31 (wants — små convs blir bit-eksakt CPU).
  Verifisert: grid-dim 70k threadgroups OK, Int32-buffere OK.
  Paritet: `pixi run test-wp11-conv` (I test-all → 15 testfiler) —
  4 former (inkl. rektangulær + dilation 2) + gates + ride-along,
  max|diff| ≤ 4.1e-5 (atol 2–5e-4). Målt (microbench_gpu_conv):
  512ch@12k 3.03x, 256@55k 3.55x, 128@216k 3.10x, up-conv1
  512→2048 2.67x (8-lane-utvidelsen ga 1.7→3.0x+).
- **Steg 7 UTFØRT (2026-07-11): device-resident attention-kjeding** —
  `gpu_attn_self_chain` i gpu/attention.mojo kjeder HELE self-attention:
  qkv-GEMM → fused bias+qk-rms+rope-kjerne (per (rad, hode)-tråd, kun
  gyldige rader) → head-major pack (pack_q_z med sdpa-skala regnet
  i-kjerne fra Int d — ingen float-skalarer; pack_kv_z transponerer k og
  NULLER paddene: av-GEMM-en multipliserer v-pad med 0 og 0·NaN ville
  forgifte utgangen) → steg 3-komposisjonen (gemm_z/softmax/gemm_z) →
  unpack_o_z (1/sum fusjonert, ALLE mp-rader skrives så out-GEMM-en
  ikke leser stale scratch) → out-GEMM. Kun x lastes opp og kun
  [T, C]-utgangen leses tilbake; per-MHA-konstantene (qkv-bias +
  begge rms-gammaene) rir i ÉN device-buffer (`GpuAttnChain`, bygget av
  dense/sparse_mha_from ETTER gamma-tilordning) så alle kjerner holder
  seg innenfor 4-bindings-loven. Scratch: linear-A holder x og deretter
  ut-GEMM-utgangen, linear-C holder qkv og deretter attention-utgangen
  (køen kjører i rekkefølge — mlp-presedens); rope-fasene lastes opp
  per kall (ny ph-scratch; dense phases kommer utenfra, sparse fra
  koordinatene via samme spatial-cache som CPU-stien). Gate:
  `chain.wants` = gpu_sdpa_wants(T, T) + qkv-linearens flops-terskel
  (out-linearen rir gratis — dens solo-break-even gjelder ikke i
  kjeden); windowed/multi-segment/cross blir på steg 3/4-stiene.
  Paritet: test-wp11-attn utvidet med 3 kjedede helhets-MHA-caser
  (dense plain, dense rms+rope, sparse rms+rope fra coords; C=1024 —
  mindre vekter går under GPU_MIN_WEIGHT) + gate-/decline-sjekker,
  max|diff| ≤ 4.8e-7 (atol 5e-4). Grønn på FØRSTE kjøring igjen.
  Målt (microbench_gpu_attn, hel MHA-forward): ss_flow-geometri
  (4096×1536, H12 D128) 171.8→90.3 ms (**1.90x** mot ukjedet GPU,
  5.98x mot CPU); slat (2369×1024, H16 D64) 66.4→28.2 ms (**2.35x**,
  4.19x mot CPU).
- **Steg 8 UTFØRT (2026-07-11): cross-attention-kjeding** —
  blokk-profilering med GPU på (`microbench_gpu_block.mojo`, ekte
  ss_flow-geometri) viste at cross-attention var blitt største post
  (92.6 ms mot self 87/mlp 81): q/out-linearene og sdpa-packen
  rundtrippet WC-minnet hver for seg. `gpu_attn_cross_chain` kjeder
  q-GEMM → `bias_rms_q` (fused bias+q-rms) → pack → sdpa-komposisjonen
  → unpack → out-GEMM device-resident; kv beregnes og k-rms-normaliseres
  på CPU-en (to_kv @~1k kontekstrader er under GPU-GEMM-break-even) og
  host-pakkes FØR enqueue-ene (map av host-skrevet buffer committer
  pending arbeid — rekkefølgen er lovpålagt). `pack_q_z` fikk
  stride-parameter (3HD for fused qkv, HD for cross-q);
  `GpuAttnChain.try_build_cross` bygger [HD q-bias][HD gamma_q]-consts
  (is_cross-flagg). Gate `wants_cross` = KUN sdpa-gaten — q/out-GEMM-ene
  rir på opplastingen sdpa-en trenger uansett, deres solo-terskler
  gjelder ikke (verifisert: slat-cross vinner 1.56x selv om q-linearen
  alene er under proxy-terskelen). Paritet: test-wp11-attn utvidet med
  3 cross-caser (dense rms, dense plain, sparse rms; + selv/cross-
  decline-sjekker), max|diff| ≤ 4.2e-7 — grønn på første kjøring.
  Målt (hel cross-MHA): ss 4096×1029 C1536 86.3→51.4 ms (**1.68x**,
  3.02x vs CPU); slat 2369×1029 C1024 44.0→28.1 ms (**1.56x**, 2.18x
  vs CPU). Blokk-totalen (ss-geometri): 289.6→244.7 ms.
- **Steg 9 UTFØRT (2026-07-11): conv-kjerne-registerblokking (rad-par)**
  — `sparse_conv_gather` kjører nå TO target-rader × 8 co-laner per
  tråd: radenes kantlister merge-vandres på kidx (stigende per rad —
  kidx-major-byggerekkefølgen overlever den stabile counting-sorten),
  så kernel-offsets som finnes i BEGGE rader laster hver vektlinje ÉN
  gang for to raders akkumulering ([ci, co]-vektplan-slicene er
  dominerende trafikk; de fleste decode-rader deler de fleste av de 27
  offsetene). Per-rad kantrekkefølge og per-akkumulator-matte er
  UENDRET → bit-identisk med enkeltrad-kjernen (paritetstall
  uforandret). Målt (microbench_gpu_conv, inkl. CSR+transfer):
  512ch@12k 216.7→152.4 ms (**1.42x**, 4.31x vs CPU), 256@55k
  230.4→146.6 ms (**1.57x**, 5.43x), 128@216k 266.6→174.3 ms
  (**1.53x**, 5.07x), up-conv1 512→2048 994.9→644.9 ms (**1.54x**,
  3.91x).
- **Steg 10 UTFØRT (2026-07-11): hel-blokk-residens** —
  `gpu/block.mojo`: `gpu_cross_block_forward` kjører HELE
  cross-blokken (dense OG sparse deler orkestratoren på flate feats
  [T, C]) device-resident: ln+modulate → self-kjede → gate_add →
  ln(affine) → cross-kjede → add → ln+modulate → mlp-kjede → gate_add,
  med ÉN x-opplasting, ÉN barrier og ÉN readback per blokk (mot seks
  transfers + ~28 ms CPU-glue). Kjedene refaktorert til enqueue-deler
  (`_attn_chain_enqueue`/`_cross_chain_enqueue`/`gpu_mlp_enqueue`) med
  device-buffer inn/ut og UTEN out-bias (host-wrapperne fuser den i
  readbacken som før; blokk-stien folder den inn i gate_add-kjernen).
  ALLE host-opplastinger (x, glue-consts, rope-faser, cross-kv) skjer
  FØR enqueue-ene — map av host-skrevet buffer committer køen (lov 2),
  så en mid-kø-opplasting ville serialisert stille. Glue-constsene
  (shift/scale/gate-par + norm2-affine + de tre out-biasene) rir i ÉN
  `bk`-buffer indeksert med Int-offset-skalar (offset-PEKERE er uprøvd
  mot marshalling-lovene — offsets-som-skalarer er gratis). Cross-kv
  fikk DEDIKERTE ckt/cvh-buffere: self-kjedens device-pack eier kt/vh i
  den fusede køen. LN-kjernen speiler LayerNorm32-formelen (biased var,
  1/sqrt(var+eps)) med eps komptime 1e-6 — dispatch-gaten sjekker
  eps==1e-6 og norm-layouten (norm1/3 uten affine, norm2 med) pluss
  alle tre kjede-gatene. Paritet: 2 hel-blokk-caser i test-wp11-attn
  (dense rms+rope + sparse rope-fra-coords), max|diff| ≤ 3.7e-6
  (atol 1e-3) — grønn på første kjøring etter to syntaksfikser.
  Målt (microbench_gpu_block, ss-geometri): 249.2→**211.4 ms**/blokk
  (**1.18x**; per-op-stien beholdt som fallback).
- **MÅLTE NEGATIVE RESULTATER (2026-07-11)** — to kø-punkter LUKKET
  uten wiring (microbench_gpu_gemm på ekte DiT-former): (1)
  GEMM-registerblokking: 64×128/4×8 ga +6 %, 128×128/8×8 +3 % —
  kjernen er ikke register-bundet. (2) bf16-lagret B-vekt (u16<<16 på
  shared-fill, EKSAKT for bf16-sjekkpunktene): FLAT innenfor støy —
  B-flisene L2-caches, så bf16/fp16-vektlagring kjøper KUN
  lastetid/minne. Kjernen ligger på ~2.9 av ~9 teoretiske TF/s
  (shared-memory-/issue-bundet); uten simdgroup_matrix i 1.0.0b2 er
  videre GEMM-tuning lav-ROI.
- **Steg 11 UTFØRT (2026-07-11): CSR-caching** — conv-CSR-sorten
  avhenger KUN av kantene, så `SparseConv3d.forward` spatial-cacher nå
  den sorterte (row_start, src, kidx)-tripletten ved siden av
  naboskapskartet (nøklet per coords + kernel + dilation); før kjørte
  counting-sorten per KALL for hver conv på samme coords.
  `GpuSparseConv.forward` tar de forhåndssorterte listene og fyller
  int32-packen chunk-parallelt. Bit-identisk (samme stabile orden);
  paritetstall uendret.
- **Steg 12 UTFØRT (2026-07-11): modellnivå-residens** —
  `gpu_cross_block_forward` splittet i primitiver
  (`gpu_block_state_upload`/`gpu_cross_block_enqueue`/
  `gpu_block_state_readback` + `gpu_block_phases`); blokkene fikk
  `_gpu_enqueue_resident` (CPU-prep: mod-chunks + kv/k-rms, deretter
  enqueue mot resident xs), og BEGGE modell-forwardene (ss_flow +
  slat) holder x device-resident over ALLE 30 blokker når hver blokk
  passerer blokk-gaten: ÉN x-opplasting + ÉN readback per forward
  (mot én per blokk), rope-fasene lastes opp én gang per forward.
  Per-blokk-uploadene (bk-consts + kv-pack) fungerer som
  inter-blokk-syncene (map committer + venter). BIT-identisk med
  per-blokk-stien — readback/upload-en som droppes var en eksakt
  kopi; verifisert i test-wp11-attn med en 2-blokks residens-driver
  mot sekvensiell kjøring: max|diff| == 0.0 EKSAKT.
- **Steg 13 UTFØRT (2026-07-11): sdpa-gulv 1024 + golden
  GPU-verifisert** — golden 12-stegs-kjøringen landet på 1857
  slat-tokens (< det gamle 2048-gulvet → hele slat-DiT-en på CPU,
  177 s). Målt: 1857 self 3.35x, 1280 2.14x, 1024 fortsatt 1.81x på
  GPU → `GPU_SDPA_MIN_Q` 2048→1024 (mater alle gatene; test-sjekkene
  flyttet til 512). Golden GPU-kjøring (12 steg + tekstur):
  **244 s = 4.1 min** mot CPU-goldenens 27.4 min = **6.74x**;
  strukturelt i praksis SAMME mesh (NN mean 2.9e-08, max 8.8e-04
  < ½ voxel; eksakt samme 1857 @32³; 0 degenerater).
- **Steg 14 UTFØRT (2026-07-11): 16-bits W^T-lagring på device** —
  «bf16-vektlagring for lastetid/minne» fra steg 11-negativene wiret:
  `GpuLinear` lagrer W^T som u16 (bf16 u16<<16 / f16 hardware-cast på
  shared-fill) når HVER vekt er bit-eksakt representerbar
  (SIMD-or-skann i try_build; DiT-ene er bf16-sjekkpunkter → bf16,
  fp16-unet-dekoderne → f16, alt annet f32 som før); alle
  vekt-GEMM-kallsteder dispatcher via `GpuLinear.enqueue_gemm`.
  Bit-eksakt ekspansjon → INGEN numerikk-variant: smoke-OBJ
  BYTE-identisk (`cmp`), alle eksisterende paritetstall uendret, nye
  test-wp11-caser (bf16/f16/blandet + bit-identitet mot f32-lagring)
  grønne på første kjøring; ekte-vekt-klassifisering verifisert
  (tests/probe_wfmt_real.mojo). Device-W^T per 1.3B-DiT ~4.8→2.4 GB;
  MAX-RSS uendret (peaken er decode-aktiveringene). e2e 71–72 s
  uendret.
- **Steg 15 UTFØRT (2026-07-11): f16-lagret sparse-conv-vekt** —
  steg 14-mønsteret på GpuSparseConv ([K, Ci, Co] som f16-bits,
  hardware-cast per vektlinje-load; delt `wfmt_scan`; bf16 bevisst
  usupportert — 4-peker-marshallingen gir ingen format-skalar, og
  ingen conv-sjekkpunkter er bf16). MERK ulikt GEMM-en: gather-kjernen
  STREAMER vektlinjene (ikke L2-cachet som B-flisene), så halvert
  vekttrafikk ga reell ytelse på vekt-tunge former: 512ch@12k 1.14x,
  up-conv1 1.13x (256/128ch flat); e2e-decode 17→14–15 s. OBJ
  BYTE-identisk, test-wp11-conv utvidet (f16/blandet + bit-identitet),
  ekte conv-vekt klassifiserer f16. GPU-køen er dermed HELT tom.
- Numerikk: GPU-akkumuleringsrekkefølge ≠ CPU → paritet mot CPU-stien
  på toleranse (som test-real), ALDRI bit-krav — bekreftet; OBJ-er
  sammenlignes strukturelt på tvers av GPU på/av (flash-presedens).

---

## Ren-Mojo-sporet (ADR 0007) — WP12–WP14

Kjøres ETTER at WP9 del 3 (hybrid runner) er ferdig; runneren er
regresjonsharness for hver utbytting. Sluttmål: Python/torch kun i
`tests/parity/` og `benchmarks/`.

### WP12 — safetensors + config-JSON i ren Mojo — ✅ FERDIG 2026-07-09
**Mål:** `ckpt_io.py` ut av runner-stien. Ny `trellis2_mojo/io/`:
- safetensors-leser: 8 bytes LE headerlengde + flat JSON-header
  (`{navn: {dtype, shape, data_offsets}}`) + rå bytes. `open().read_bytes()`
  er verifisert i 1.0.0b2. bf16→f32 = `u16 << 16`-bitshift; f16 via
  `DType.float16`-cast. Gjenbruk SIMD-kopimønsteret fra interop.mojo.
- mini-JSON-parser (objekter/lister/strenger/tall/bool — nok for
  safetensors-headere, ckpt-configene og pipeline.json).
- `checkpoints.mojo` byttes til Mojo-leseren.

**Ferdig når:** state_dicts er bit-identiske med ckpt_io.py på alle 8
sjekkpunktene (egen paritetstest) og `pixi run test-real` er grønn med
Mojo-leseren i lastestien. Anslag: 1–2 økter.

### WP13 — DINOv3 ViT-L/16 i Mojo — ✅ FERDIG 2026-07-09
**Mål:** `models/dinov3.mojo` erstatter interop-kondisjoneringen fra WP9
del 3 steg 3. Arkitektur (fra HF-config + `DinoV3FeatureExtractor`):
24 lag × {LN → MHA → LayerScale → res; LN → MLP(gelu, 4096, IKKE gated) →
LayerScale → res}, 1024 ch / 16 hoder, patch-conv 16×16, cls + 4
register-tokens, 2D-RoPE theta=100 med `pos_embed_rescale=2.0`
(transformers-varianten — speil den, ikke vår modell-rope), q/v-bias men
IKKE k-bias (vår fusede qkv må splittes eller nullpadde k-bias),
slutt-`F.layer_norm` UTEN affine. Returnerer ALLE tokens
[B, 1+4+N, 1024]. Vekter (1.1 GB safetensors, HF-navngiving → egen
loader-mapping) via WP12-leseren. Merk cascade-stien: cond trengs på både
512 (1029 tokens) og 1024 (4101 tokens); '512'-pipeline trenger bare 512.

**Ferdig når:** paritet mot transformers-modellen på seedede bildetensorer
(atol-trapp som WP8) + runner-output uendret (strukturelt) med Mojo-cond.
Anslag: 2–4 økter — største enkeltjobb i sporet.

**Utfall (2026-07-09):** `trellis2_mojo/models/dinov3.mojo` +
`checkpoints.mojo::load_dinov3()`; `ImageConditioner` kjører ren-Mojo-ViT
(cond_io.py er kun preprocess + pikseltensor, transformers ute av
runner-stien). Avvik fra planteksten: pos_embed_rescale er KUN
treningsaugmentering (transformers gater på self.training) — eval hopper
over den; k-bias løses med separate q/k/v-projeksjoner (ingen fusjonering,
så ingen nullpadding trengs); rotate_half-splitt (NeoX-stil), ikke parvis
interleave. Paritet: `pixi run test-wp13` (i test-all, random liten
config, ~1.7e-6, inkl. ikke-kvadratisk grid) og `pixi run test-cond`
(ekte ViT-L-vekter, 3.4e-5 på 128/512, atol 5e-4). Gikk på 1 økt.

### WP14 — bilde-IO + preprocess i ren Mojo — ✅ FERDIG 2026-07-09
**Mål:** siste Python-biten ut av runneren.
- PAM-leser (P7, har alfa; PPM P6 for RGB) + dokumentert
  konverteringskommando (`sips`/ImageMagick) i README_MOJO.
- Port av `preprocess_image`: alfa-bbox (>0.8·255), kvadratisk crop,
  alfa-premultiply, /255, ImageNet-normalisering.
- Lanczos-resize med PIL-semantikk (a=3, separabel) — paritet mot
  PIL/numpy-referanse; brukes både til 1024-nedskalering og 512/1024-
  kondisjoneringsstørrelsene.
- Valgfritt (eget beslutningspunkt, se ADR 0007): ren-Mojo PNG-dekoder
  (zlib inflate + filtre, kun 8-bit ikke-interlaced RGBA).

**Ferdig når:** runner tar PAM/PPM (evt. PNG) uten Python i stien, og
preprocess-paritetstest mot PIL-referansen er grønn. Anslag: 1–2 økter.

**Utfall (2026-07-09):** `io/image.mojo` (PAM P7/PPM P6) +
`imaging/{resize,preprocess}.mojo`. ALT er BIT-eksakt mot PIL
(`pixi run test-wp14`, i test-all): Pillows fixed-point-resampling er
speilet i heltall (flyttall kan ikke matche pga. u8-kvantisering mellom
passene), inkl. RGBa-premultiply-rundturen PIL gjør for RGBA-resize
(MULDIV255 inn, trunkerende divisjon ut — funnet empirisk) og
copy()-kortslutningen ved uendret størrelse. Crop bruker PILs
round-half-even. PNG-dekoder bevisst ugjort — PNG→PAM-énlinjer
dokumentert i README_MOJO (omskrevet med komplett kjøreeksempel).
test-cond-driften UENDRET etter byttet; wp14-smoke ga byte-identisk OBJ
med wp13-smoken. Gikk på 1 økt. REN-MOJO-SPORET (WP12–WP14) ER I MÅL.

---

## Fase 2 — teksturert eksport (ADR 0008, startet 2026-07-11)

### WP15 — GLB med vertex-attributter — ✅ FERDIG 2026-07-11
**Mål:** runneren leverer en teksturert, direkte viewbar GLB i ren Mojo —
uten den CUDA-bundne UV-baking-stien (cumesh/nvdiffrast/cv2/xatlas, se
ADR 0008 for hvorfor vertex-attributter er ~tapsfrie uten decimering).

1. `meshing/vertex_attrs.mojo`: `grid_sample_trilinear` med EKSAKT
   flex_gemm-semantikk (trunkerte p±0.5-naboer, vekt prod(1−|nabo+0.5−p|),
   manglende voxels vekt 0, renormalisering clamp_min 1e-12; sparse oppslag
   via pakket-nøkkel-Dict som i fdg_mesh) + arealvektede vertex-normaler.
2. `io/glb.mojo`: GLB 2.0-container (JSON-chunk + BIN-chunk) med
   POSITION/NORMAL/COLOR_0/indices + pbrMetallicRoughness (globale
   metallic/roughness-faktorer = gjennomsnitt av samplede verdier);
   oppstrøms akse-swap (y,z → z,−y).
3. Runner: etter `decode_tex` — sample [T,6]-volumet på verteksene,
   skriv `prefix.glb` (OBJ + npz uendret).

**Ferdig når:** paritetstest mot ren-torch-reimplementasjon av
flex_gemm-formelen grønn (`test-wp15`, i test-all), GLB-roundtrip
bit-eksakt, golden-kjøring med tekstur gir GLB der COLOR_0 matcher
referanse-sampling av npz-en på samme vertekser.

**UTFØRT (2026-07-11, én økt):** test-wp15 grønn på første kjøring
(trilinear ≤ 6e-8 mot ren-torch-fasit, tomt volum eksakt 0, normaler
≤ 2.5e-7, GLB-roundtrip BIT-identisk via avhengighetsfri Python-leser —
trimesh er ikke i pixi-env). Golden (12 steg + tekstur): OBJ/npz
BYTE-identiske med golden-artefaktene, GLB-steget < 1 s @ 514k
vertekser, COLOR_0 matcher uavhengig numpy-referanse på 2.4e-7,
materialfaktorer 1e-11, trimesh (trellis-mac-venv) laster rent
(514 604 V / 1 055 568 F, 33 MB). test-all = 16 filer.

### WP16 — mikrohull-fylling i GLB-eksporten — ✅ FERDIG 2026-07-11 (REVIDERT)
**Mål:** fylle FDG-ens mikrohull (brukersynlige; finnes også i
trellis-mac) slik oppstrøms gjør i cumesh-postprosessen
(`fill_holes(max_hole_perimeter=3e-2)`); kun GLB (OBJ/npz rå som
oppstrøms).

**UTFØRT (v1, sykel-vandring — ERSTATTET):** kjedet randkanter til rene
sykler (åttetalls-splitt + full blindvei-revert); fylte 524/~526 slike,
men brukeren så fortsatt hull: alle FLETTEDE klynger ved
non-manifold-kryss forble åpne.

**UTFØRT (v2, cumesh-formuleringen):** CuMesh-kilden er publisert
(github.com/JeffreyXiang/CuMesh) — dens fill_holes er
KOMPONENT-basert, ikke sykel-basert: union-find-komponenter av
randkanter, avvis kun komponenter med grad-1-vertekser (blindveier;
kryss er lov), fyll hele komponenten med ÉN centroid (snitt av
kant-midtpunkter) + (b, a, c)-trekant per randkant.
`meshing/postprocess.mojo` omskrevet; `test-wp16` oppdatert
(delt-verteks-hull = ÉN komponent). LÆRDOM: les den faktiske kilden
FØR man designer en «intensjonsport» — antakelsen «hull = ren sykel»
var feil og kostet en hel revisjonsrunde.

**UTFØRT (v3, non-manifold-reparasjon):** brukeren så fortsatt hull —
100 % av blindvei-verteksene satt på ikke-mangfoldige kanter: ringer
med én grad-3-kant leses som åpne stier (22 087 komponenter på
1024-goldenen). Portert `repair_non_manifold_edges` fra cumesh
(hjørne-basert splitting: union-find over 3F hjørner, merge kun over
mangfoldige kanter → vifter splittes i mangfoldige ark, der all rand
er lukkede sløyfer); runner kjører fill → repair → fill som
oppstrøms. GitHub-issue #105 bekrefter at hull består selv oppstrøms
uten remeshing — decimering/remesh-stien er fortsatt bevisst utenfor
(ADR 0008).

**UTFØRT (v4, orienterings-unify):** rotårsaken til det SYNLIGE
symptomet: FDG-vindingen er ~50/50 myntkast (995 684 samme-retning-
kantpar; culled baksider + kansellerte vertex-normaler ser ut som
mikrohull). Portert cumesh `unify_face_orientations`
(paritets-union-find, flipp odde paritet): 995 684 → 6 845 (99.3 %).
En GLOBAL per-ark inne/ute-stemme ble prøvd i to varianter
(okkupans-probe; flood-fill-orakel) og FJERNET etter måling — begge ga
51 %, fordi FDG-flaten er stedvis foldet (begge fortegn i samme ark =
nullsum). Visning garanteres av doubleSided=true. Sekvens:
fill → repair → fill → unify. Endelige artefakter:
`outputs/shoe_{512,1024}_final.*`.

**UTFØRT (v5, null-normaler i fyll-centroids, 2026-07-12):** Khronos-
gltf-validatoren (brukerens rapport) fant `ACCESSOR_VECTOR3_NON_UNIT`:
flettede/foldede fyll-vifter kansellerer arealvekt-summen i
centroid-verteksene (65 @512 / 379 @1024) — null-normaler skygges som
svarte prikker. `vertex_normals` garanterer nå enhetslengde
(nabo-snitt av enhetsnormaler → deterministisk +z-fallback);
kansellert-vifte-case i test-wp15.

**UTFØRT (v6, småfragment-fjerning, 2026-07-12):**
`remove_small_connected_components(1e-5)` portert fra cumesh-kilden —
som viste seg å ligge LOKALT i `trellis-mac/deps/mtlmesh/src/
{clean_up,connectivity}.cu` (mtlmesh = Metal-porten av CuMesh; ingen
GitHub-henting nødvendig): flate-komponenter unioneres KUN over
mangfoldige kanter (delt verteks eller non-manifold kant kobler ikke),
komponentareal = sum av 0.5·|cross|, fjern < 1e-5, kompakter
ureferererte vertekser i original rekkefølge. Runner-sekvens per
oppstrøms to_glb (minus de CUDA-bundne simplify-stegene):
fill → repair → **remove_small** → fill → unify. Golden: 13 897 ark /
60 472 flater fjernet @512 (GLB 648k → 552k V), 34 597 / 167 997 @1024
(→ 2.13M V); OBJ/npz byte-identiske (kun GLB-stien påvirkes);
GLB-sjekkene grønne; 0 komponenter < 1e-5 igjen (uavhengig
numpy-union-find). Diagnostisk A/B mot trellis-mac-meshen (samme seed,
`tests/checks/ab_mac_vs_mojo.py`): oppstrøms rå-mesh har SAMME
strukturklasse (11 040 fragment-komponenter < 1e-5, 42k non-manifold-
kanter, 15k blindveier mot våre 13 909/51k/17k) — prikk-patologien er
modellens natur, ikke porteringsartefakt (jf. issue #105).

**UTFØRT (v7, sprekk-sying, 2026-07-12 — EGEN semantikk, brukerens
valg):** visuell A/B avgjorde at oppstrøms teksturerte GLB har samme
prikker (modellens natur); brukeren valgte sying fremfor
remesh-grenen. `sew_boundary_seams`: sveiser rand-vertekser med
BIT-identiske posisjoner (ingen epsilon; spatial hash + eksakt
verifisering; først-sett representant = deterministisk) — de
gjenværende sømringene er null-bredde ark-grenser med eksakte
duplikater fra hjørnesplitten/foldene. Sekvens: fill → repair →
remove_small → fill → **sew → fill** → unify. Golden: 24 798/30 230
vertekser sveiset @512/@1024, randkanter i ferdig GLB −72 %/−73 %
(59 794→16 456 / 72 243→19 543); OBJ/npz byte-identiske; sjekker
grønne. Kjente rester: non-manifold-kanter gjenskapt ved sveisekryss
(unify hopper over; IKKE kjør repair etter sew), og 145
sub-terskel-sløyfer @512 i blandede komponenter (loop-nivå-fylling er
neste kandidat hvis prikker består).

### WP18 — remesh-grenen: narrow-band dual contouring — ✅ FERDIG 2026-07-12 (@512)
**Mål:** oppstrøms' «egentlige svar» på prikk-/sprekk-klassen
(brukerens valg: aggressivt først, finjuster etterpå; kun 512):
`cumesh.remeshing.remesh_narrow_band_dc` — ekstraher offset-flaten
UDF − eps = 0 rundt FDG-meshen med dual contouring, projiser 90 %
tilbake. Sprekker < ~2·eps svelges per konstruksjon; lukket flate med
konsistent vinding.

**UTFØRT:** `meshing/remesh.mojo` (triangel-grid CSR i stedet for
cuBVH, direkte AABB-stempling i stedet for oktre — samme voxelsett,
eksakt UDF-filter; DC-kjernen eksakt port; INTENDERT split-align —
NIENDE oppstrøms-bug funnet og dokumentert: deres align-indeksering
sammenligner trekant 1 med seg selv, split 1 velges alltid).
Runner-flagg `--remesh` erstatter hele cleanup-kjeden i GLB-stien.
test-wp18 i test-all (→ 18 filer): kube → vanntett dobbelt-skall m/
positivt fortegnsvolum, PUNKTERT kube → fortsatt vanntett,
offset-avstand ~eps, projeksjon < 0.16·eps, determinisme. Golden
@512 (`outputs/shoe_512_remesh.glb`, sying-finalen beholdt for A/B):
975 708 V / 1 952 036 F, **0 randkanter**, 305 non-manifold DC-kryss
(tvetydig-celle-artefakt, usynlig), COLOR_0 2.4e-7, 256 s totalt
(remesh ~10 s); OBJ/npz byte-identiske. Finjusterings-skruer: band,
project_back, DC-oppløsning, decimering (fortsatt utenfor scope).

### WP17 — 1024-kaskaden + sdpa-hodegruppering — ✅ FERDIG 2026-07-11
**Mål:** kvalitetsspaken fra WP15/16-oppsummeringen: kjør oppstrøms
`1024_cascade` (DERES default) i runneren — dobbel lineær oppløsning.
Alt lå klart siden WP9 (cascade_coords/upsample_coords paritetstestet,
1024-sjekkpunktene i cachen, cond tar res-param).

**UTFØRT:**
- Runner: `--pipeline 1024` — cond@512+1024 (4101 tokens), ss@32³
  uendret, LR-slat (512-DiT) → `run_cascade_stage` (shape-dekoderens
  subdivisjons-upsample 4 nivåer → kvantiser til 64³, dedupe;
  token-budsjett-løkka er no-op for 1024-målet) → HR-slat (1024-DiT,
  cond@1024) → tex (1024-DiT) → decode/mesh/npz/GLB @1024³.
  Stage-funksjonene tar nå modellnøkkel som parameter.
- **SJETTE b2-Metal-felle (probet)**: kjerneskriv forbi 4 GiB
  byte-offset i ÉN buffer-binding tapes STILLE (alloc lykkes, les under
  grensen er fine) — full-H scores for HR-slat (T~12k → 9 GB) er
  umulig som én binding. Dokumentert i gpu/linear.mojo-lovlisten.
- **Løsning: hodegruppert sdpa** (`_enqueue_sdpa_groups` i
  gpu/attention.mojo): qk→softmax→av enqueues i hodegrupper mot ÉN
  scores-scratch ≤ GPU_SDPA_MAX_SCORES (uendret 2^28); gaten ble
  per-HODE (mp·lp ≤ 2^28). Køen er in-order → gjenbruk på tvers av
  grupper er trygt; én gruppe (hg == h) er identisk med gammel sti —
  ALLE eksisterende paritetstall uendret. Ny test-case: self 4160²
  H16 (15+1 grupper, tidligere AVVIST form) ≤ 2e-7.
- Smoke @1024 (steps 2, --no-tex): HR-slat-sampling **698 s (CPU) →
  120 s** (5.8x); totalt 836→256 s; 3.26M voxels @1024³ → 6.44M
  triangler; 273 borderline-flips av 3.26M (0.008 %) mot
  CPU-attention-kjøringen — kjent toleranseklasse. Peak RSS 18.4 GB
  (48 GB-maskin).
- Golden @1024 (12 steg + tekstur): **836 s = 13.9 min** (512: 244 s);
  1857 @32³ → 7545 tokens @64³ → **2 058 563 voxels @1024³ = 4.00x**
  512-goldenen → 2.06M V / 4.16M F; 2080 mikrohull fylt; GLB-sjekk
  grønn (COLOR_0 3e-7 mot sparse numpy-referanse); peak RSS 16.6 GB.

---

## 3. Arbeidsflyt per fil (definisjonen av «å portere»)

1. Les mapping-doc for filen (`docs/mapping/...`) — **skriv den hvis den mangler**
   (mange i indeksen eksisterer ikke ennå; se tracker).
2. Sett status `🔄 pågår` i [07_PORT_TRACKER.md](07_PORT_TRACKER.md).
3. Port til `trellis2_mojo/` med samme relative sti.
4. Skriv/utvid paritetstest i `tests/parity/`.
5. Grønn test → status `✅ portert`. Avvik utenfor toleranse → noter i tracker, ikke gå videre.
6. Oppdater `MOJO_STATUS.md` ved fullført WP.

**Definisjon av ferdig (per WP):** alle filer `✅`, paritetstester grønne i CI/lokalt,
mapping-docs oppdatert til å reflektere faktisk implementasjon.

## 4. Topprisikoer (fra 04, konkretisert)

| Risiko | Trigger | Mottrekk |
|---|---|---|
| SparseTensor-design skalerer ikke til attention-behovene | WP5 trenger layout-info WP3 ikke eksponerer | Les WP5/WP6-kildene **før** WP3-design fryses |
| Interop-overhead spiser gevinsten | Per-kall-kost Python↔Mojo > kernel-tid | Mål i WP1-smoke-test; batch kall på blokknivå, ikke op-nivå |
| fp16-avvik akkumulerer over 25 sampler-steg | Paritet per-op grønn, E2E rød | WP0-b golden per-steg-dumps gjør bisect mulig |
| Mojo-API-endringer (språket er ungt) | Byggbrudd etter oppgradering | Versjonslås i pixi.toml; oppgrader kun bevisst |
