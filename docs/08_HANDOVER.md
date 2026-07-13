# 08 — Handover: TRELLIS.2 → Mojo-port

Skrevet 2026-07-07, oppdatert 2026-07-12 (WP11 steg 2–15 — GPU-spor: golden 27.4 min → 4.1 min = 6.74x; 16-bits vektlagring bit-eksakt. FASE 2/WP15–17: teksturert GLB via vertex-attributter + mesh-postprosess i 7 revisjoner (endelig: fill → non-manifold-repair → remove_small(1e-5) → fill → sprekk-SYING → fill → paritets-unify + unit-normaler) + 1024-KASKADEN m/hodegruppert sdpa, ADR 0008. PRIKK-SAKEN: A/B avgjort — oppstrøms teksturerte GLB har SAMME prikker (modellens natur); v7-sying sveiset 24 798/30 230 vertekser (randkanter −72/−73 %), og WP18-REMESHEN (narrow-band dual contouring, `--remesh`, kun 512 foreløpig) gir GARANTERT 0 randkanter — to GLB-kandidater venter brukerens visuelle valg, se ÅPEN SAK-seksjonen).
Alt under her er verifisert kjørende på denne maskinen (macOS arm64,
M4 Pro — Metal-GPU-en brukes nå av WP11-stien bak env-flagg). Dette
dokumentet er nok til å plukke opp arbeidet uten annen kontekst — les det
sammen med [06_MASTER_PLAN.md](06_MASTER_PLAN.md) (arbeidspakker) og
[07_PORT_TRACKER.md](07_PORT_TRACKER.md) (fil-for-fil-status).

## Tilstand: WP1–WP10 + WP9 del 3 + WP12–WP15 ferdig — REN-MOJO-SPORET ER I MÅL, FASE 2 STARTET

Hele ADR 0007-målet er nådd: pipeline-kjøringen er ren Mojo fra PAM-fil til
OBJ/npz — eneste Python i runneren er `torch.randn` (bevisst beholdt for
støy-strøm-kompatibilitet med originalen). Python/torch ellers kun i
tests/parity og benchmarks. WP11 steg 2–15 er utført (2026-07-10/11):
den tilede Metal-GEMM-en er WIRET BAK `linear`, SDPA kjører som
GEMM-komposisjon på GPU-en for BÅDE dense MHA (ss_flow) og
enkeltsegment sparse MHA (B=1-slat, q-padding), FFN-ene kjeder
lin→gelu→lin med intermediatet device-resident, sparse conv har en
CSR-gather-kjerne med rad-par-registerblokking (steg 6+9,
decode-stadiet), HELE self-attention kjeder qkv→bias/rms/rope→sdpa→out
device-resident (steg 7), cross-attention kjeder q-siden
device-resident med host-pakket kv (steg 8), og HELE cross-blokken
(begge DiT-ene) kjører device-resident med glue som GPU-kjerner og én
transfer-rundtur per blokk (steg 10), CSR-en spatial-caches (steg 11)
og BEGGE DiT-forwardene holder x device-resident over alle 30 blokker
(steg 12) — alt bak env-flagget `TRELLIS2_GPU=1` (CPU-fallback som
default) med paritetstester i suiten
(`test-wp11`/`test-wp11-attn`/`test-wp11-conv`) — se WP11-seksjonene
under og i 06_MASTER_PLAN. GEMM-tuning og fp16-som-ytelse er MÅLT DØDE
og lukket (steg 11-negativene); windowed-batching STRØKET (slat-DiT-en
er attn_mode='full' overalt). Steg 14+15 (2026-07-11) lagrer vektene i
16 bits på device når det er bit-eksakt (bf16 for DiT-ene, f16 for
unet-dekoderne — GEMM-W^T OG conv-vekten) — halverer
device-vektavtrykket, bit-identisk output; conv-gatheren fikk attpåtil
1.13–1.14x på de vekt-tunge formene (decode 17→14–15 s).
FASE 2 (ADR 0008): WP15 (teksturert GLB via vertex-attributter),
WP16 (mesh-postprosess i 7 revisjoner — fyll/repair/fragment-fjerning/
sprekk-sying/unify/unit-normaler, se egen seksjon) og WP17
(**1024-kaskaden**: `--pipeline 1024` gir 4.00x flere voxels på
13.9 min; sjette Metal-felle probet → hodegruppert sdpa) er FERDIGE —
se egne seksjoner. Gjenstående spor: WP0 golden outputs (krever
CUDA-maskin); UV-baking-stien (cumesh/nvdiffrast/xatlas) forblir
utenfor scope per ADR 0008 (kun meningsfull sammen med decimering).
Ytelsespass 7+8 (2026-07-09) tok
e2e-smoken (steps 2) fra 627 til 252 s; WP11-flagget tar den videre
til 72 s = 3.50x, og den FULLE kvalitetskjøringen (12 steg + tekstur)
er GOLDEN GPU-VERIFISERT: 27.4 min → **4.1 min = 6.74x**, geometrisk
i praksis samme mesh som CPU-goldenen (se WP11-seksjonene).

**Alt under modellnivået, alle inferens-modellene OG hele den modellvendte
pipeline-kjeden (shape → cascade → tekstur) er portert til Mojo og
paritetsverifisert mot torch-originalen** (`SparseStructureFlowModel` dense
DiT, `SparseStructureDecoder` SS-VAE-dekoder, `SLatFlowModel` sparse DiT,
`SparseUnetVaeDecoder`/FDG-hodet for sc_vaes, og
`trellis2_mojo/pipelines/image_to_3d.mojo`). Hver ported komponent
sammenlignes op-for-op mot original PyTorch-kode med seedede tilfeldige
data og ekte vekter.

```
pixi run test-all        # hele suiten (18 testfiler, alle grønne per handover)
pixi run mojo --version  # Mojo 1.0.0b2 (låst i pixi.toml)
```

Enkelt-tester: `test-sparse`, `test-parity`, `test-flow-euler`, `test-wp4`,
`test-wp5`, `test-wp6`, `test-wp7`, `test-wp8`, `test-wp9`, `test-mesh`,
`test-wp13`, `test-wp14`, `test-wp15`, `test-wp16`, `test-wp18`,
`test-wp11`, `test-wp11-attn`, `test-wp11-conv`
(se pixi.toml; test-mesh/wp13–wp16/wp18/wp11* er OGSÅ i test-all — raske
og cache-uavhengige; wp11-testene krever Metal-GPU-en og SKIP-er høylytt
uten).
Benchmarks: `pixi run bench` (WP10, se `docs/benchmarks/RESULTS.md`).
Ende-til-ende: `pixi run e2e -- bilde.pam [--seed N] [--steps N] [--out
prefix] [--no-tex] [--pipeline 512|1024]` (WP9 del 3 steg 5; PAM P7 RGBA
siden WP14 — PNG→PAM-énlinjer i README_MOJO — + HF-cachen). Med tekstur
skrives .obj + _texvoxels.npz + **.glb** (WP15 — teksturert, direkte
viewbar); `--pipeline 1024` kjører oppstrøms 1024_cascade (WP17 —
~4x flere voxels).
Med `--remesh` (WP18) byttes GLB-postprosessen mot narrow-band dual
contouring (garantert tett flate — se egen seksjon).
Ekte sjekkpunkter: `pixi run test-real` — laster ALLE pipeline-modellene fra
den lokale HF-cachen (safetensors bf16/fp16 → f32, ren Mojo siden WP12) og
sammenligner fulle forwards mot torch-originalen; bevisst IKKE i test-all
(leser ~10 GB per kjøring, ~5 min). Kondisjonering: `pixi run test-cond` —
ren-Mojo DINOv3 (WP13) med ekte ViT-L-vekter mot originalens ekstraktor
(laster 2×300M fra HF-cachen; heller ikke i test-all; liten-config-paritet
ligger i test-wp13 som ER i test-all). Lasting: `pixi run test-io` — ren-Mojo
safetensors/JSON/HF-cache mot ckpt_io.py, bit-identisk på alle 8
sjekkpunktene (~14 GB lest; heller ikke i test-all).

### Portert (trellis2_mojo/)

| Modul | Innhold |
|---|---|
| `sparse/tensor.mojo` | Minimal CPU-tensor (flat data + shape), IntMatrix (coords), Frac (eksakt skala), stable_argsort |
| `sparse/basic.mojo` | VarLenTensor + SparseTensor (offsets i stedet for slices, ArcPointer-delt skala-nøklet spatial-cache, komposisjon i stedet for arv) |
| `sparse/attention/` | full_attn (én blokk-diagonal varlen-SDPA-kjerne for alt), windowed_attn, rope, modules (SparseMultiHeadAttention + RMSNorm) |
| `sparse/spatial/` | Downsample/Upsample, Spatial2Channel/Channel2Spatial |
| `sparse/conv.mojo` | SparseConv3d (conv_none-backend, submanifold) |
| `sparse/transformer/` | blocks + modulated (adaLN/share_mod) |
| `modules/nn.mojo` | SparseLinear/linear, LayerNorm32/GroupNorm32/ChannelLayerNorm32, relu/silu/gelu/gelu-tanh, modulate |
| `modules/attention.mojo` | dense MultiHeadAttention (self/cross, rope-overload) |
| `modules/rope.mojo` | dense RotaryPositionEmbedder + apply_rotary_embedding [N,L,H,D] |
| `modules/transformer/` | dense blocks + modulated (cross-blokk har rope-overload), AbsolutePositionEmbedder |
| `models/sparse_structure_flow.mojo` | **WP8.1**: TimestepEmbedder + SparseStructureFlowModel + `sparse_structure_flow_from`-loader |
| `modules/spatial.mojo` | pixel_shuffle_3d (patchify/unpatchify er død kode oppstrøms) |
| `modules/conv.mojo` | dense Conv3d (kubiske kjerner, uniform stride/padding). Ytelsespass 7 (2026-07-09): SIMD over zd-laner + OU=4-utkanalblokk + parallelize, bit-identisk — var SS-VAE-dekoderens flaskehals (0.86 GF/s naiv → 72–150) |
| `models/sparse_structure_vae.mojo` | **WP8.2**: SparseStructureDecoder + ResBlock3d/UpsampleBlock3d/norm_layer + `sparse_structure_decoder_from`-loader (encoder = trening, ikke portert) |
| `models/structured_latent_flow.mojo` | **WP8.3**: SLatFlowModel (sparse DiT) + concat_cond + `slat_flow_from`-loader (liste-cond/VarLen-kontekst og elastic mixin ikke portert) |
| `models/sc_vaes/sparse_unet_vae.mojo` | **WP8.4**: SparseConvNeXtBlock3d + SparseResBlockC2S3d + SparseUnetVaeDecoder (pred_subdiv/guided/upsample_coords) + loadere |
| `models/sc_vaes/fdg_vae.mojo` | **WP8.4**: `fdg_head` (FlexiDualGrid-transformene) |
| `meshing/fdg_mesh.mojo` | **WP9 del 3 steg 4**: `flexible_dual_grid_to_mesh` i REN Mojo (port av trellis-mac-stubben, «identical output to the CUDA version for inference») — Dict-basert koordinatoppslag (pakket 21-bit-nøkkel), kant-nabo-tabell, quad-bygging i (i, axis)-rekkefølge, begge split-moduser (split_weight = pipeline-stien, normal-justering for kompletthet) + `write_obj`. MERK: 0 quads → 0 vertices (stubbens early-out) |
| `pipelines/image_to_3d.mojo` | **WP9 del 1+2**: SSFlowVelocity/SlatFlowVelocity-adaptere, sample_sparse_structure, occupancy_to_coords, sample_slat, decode_shape, cascade_coords, normalize_slat, decode_tex |
| `../run_image_to_3d.mojo` (repo-rot) | **WP9 del 3 steg 5**: ende-til-ende-runner, Mojo-vert (`pixi run e2e -- bilde.pam [--seed N] [--steps N] [--out prefix] [--no-tex]`, PAM P7 siden WP14), 512-pipelinen: cond → ss → shape-slat → tex-slat → decode → ren-Mojo-mesh → OBJ + tex-voxels-npz. Params/normalisering fra pipeline.json; én modell lastet om gangen (stage-funksjoner → ASAP-frigjøring); støy trekkes fra torch-strømmen i oppstrøms rekkefølge (seed ETTER cond — se RNG-notatet i filheaderen) |
| `samplers/` | FlowEuler + CFG + guidance-interval (VelocityModel-trait), PyVelocityModel (Mojo-løkke driver torch-modell) |
| `interop.mojo` | tensor/intmatrix ↔ torch-broer — omskrevet til peker-kopi (`data_ptr()` → `UnsafePointer(unsafe_from_address=...)`, SIMD-løkke) fordi element-vis PythonObject-konvertering var uholdbar for 1.3B-vekter; bit-identiske verdier, hele suiten mye raskere |
| `checkpoints.mojo` | **WP9 del 3 steg 2, REN MOJO siden WP12**: bygger alle pipeline-modellene fra de EKTE sjekkpunktene via `io/`-laget — ingen Python i lastestien. `load_dinov3()` (WP13) leser config+vekter fra facebook-repoets snapshot. `ckpt_io.py` finnes fortsatt, men KUN som torch-referanse for paritetstestene (test-io/test-real) |
| `io/json.mojo` | **WP12**: mini-JSON-parser (arena/indeks-basert JsonDoc, ingen rekursiv verditype) — objekter/lister/strenger (escapes inkl. \uXXXX + surrogatpar), tall (Int-mantisse + eksakt 10-potens → samme f64 som strtod for ≤18 sifre), bool/null. Nok for safetensors-headere, ckpt-configer og pipeline.json |
| `io/safetensors.mojo` | **WP12**: ren-Mojo safetensors→f32-leser. Per-tensor sekvensiell lesing i offset-rekkefølge (macOS capper read() på 2 GiB — hele filen kan IKKE slurpes; gir også lavere peak: dict + én rå-chunk). bf16 = u16<<16 via `SIMD(from_bits=)`, f16 = hardware-cast, alignment=1-laster (data-seksjonen er ualignert). BIT-identisk med ckpt_io.py på alle 8 sjekkpunktene |
| `io/hf_cache.mojo` | **WP12**: HF-cache-oppslag i ren Mojo (getenv/listdir/sort — samme snapshot-valg som ckpt_io's glob), ckpt_base med kryss-repo-formen, load_config_json/pipeline_config_json/model_path |
| `io/state_dict.mojo` | **WP12**: `StateDict`-fasade — ALLE loadere tar denne. `@implicit`-ctor fra PythonObject (paritetstestene sender torch-dicts uendret) eller Mojo-Dict fra safetensors-leseren (runner-stien) |
| `models/dinov3.mojo` | **WP13**: DINOv3 ViT-L/16 i REN Mojo, speiler transformers-implementasjonen slik ekstraktoren driver den — patch-conv 16×16 som im2col+`linear`, [cls; 4 reg; patches], 2D-RoPE theta=100 (inv_freq over head_dim/4, vinkelrad [y-del, x-del] + tile(2), rotate_half-splitt — IKKE parvis interleave som modell-ropene; kun patch-tokens roteres; pos_embed_rescale er trenings-augment og hoppes over i eval), separate q/k/v (k UTEN bias), LayerScale, eksakt-erf gelu-MLP (ikke-gated), slutt-LN uten affine (modellens `norm`-vekter brukes aldri av ekstraktoren og leses ikke). `dinov3_from`-loader med HF-nøkkelnavn; gjenbruker `linear`/`LayerNorm32`/`dense_sdpa_q_k_v` |
| `pipelines/conditioning.mojo` | **WP9 del 3 steg 3, HELT ren Mojo siden WP13+WP14**: `ImageConditioner.get_cond(path, res)` → [1, L, 1024]-cond (+ `zeros_like_cond` for neg_cond) — PAM-dekoding → preprocess → Lanczos/normalisering → Mojo-ViT, ingen Python. rembg AVVIST per ADR 0007 (RGBA-krav). `cond_io.py` er KUN PIL/torch-referanse for paritetstestene (samme rolle som ckpt_io.py) |
| `io/image.mojo` | **WP14**: PAM P7- og PPM P6-lesere (8-bit, DEPTH 3/4), `read_image`-dispatch på magic bytes. PNG er et dokumentert konverteringssteg (README_MOJO) — ren-Mojo PNG-dekoder forble bevisst ugjort (eget beslutningspunkt per ADR 0007) |
| `imaging/resize.mojo` | **WP14**: PIL-EKSAKT Lanczos (a=3): Pillows fixed-point-numerikk (PRECISION_BITS=22, koeffisienter kvantisert med round-half-away, clip8, horisontal→vertikal med u8-kvantisering MELLOM passene) + RGBa-rundturen PIL gjør for RGBA (MULDIV255-premultiply → resample → trunkerende dedivisjon, alpha 0/255 passthrough; funnet empirisk — uten den bommer ~70 % av pikslene) + copy()-kortslutning ved uendret størrelse FØR rundturen. BIT-identisk med PIL på alle testcaser |
| `imaging/preprocess.mojo` | **WP14**: alpha-grenen av preprocess_image + `cond_pixels` — >1024-nedskalering (int-trunkert målstørrelse), alpha-bbox (>204, dvs. 0.8·255-f64-semantikken), kvadratisk crop med PIL-avrunding (round-half-even!) og nullpadding utenfor bildet, premultiply i f32 med u8-trunkering (numpy astype), ImageNet-normalisering → [1,3,R,R]. BIT-identisk mot cond_io/PIL på begge stier |
| `loaders.mojo` | bygger Mojo-moduler fra torch state_dicts — **dette er mønsteret for WP8-vektlasting**. Siden WP11: `lin_from` hekter `GpuLinear` på når `sd.gpu` er satt |
| `gpu/context.mojo` | **WP11 (2026-07-10)**: delt `GpuContext` (DeviceContext + ArcPointer-scratches for linear/attention + fence-buffer = commit+vent-primitivet) skapt av `TRELLIS2_GPU=1` (`gpu_context_from_env` → CPU-fallback ellers). Kjører OFFER-SYKLUS + VERIFISERT SELVTEST ved opprettelse (femte b2-felle: første komplette skriv→kjerne→les-syklus i en prosess gir korrupte les ved 256-byte-grenser, uavhengig av fencing/warm-up) |
| `gpu/linear.mojo` | **WP11 steg 2 (2026-07-10)**: tiled Metal-GEMM (64×64-fliser, 4×4-registerblokker, 2.2–2.7 TF/s) bak `SparseLinear.forward`. Konteksten rir på StateDict inn i loaderne; W^T lastes opp én gang per modell (`GpuLinear.try_build`); bias adderes på CPU i chunket-parallell readback. Dispatch: co%64==0, ci%16==0, vekt ≥ 2^19, rows ≥ 512, rows·co·ci ≥ 2^32. Fil-headeren dokumenterer fellene 1–4 (bindingsgrense, commit-semantikk, WC-minne-les, Span-lengde) — LES DEN før mer GPU-kode. **Steg 14 (2026-07-11)**: W^T lagres som u16 (bf16-bits << 16 / f16-hardware-cast på shared-fill) når HVER vekt er bit-eksakt representerbar — klassifisering ved try_build (parallell SIMD-or-skann), alle vekt-GEMM-kallsteder dispatcher via `GpuLinear.enqueue_gemm`; `allow_16bit=False` tvinger f32 (test/debug) |
| `gpu/conv.mojo` | **WP11 steg 6 (2026-07-10) + steg 9 (2026-07-11)**: submanifold sparse conv — CSR-sortert gather-kjerne, vekt [K, Ci, Co] på device én gang per modell, ALLE dims i edges-headeren (4-peker-marshalling tillater null skalarer), bias i CPU-readback. Steg 9: 2 rader × 8 co-laner per tråd med kidx-merge-vandring (vektlinjer deles der offsetet finnes i begge rader; bit-identisk per rad). Gates: co%64, E·ci·co ≥ 2^31. 3.9–5.4x på decode-formene. **Steg 15 (2026-07-11)**: vekten lagres som f16-bits når HVER vekt er f16-eksakt (fp16-dekoder-sjekkpunktene = alle sparse convs i pipelinen; bf16 bevisst usupportert — 4-peker-grensen gir ingen format-skalar, to kjerner med host-dispatch); 1.13–1.14x på vekt-tunge former |
| `gpu/attention.mojo` | **WP11 steg 3+4 (2026-07-10)**: SDPA som GEMM-komposisjon med device-resident scores — `gemm_z` (tiled GEMM grid-z-batched over hoder), `softmax_rows_z` (maskert radsoftmax + radsummer; scale forhåndsbakt i q), av-GEMM, 1/sum fusjonert i parallell CPU-readback. BEGGE sider paddes til 64 (q-null-rader droppes i readback → vilkårlige lengder). Innganger: `gpu_dense_sdpa` ([1,L,H,D]) + `gpu_varlen_sdpa_single` ([T,H,D], B=1-segmentet). Wiret i dense `MultiHeadAttention` (self/rope/cross) og `SparseMultiHeadAttention` (full self + cross, enkeltsegment); gate: L ≥ 2048, Lkv ≥ 512, D%64==0, scores ≤ 2^28. self-4096: 3.62x, cross: 2.91x, slat-self @2369: 2.92x. **Steg 7 (2026-07-11)**: `gpu_attn_self_chain` kjeder HELE self-attention device-resident (qkv-GEMM → fused bias+rms+rope-kjerne → pack → sdpa → unpack → out-GEMM); per-MHA-konstanter i ÉN buffer (`GpuAttnChain`), sdpa-skala regnes i-kjerne fra Int d (ingen float-skalarer). Hel-MHA: ss_flow-geometri 1.90x, slat 2.35x mot ukjedet GPU. **Steg 8 (2026-07-11)**: `gpu_attn_cross_chain` kjeder cross-q-siden (q-GEMM → bias+q-rms → sdpa → out) med kv + k-rms på CPU, host-pakket før enqueue-ene; gate = kun sdpa-gaten. Hel cross-MHA: ss 1.68x, slat 1.56x mot ukjedet |
| `meshing/vertex_attrs.mojo` | **WP15 (fase 2, 2026-07-11)**: `grid_sample_trilinear` — ren-Mojo port av flex_gemm grid_sample_3d (trilinear, B=1) med EKSAKT referansesemantikk (trunkerte p±0.5-naboer inkl. duplikat-quirken ved p<0.5, vekt prod(1−\|nabo+0.5−p\|), miss = vekt 0, renormalisering clamp 1e-12; pakket-nøkkel-Dict som fdg_mesh) + arealvektede vertex-normaler |
| `io/glb.mojo` | **WP15 (fase 2, 2026-07-11)**: ren-Mojo GLB 2.0-writer — POSITION/NORMAL/COLOR_0 (VEC4 f32, clampet [0,1]) / u32-indekser + pbrMetallicRoughness med GLOBALE metallic/roughness-faktorer; `to_glb_axes` speiler oppstrøms akse-swap (y,z→z,−y); to write_bytes-kall (byte-appendet hode + peker-fylt binær-payload) |
| `gpu/block.mojo` | **WP11 steg 10+12 (2026-07-11)**: hel-blokk- og modellnivå-residens — `gpu_cross_block_forward` kjører hele cross-blokken (dense + sparse deler den) i ÉN kø: ln+modulate/gate_add som GPU-kjerner (`ln_mod_rows`/`gate_add_bias_rows`, out-biasene foldet inn), kjedene som enqueue-deler, glue-consts i én bk-buffer med Int-offsets. Steg 12: splittet i upload/enqueue/readback-primitiver så modell-forwardene holder x resident over ALLE blokker (én transfer-rundtur per forward; bit-identisk — residens-driver-testen gir diff 0.0). Blokk 249→211 ms |

### Testarkitektur (tests/parity/)

Hver `torch_ref_wpN.py` kjører ORIGINALEN (CPU, naive backends) på seedede
caser og eksponerer resultater + state_dicts; `parity_wpN_vs_torch.mojo`
bygger Mojo-motparten via `loaders.mojo`, kjører samme data og sammenligner
(atol 2e-5–1e-4 avhengig av dybde). `torch_ref_wp5.py` patcher
`correct_sdpa` inn i originalens attention (se bugs under) — **wp7-referansen
importerer wp5 og arver patchen; det må enhver ny ref-fil som bruker sparse
attention også gjøre.**

## Bugs funnet i originalen (viktig for verifisering)

1. **Sparse attention naive/sdpa-fallback er feil**: null-padder k/v uten
   maske → attenderer til padding ved ulike batchlengder (avvik ~1.8).
   Flash/xformers-backendene er korrekte; Mojo-porten matcher dem. CPU-kjøring
   av originalen gir altså stille feil svar — ikke bruk den som fasit direkte.
2. `SparseTensor.to_dense()` med none-backend krasjer (`list + tuple`,
   basic.py:687). Riktig semantikk i `tests/parity/torch_ref.py:dense_ref`.
3. `SparseLayerNorm` (sparse/norm.py) ville krasjet ved kall — ubrukt.
4. Windowed attention har ingen CPU-backend i originalen overhodet.
5. Fantom-eksporter i `sparse/__init__.py`: `serialize`-modulen,
   `SparseSubdivide`, interpolate-funksjonene finnes ikke.
6. `SparseStructureFlowModel`s `rope_freq`-argument er virkningsløst:
   rope-fase-bufferet beregnes alltid med default-frekvenser (1, 10000);
   argumentet videresendes bare til blokkene, som aldri bruker det (fasene
   kommer utenfra). Mojo-porten speiler dette.
7. `patchify`/`unpatchify` i `modules/spatial.py` er død kode — ingen
   kallere; kun `pixel_shuffle_3d` brukes (av SS-VAE-dekoderen).
8. SS-VAE-en videresender aldri `norm_type` til res-blokkene sine —
   `ResBlock3d(ch, ch)` konstrueres uten argumentet og faller tilbake til
   default «layer» (middle + stages, både encoder og decoder); `norm_type`
   påvirker kun `out_layer`. Mojo-loaderen speiler dette (kostet en
   debug-runde å finne — paritetsavvik ~1.0 i første middle-blokk med
   «group»).
9. mtlmesh `cumesh/remeshing.py:220–231` (remesh_narrow_band_dc):
   split-valgets align-test leser kolonne 1,2,3 av 6-indeks-raden i
   stedet for trekant 2 (kolonne 3,4,5) — «align» sammenligner trekant
   1 med seg selv (split 1) og en degenerert null-normal (split 2), så
   split 1 velges alltid. WP18-porten implementerer intensjonen
   (|n(tri1)·n(tri2)| per reell splitt) og dokumenterer avviket.

## Mojo 1.0.0b2-syntaks (lært via kompilatoren — gjelder all ny kode)

- `fn` er FJERNET — kun `def`, og `raises` må angis eksplisitt (nesten alt
  vårt raiser). `alias` → `comptime`.
- Struct-parametre: `Self.dtype`, ikke `dtype`, i felt/metoder.
- Ingen implisitt kopi: returner lokale med `return x^`; listelitteraler med
  ikke-trivielle typer trenger `.copy()`/`^` per element. Delvis transfer ut
  av struct-felt (`s.field^`) er ulovlig — bruk `.copy()`.
- Stdlib under `std.`: `std.math`, `std.memory.ArcPointer`, `std.testing`,
  `std.python`. PythonObject→skalar: `Int(py=obj)` / `Float64(py=obj)`.
  Mojo-list → Python: bygg `Python.list()` manuelt.
- `ref` og `case` er reserverte ord (traff begge som variabelnavn).
- Pakker krever `__init__.mojo` i HVER mappe (manglende fil gir «does not
  refer to a nested package»). Kjør alltid fra repo-rot med `-I .`.

## WP8 — modellene (ferdig)

Alle fire steg porterte og paritetsverifiserte i `test-wp8`
(`torch_ref_wp8.py`/`parity_wp8_vs_torch.mojo`):
1. ✅ `models/sparse_structure_flow.py` → `models/sparse_structure_flow.mojo`.
   Modellen bruker IKKE patchify — forward flater bare romdimensjonene.
   Dense rope kom med denne (`modules/rope.mojo` + phases-overloads); det
   ekte sjekkpunktet (`configs/gen/ss_flow_img_dit_1_3B_64_bf16.json`)
   kjører rope + share_mod + qk_rms_norm begge veier.
2. ✅ `models/sparse_structure_vae.py` (decoder) →
   `models/sparse_structure_vae.mojo` (+ `modules/spatial.mojo`
   pixel_shuffle_3d, `modules/conv.mojo` dense Conv3d). Merk funn 8: res-
   blokkene bruker alltid «layer»-norm.
3. ✅ `models/structured_latent_flow.py` → `models/structured_latent_flow.mojo`
   inkl. concat_cond (tekstur-SLat-stien); liste-cond (VarLen kryss-kontekst)
   er IKKE portert — pipelines sender dense [N, Lc, C]. Ekte config er rope +
   share_mod + qk_rms begge veier.
4. ✅ `models/sc_vaes/{sparse_unet_vae,fdg_vae}.py` (decoder) →
   `models/sc_vaes/*.mojo`. Configene bruker kun ConvNeXt-blokker +
   C2S-oppsampling; shape-dekoderen predikerer subs (pred_subdiv=True,
   `forward` → (h, subs)), tex-dekoderen styres av dem (pred_subdiv=False,
   `forward_guided`) — paritetstesten speiler den handoveren, pluss
   `upsample_coords` og FDG-hodet. Encoder-siden og ubrukte blokk-typer
   (SparseResBlock3d/Upsample3d) er ikke portert; C2S med subdiv=None
   støttes ikke (pipelines sender alltid guide_subs). Torch-fella: dekoderen
   må stå i `.eval()` — forward har en treningsgren som returnerer 3 verdier.

Fremgangsmåte som har fungert per WP: les kilden → sjekk hva inferens-stien
FAKTISK bruker (mye er dødt/trening) → port → `torch_ref_wpN.py` med
randomiserte vekter + state_dict → `parity_wpN_vs_torch.mojo` via
loaders.mojo → pixi-task → utvid `test-all` → oppdater tracker +
MOJO_STATUS. Husk env-guardene øverst i ref-filer (flash_attn-import
feiler ellers) og wp5-importen når sparse attention er involvert.

## WP9 del 1+2 — pipeline-kjernen (ferdig)

`trellis2_mojo/pipelines/image_to_3d.mojo` porterer ALLE modell-stadiene i
`Trellis2ImageTo3DPipeline`: `SSFlowVelocity`/`SlatFlowVelocity` plugger
WP8-modellene inn i WP2-sampleren (sampler-matten er formuavhengig, så
sparse sampling gjenbruker den dense løkka — feats er tilstand, coords/cond
bæres av adapteren; tekstur-concat via `set_concat`).
- Shape: `sample_sparse_structure` (decode → occupancy>0 → maxpool →
  argwhere), `sample_slat` (mean/std-denorm), `decode_shape`
  (unet + fdg_head).
- Cascade/1024: `upsample_coords` → `cascade_coords`
  (kvantiser `int((c+0.5)/lr_res*k)` → unique(dim=0)-ekvivalent sort/dedupe
  → token-budsjett-løkke, −128 per runde med 1024-gulv) → HR-sampling.
- Tekstur: `normalize_slat` (shape-slat normaliseres med SHAPE-mean/std) →
  sampling med concat → `decode_tex` (guided av shape-subs, `*0.5+0.5`).

Integrasjonstest: `pixi run test-wp9`
(`torch_ref_wp9.py`/`parity_wp9_vs_torch.mojo`) kjører identiske vekter og
støy gjennom HELE kjeden (shape → cascade → tekstur) mot originalens
samplere + pipeline-glue. Viktig teknikk der: terskler (`>0` på
occupancy/subdivision/intersected) kan flippe på numerisk drift, så testen
sammenligner rå-tensorer numerisk, verifiserer koordinat-/pooling-logikken
eksakt på torch-tensorene, og godtar bit-avvik kun der torch-logiten er
borderline (< 2e-2).

## WP10 — benchmarks + ytelsespass 1–8 (pass 7–8: 2026-07-09)

`pixi run bench` (`benchmarks/bench_wp10.mojo` + `bench_torch_ref.py`,
samme interop-mønster som paritetstestene) måler Mojo mot torch-originalen
på de varme stiene: full/windowed sparse attention, conv3d, modulert blokk
og hele FlowEuler-CFG-sampling-løkka. `BENCH_TORCH_THREADS=1` pinner torch.
Tall + metodikk + gjenstående optimaliseringskø: `docs/benchmarks/RESULTS.md`.

Seks optimaliseringspass er landet (hele paritetssuiten grønn etter hvert):
- Pass 1 SIMD: `varlen_sdpa` (SIMD qk-dot, av-akkumulering ombyttet til
  per-nøkkel-axpy i kontiguøs out-rad, pekertilgang) og `linear` (SIMD-dot).
- Pass 2 parallellisme: `std.algorithm.parallelize` over (segment, head) i
  `varlen_sdpa`, rader i `linear`, utkanal-blokker i `conv3d` (+ SIMD der)
  — alle med disjunkte skriveregioner → bit-identisk med seriell sti, og
  flops-proxy-terskler for seriell fallback (parallellisering av små ops
  taper: spawn/join × ~400 kall per sampler-trajektorie).
- Pass 3: q-tiling i `varlen_sdpa` (8 q-rader per tile, nøkkel-løkka
  ytterst → k/v-rader gjenbrukes fra L1; bit-identisk) + SIMD i
  `LayerNorm32`/`activation`/`modulate`.
- Pass 4 registerblokker + lastbalanse: qk kjører 2 nøkler × 4 q-rader per
  blokk (8 uavhengige FMA-kjeder, delte laster), av akkumulerer out-raden
  i registre over alle nøkler med divisjonen fusjonert i lagringen, lange
  q-segmenter deles i 64-raders chunks per work-item (attn-L: 16→256
  items på 14 kjerner), og `linear` fikk 4 rader × 2 utkanaler per blokk
  med radblokk-parallelisering. Per-par-matte og per-lane-rekkefølge
  uendret → alt fortsatt bit-identisk.
- Pass 5 GEMM + tensor-lim: `linear` fikk en pakket-GEMM-sti for store
  input (vekt pakket i [k][16]-paneler, x-blokk i [k][4], 4×16-flisen i
  8 registre gjennom hele k-løkka; 270–775 GF/s — MERK: endrer
  akkumuleringsrekkefølge, ikke bit-identisk med dot-stien, men innenfor
  paritetstoleransen og deterministisk). Profilering
  (`benchmarks/microbench_block.mojo`) viste dessuten at blokk-tiden lå i
  det SKALARE tensor-limet, ikke matmulene: SIMD-span-hjelpere i
  `sparse/tensor.mojo` (reshape/unbind/binop/cat/slice/stack — alle
  verdibevarende → bit-identiske) tok shift/scale fra 1.16 til 0.11 ms og
  windowed attention fra 6.1 til 1.7 ms; `activation` parallelisert i
  chunks.
- Pass 6 normer + rope (`microbench_norms.mojo` på modellrealistiske
  former): `MultiHeadRMSNorm` (qk_rms — ekte sjekkpunkter) 6.3→1.1 ms,
  rope `_rotate` via deinterleave/interleave (bit-identisk per-par-formel)
  5.1→2.3 ms, `GroupNorm32` 4.3–8.8→0.6–1.1 ms, `ChannelLayerNorm32`
  (verstingen: skalar per-posisjon-vandring med 16 KB stride, brukes av
  ALLE res-blokkene i SS-VAE-dekoderen per funn 8) 20–33→1.7 ms.
- Netto fra naiv v1: attn-L 1429→15.6 ms, conv 27→1.4 ms, mod-blokk
  208→3.1 ms, sampling-løkka 323→20.6 ms, windowed 21→1.7 ms. **Mojo slår
  nå torch på ALLE caser mot både default-tråder (0.08–0.99x) og 1 tråd
  (0.1–0.6x)** — måltallet i target_metrics.md er nådd over hele linjen.
- Pass 8 (2026-07-09): flash-sti i `varlen_sdpa` for segmenter med
  kv_len ≥ 1024 — fase-attribusjon (`microbench_sdpa.mojo`, degenererte
  ci/co-dimensjoner) viste at 4096-token-attention var bundet av
  scores-materialiseringen (1 GB/forward) og v-re-streaming, IKKE FMA.
  Online softmax over 128-nøkkels blokker, alt i L1; qk beholder
  4×4-registerblokkene, av kjører 2 rader per blokk. IKKE bit-identisk
  (exp-reskalering + blokkvis nevner — pass 5-presedens); alle
  paritetstest-former (< 1024 kv) tar fortsatt den eksakte stien
  (verifisert: cond(128) bit-uendret). Ekte vekter: test-cond 512
  3.4e-5→4.1e-5, test-real ss_flow 1.1e-4 (atol 2e-3). self-sdpa
  284→197 ms (1.44x); e2e-smoke 271→252 s, strukturelt uendret (2 av
  ~950k voxels flippet). MERK for fremtidige numerikk-sammenligninger:
  OBJ er ikke lenger byte-identisk på tvers av flash-endringer —
  sammenlign strukturelt (som MPS-referansen).
- Pass 7 (2026-07-09, ETTER WP13/WP14): e2e-profilering avslørte at
  ss-stadiets 519 s (steps 2) IKKE lå i DiT-en — `linear` gjør 700–900
  GF/s på ekte former og hele cross-blokken ~550 ms
  (`microbench_dit_block.mojo`) — men i dense `Conv3d`
  (`modules/conv.mojo`): SS-VAE-dekoderens kjerne var fortsatt naiv
  skalar én-tråds, MÅLT 0.86 GF/s = 67.6 s (!) per 512-kanals res-conv
  @16³. SIMD over zd-laner (interiør-span, W8+W4-rung), OU=4-utkanal-
  registerblokk med delte x-laster, skalar kant/rest med identisk
  akkumuleringsrekkefølge → BIT-identisk (wp8-paritet + byte-identisk
  smoke-OBJ), 750 ms på samme form (90x). e2e-smoke steps 2 (ren
  kjøring): ss-stadiet 519→162 s, totalt 627→271 s; tall i RESULTS.md.

Mønsteret for flere kernels: `List.unsafe_ptr()` + `load[width=8]`/`store`
+ skalar-restløkke; `@parameter def`-closure + `parallelize[body](n)` (fn
finnes ikke i 1.0.0b2 — def-closures fungerer); behold eksakt torch-numerikk
der terskler er nære (divisjon, max-subtraksjon i softmax), og hold
akkumuleringsrekkefølgen uendret i parallel-stier (disjunkte regioner).
Registerblokk-oppskriften: spesialiser full blokk (f.eks. 2×4) med navngitte
SIMD-akkumulatorer og la kantene falle tilbake til enkeltpar-stien.
Skalar-hjelpere som raiser kan ikke kalles fra parallelize-closures —
inline formelen. Profilér før nye pass (microbench_*-mønsteret);
gjenstående valgfri kø står i RESULTS.md (uinitialisert Tensor-alloc,
større GEMM-kjerner, rope-fasekopien).

## WP11 steg 2 — GPU-linear (FERDIG 2026-07-10)

Den tilede Metal-GEMM-en fra steg 1 er wiret bak `linear`:
`trellis2_mojo/gpu/linear.mojo` + `TRELLIS2_GPU=1` (av som default —
CPU-stien er helt uendret uten flagget). Arkitektur: `GpuContext`
(delt DeviceContext + grow-only A/C-scratch bak ArcPointer + 1-elements
fence-buffer) skapes én gang i runneren og RIR PÅ `StateDict.gpu` inn
gjennom loaderne (null signaturendringer i mellomlagene);
`lin_from`/`_lin_from` hekter en `GpuLinear` (W^T lastet opp én gang)
på `SparseLinear`, som dispatcher per kall på form + flops-terskler.
Bias adderes på CPU i den chunket-parallelle readback-passen.

- **Paritet**: `pixi run test-wp11` (I test-all → 13 testfiler):
  5 DiT-former + 3D-dispatch + decline-caser + StateDict-ride-along,
  max|diff| ≤ 4.3e-6 (atol 2e-4). GPU-en er bit-identisk med naiv
  seriell CPU-dot (sekvensiell k-akkumulering per element); toleranse
  gjelder mot SIMD-/pakket-stiene. Kleine rader tar eksakt CPU-sti
  (bit-verifisert i testen).
- **Ytelse (hele forward inkl. transfer)**: 1.57–1.70x på
  qkv/mlp-formene @4096 tokens, 1.36x @2369; to_out @4096 er
  break-even og små-output-former taper → flops-proxy-terskelen
  rows·co·ci ≥ 2^32 holder dem på CPU. Kjernen alene 2.2–2.7 TF/s;
  gapet er transfer-skatt (se RESULTS.md-tabellen).
- **FIRE nye b2-Metal-feller** (alle verifisert med prober 2026-07-10,
  dokumentert i gpu/linear.mojo-headeren — les den før mer GPU-kode):
  (1) >4 arg-bindinger totalt gir søppel-pekere i kjernen (alle
  skalarer = ÉN binding; 3 ptr + skalarer OK, 4 ptr + 0 skalarer OK).
  (2) `ctx.synchronize()` committer IKKE pending arbeid — kun
  map_to_host av en HOST-SKREVET buffer committer + venter
  (fence-mønsteret); en aldri-host-skrevet buffer mapper til stale
  data for alltid. Steg 1-benchens timing var et forskjøvet vindu
  (hver enqueue committer forgjengeren) — remålt ærlig: samme
  størrelsesorden. (3) Mappet minne er write-combined: seriell LES
  2.2 GB/s (!), parallell 9.2 GB/s, skriv ~8 GB/s. (4)
  enqueue_copy(Span, …) IGNORERER Span-lengden — alltid hel buffer.
- **e2e-smoke med flagget på** (steps 2, --no-tex, shoe_3q.pam seed 42,
  `outputs/wp11_gpu_smoke.obj`): totalt 230 s mot 252 s CPU-referanse
  (~9 %); ss-stadiet 145→128 s, shape-slat ~65→55 s (dekoding er
  CPU-conv, uendret). STRUKTURELT identisk: eksakt samme 2369 voxels
  @32³, 948 567 vs 948 568 voxels @512³ (1 borderline-voxel av ~950k —
  samme klasse som flash-passets 2), bbox lik. Beskjeden e2e-gevinst er
  ventet: bare linears over tersklene offloades, og transfer-skatten
  spiser mesteparten av kjernens 3x.

## WP11 steg 3 — dense SDPA på GPU (FERDIG 2026-07-10)

`trellis2_mojo/gpu/attention.mojo` + refaktor til `gpu/context.mojo`
(GpuContext delt av linear + attention; re-eksport fra gpu/linear
beholder gamle imports). GEMM-komposisjon med device-residente
intermediater — scores-matrisen (1 GB/forward @4096 self, flaskehalsen
som motiverte CPU-flash i pass 8) forlater aldri GPU-en: qk-GEMM
grid-z-batched over hoder → maskert radsoftmax + radsummer (scale
forhåndsbakt i q — ingen float-skalar-args) → av-GEMM → 1/sum fusjonert
i parallell CPU-readback. kv paddes til 64 med maskering; q krever %64.
Wiret i dense `MultiHeadAttention` (self/rope/cross) via
`dense_mha_from`; gate L ≥ 2048, Lkv ≥ 512, D %64, scores ≤ 2^28.

- **Paritet**: `pixi run test-wp11-attn` (I test-all → 14 testfiler):
  4 former (self 2048/4096, maskert odde-kv 1029 og 999) + gate +
  MHA-dispatch, max|diff| ≤ 2.3e-7 (atol 5e-5). Grønn på FØRSTE
  kompilering — probene og selvtesten under gjorde jobben.
- **FEMTE b2-Metal-felle**: den FØRSTE komplette
  map-skriv→kjerne→map-les-syklusen i en prosess gir korrupte les ved
  256-byte-grenser — UAVHENGIG av fencing og per-kjerne-warm-up
  (kjernekompilering er ikke årsaken; kun å brenne én hel syklus
  hjalp, deterministisk over alle probekjøringer). `GpuContext.__init__`
  kjører derfor offer-syklus + VERIFISERT selvtest og kaster → CPU-
  fallback hvis syklus to feiler.
- **Ytelse** (microbench_gpu_attn, hele kallet): self 4096 H16 D64
  207.7→57.4 ms (**3.62x**), cross 4096×1029 58.3→20.1 ms (**2.91x**),
  self 2048 2.87x. Attention per ss_flow-blokk ~266→~78 ms.
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s3_gpu_smoke.obj`): totalt **171 s** mot 230 s (steg 2
  alene) og 252 s (ren CPU) = 1.47x e2e; ss-stadiet 128→**71 s**.
  OBJ-en er BYTE-identisk med steg 2-smoken — forventet: GPU-sdpa
  påvirker kun ss-logits som binariseres til okkupans (ingen av de
  2369 tersklene flippet), og alt nedstrøms er uendret numerikk.
## WP11 steg 4 — varlen/sparse sdpa via q-padding (FERDIG 2026-07-10)

`_sdpa_core` i gpu/attention.mojo padder nå BEGGE sider til 64: q får
null-rader (komposisjonen er q-rad-uavhengig — pad-radenes uniforme
softmax droppes i readbacken), kv får null-kolonner/rader + maske.
Vilkårlige lengder begge veier; q-%64-kravet er fjernet fra
`gpu_sdpa_wants`. Ny inngang `gpu_varlen_sdpa_single` tar
varlen_sdpa-layouten [T, H, D] — B=1-enkeltsegmentet slat-DiT-ene
kjører. `SparseMultiHeadAttention` (full self + cross) dispatcher når
len(offsets)==2 og gaten holder, via `sparse_mha_from` + `sd.gpu`;
windowed/double-windowed og multi-segment blir på CPU (egen batching —
fremtidig steg).

- **Paritet**: test-wp11-attn utvidet — 7 former (inkl. odde T/Tkv
  2369/2113/999) + dense- OG sparse-MHA-dispatch, max|diff| ≤ 2.3e-7.
  Grønn på første kompilering.
- **Ytelse**: slat-self @2369 H16 67.2→23.0 ms (**2.92x**),
  slat-cross @2369×1029 30.0→14.3 ms (**2.10x**).
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s4_gpu_smoke.obj`): totalt **156 s** (steg 3: 171,
  steg 2: 230, CPU: 252 = 1.62x); shape-slat-stadiet 55→**40 s**.
  Strukturelt identisk: samme 2369 @32³, 948 566 vs 948 567 voxels
  (1 borderline-flip av ~950k — flash-presedens), bbox lik til
  7 siffer. Størrelsesposter nå: ss 71 s (varme: dense DiT-glue +
  linears under terskel), decode-conv 40 s, slat 40 s.
## WP11 steg 5 — device-resident mlp-kjeding (FERDIG 2026-07-10)

`gpu_mlp_forward` i gpu/linear.mojo kjeder lin2(gelu_tanh(lin0(x)))
med [rows, hidden]-intermediatet på GPU-en — ss-blokkens 134 MB
WC-rundtur per mlp er borte. lin0-bias adderes på GPU før gelu (ny
`bias_dev` i GpuLinear); `gelu_tanh_bias`-kjernen regner **tanh via
exp** (GPU-bibliotekets tanh er en rask approksimasjon med ~2e-3 avvik;
exp er presis — softmaxen beviste 2.3e-7); A-scratchen gjenbrukes som
utgangsbuffer for andre GEMM (fri etter første). Wiret i
`SparseFeedForwardNet` + dense `FeedForwardNet` når begge linears er
GPU-kvalifiserte. Runneren fase-instrumentert ([ss]/[slat]-linjer:
last vs sampling).

- **Paritet**: test-wp11 utvidet — mlp-kjede (1500×1024→4096→1024) +
  SparseFeedForwardNet-dispatch, max|diff| 1.6e-7 (atol 5e-4).
- **Ytelse** (microbench_gpu_linear): ss-mlp 4096×1536→8192 chain
  **84.2 ms** vs 131.8 ukjedet-GPU vs 279.1 CPU (**3.31x**); slat-mlp
  @2369: 59.1 vs 87.9 vs 178.4 (**3.02x**).
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s5_gpu_smoke.obj`): totalt **136 s** (steg 4: 156,
  CPU: 252 = **1.85x**); ss-sampling 66→59 s, slat-sampling 36→28 s.
  Strukturelt identisk: samme 2369 @32³, 948 578 vs 948 566 voxels
  (12 borderline-flips av ~950k — gelu-numerikken endret seg i ALLE
  mlp-er; bbox lik til 7 siffer).
- **PROBE-LÆRDOM**: bruk GpuContext (verifisert selvtest) i alle
  GPU-prober — håndrullede offer-sykluser er ikke pålitelige
  (256-byte-korrupsjonen overlevde en; kostet en feilsporingsrunde der
  «tanh-upresisjon» viste seg å være én korrupt celle på indeks 192).

## WP11 steg 6 — sparse conv på GPU (FERDIG 2026-07-10)

Decode-instrumentering (midlertidig, revertert) viste at conv dominerte
decode: 23 s ConvNeXt-convs + mesteparten av 17 s upsample-blokker av
43 s. `trellis2_mojo/gpu/conv.mojo`: edge-listene fra SparseConv3d-ens
cachede naboskapskart counting-sorteres STABILT til CSR per target på
host; gather-kjernen beregner hver utgangsrad ved å vandre radens
edge-range (tråd = rad × 8 co-laner — to vec4-akkumulatorer deler
x-broadcasten, naborad-tråder leser koalescerte vektlinjer; 8-lane-
utvidelsen tok 1.7–2.0x → 3.0–3.5x). Vekten lastes opp én gang per
modell som [K, Ci, Co] (`GpuSparseConv.try_build` via
`sparse_conv3d_from` + sd.gpu); x/edges per kall; bias i
CPU-readbacken. MARSHALLING: 4 pekere → null skalarer → alle dims rir
i edges-headeren (int32: n, ci, co, E + row_start + src + kidx).
Gates: co%64==0 (build), E·ci·co ≥ 2^31 (dispatch — små convs forblir
bit-eksakt CPU). Grid 70k threadgroups og Int32-buffere probet OK.

- **Paritet**: `pixi run test-wp11-conv` (I test-all → 15 testfiler):
  4 former (inkl. rektangulær 128→512/512→256 og dilation 2) + gates +
  CPU-fallback + ride-along, max|diff| ≤ 4.1e-5 (atol 2–5e-4).
- **Ytelse**: 512ch@12k 3.03x, 256@55k 3.55x, 128@216k 3.10x,
  up-conv1 512→2048 2.67x (tallene inkluderer CSR-bygg + transfers).
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s6_gpu_smoke.obj`): decode **43→19 s** (ConvNeXt-sum
  23→10.9 s, up-blokker 17→8.4 s), totalt **116 s** = **2.17x** mot ren
  CPU. Strukturelt identisk (948 578 voxels, bbox lik til 7 siffer).
  Fordeling nå: ss 61 s, slat 30 s, decode 19 s.
- **Neste**: (a) ✅ kjede qkv→(rms/rope)→sdpa→out — GJORT som steg 7
  (under); (b) windowed attention-batching (senere STRØKET — se steg
  9); (c) ✅ fp16-vekter — ytelse MÅLT DØD (steg 11-negativene),
  16-bits lagring GJORT som steg 14;
  (d) conv-kjerne-registerblokking (rad-par som deler vektlinjer)
  hvis decode skal lenger ned; (e) ✅ cross-attention-kjeding — GJORT
  som steg 8 (under).

## WP11 steg 7 — device-resident attention-kjeding (FERDIG 2026-07-11)

`gpu_attn_self_chain` i gpu/attention.mojo kjeder HELE self-attention
på GPU-en: qkv-GEMM → `bias_rms_rope_qkv` (fused kjerne, tråd per
(rad, hode), kun gyldige rader; per-element-formler identiske med CPU-
MultiHeadRMSNorm/_rotate) → `pack_q_z`/`pack_kv_z` (head-major, sdpa-
skala regnet i-kjerne fra Int d — ingen float-skalarer; kv-pads NULLES:
av-GEMM-en multipliserer v-pad med 0 og 0·NaN ville forgifte utgangen)
→ steg 3-komposisjonen → `unpack_o_z` (1/sum fusjonert; ALLE mp-rader
skrives så out-GEMM-en ikke leser stale scratch) → out-GEMM. Kun x
lastes opp, kun [T, C] leses tilbake. Per-MHA-konstantene (qkv-bias +
begge rms-gammaene) rir i ÉN device-buffer (`GpuAttnChain`, bygget av
`dense_mha_from`/`sparse_mha_from` ETTER gamma-tilordningen) —
4-bindings-loven. Scratch-gjenbruk som mlp-kjeden: linear-A = x og
senere ut-GEMM-utgang, linear-C = qkv og senere attention-utgang (køen
kjører i rekkefølge); rope-faser lastes opp per kall (ny ph-scratch i
GpuAttnScratch; dense faser kommer utenfra, sparse fra koordinatene via
SAMME spatial-cache som CPU-stien). Gate `chain.wants`: sdpa-gaten +
qkv-linearens flops-terskel — out-linearen rir gratis (dens solo-
break-even ved 4096×1024×1024 gjelder ikke i kjeden). Windowed/multi-
segment/cross blir på steg 3/4-stiene.

- **Paritet**: test-wp11-attn utvidet — 3 kjedede helhets-MHA-caser
  (dense plain, dense rms+rope, sparse rms+rope fra coords; C=1024 —
  mindre vekter går under GPU_MIN_WEIGHT så C=256-casene kan ikke
  treffe kjeden) + gate-/decline-sjekker (cross-MHA bygger ingen
  kjede), max|diff| ≤ 4.8e-7 (atol 5e-4). Grønn på FØRSTE kjøring.
- **Ytelse** (microbench_gpu_attn, HEL MHA-forward inkl. rms/rope;
  «ukjedet» = steg 2-linears + steg 3/4-sdpa med CPU-rundturer):
  dense 4096×1536 H12 D128 (ss_flow-geometrien — merk D=128, dekkes av
  D%64-gaten) 171.8→**90.3 ms** (**1.90x**; 5.98x mot CPU 540 ms);
  sparse 2369×1024 H16 D64 (slat) 66.4→**28.2 ms** (**2.35x**; 4.19x
  mot CPU).
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s7_gpu_smoke.obj`): totalt **94 s** (steg 6: 116,
  CPU: 252 = **2.68x**); ss-stadiet 61→**48 s**, slat 30→**23 s**,
  decode 18 s. Strukturelt identisk: EKSAKT samme 948 578 voxels
  @512³ og 2369 @32³, 2 055 490 vs 2 055 492 triangler (2 borderline-
  flips av ~2M — flash-presedens), bbox lik.

## WP11 steg 8 — cross-attention-kjeding (FERDIG 2026-07-11)

Blokk-profilering med GPU på (`benchmarks/microbench_gpu_block.mojo` —
NY: kjører den EKTE cross-blokk-sekvensen via loaderne med sd.gpu, på
ekte ss_flow-geometri) viste etter steg 7: self 87.2 ms (kjedet),
**cross 92.6 ms = største post**, mlp 80.6 ms, glue ~28 ms → 289.6
ms/blokk. `gpu_attn_cross_chain` i gpu/attention.mojo kjeder q-siden:
q-GEMM → `bias_rms_q` (fused bias+q-rms) → `pack_q_z` (fikk
stride-param: 3HD for fused qkv, HD for cross-q) → sdpa-komposisjonen →
`unpack_o_z` → out-GEMM. kv beregnes og k-rms-normaliseres på CPU-en
(to_kv @~1k kontekstrader er under GPU-GEMM-break-even) og HOST-PAKKES
FØR enqueue-ene — map av host-skrevet buffer committer pending arbeid,
så rekkefølgen er lovpålagt. `GpuAttnChain.try_build_cross` bygger
[HD q-bias][HD gamma_q]-consts (is_cross-flagg); bygges av begge
mha_from-loaderne når is_cross (dummy-to_q gir None for self-MHA-er og
vice versa). Gate `wants_cross` = KUN sdpa-gaten: q/out-GEMM-ene rir på
opplastingen sdpa-en trenger uansett — verifisert: slat-cross vinner
1.56x selv om q-linearen alene er under proxy-terskelen.

- **Paritet**: test-wp11-attn utvidet — 3 cross-caser (dense rms,
  dense plain, sparse rms @T=2369) + build-decline-sjekker begge veier,
  max|diff| ≤ 4.2e-7 (atol 5e-4). Grønn på FØRSTE kjøring.
- **Ytelse** (hel cross-MHA inkl. CPU-kv): ss 4096×1029 C1536 H12
  86.3→**51.4 ms** (**1.68x**; 3.02x mot CPU); slat-geometri 2369×1029
  C1024 H16 44.0→**28.1 ms** (**1.56x**). Blokk-totalen (ss):
  289.6→**244.7 ms**.
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s8_gpu_smoke.obj`): totalt **88 s** (steg 7: 94,
  CPU: 252 = **2.86x**); ss-stadiet 48→**45 s**, slat 23→**20 s**,
  decode 19 s. Strukturelt identisk: 948 577 vs 948 578 voxels
  (1 borderline-flip), bbox lik.
- **Neste i GPU-køen**: se steg 9 under; deretter glue-ops på GPU /
  hel-blokk-residens og fp16-vekter.

## WP11 steg 9 — conv-registerblokking med rad-par (FERDIG 2026-07-11)

`sparse_conv_gather` i gpu/conv.mojo kjører nå TO target-rader × 8
co-laner per tråd: radenes kantlister merge-vandres på kidx — stigende
innen hver rad fordi kidx-major-byggerekkefølgen (k-ytterst-løkka i
`_neighbor_map`) overlever den stabile counting-sorten — så
kernel-offsets som finnes i BEGGE rader laster hver vektlinje ÉN gang
for to raders akkumulering. [ci, co]-vektplan-slicene er dominerende
trafikk, og de fleste decode-rader deler de fleste av de 27 offsetene.
Per-rad kantrekkefølge og per-akkumulator-matte er UENDRET →
**bit-identisk med enkeltrad-kjernen** (verifisert: steg 9-smoke-OBJ-en
er BYTE-identisk med steg 8-smoken via `cmp`). Grid-raddimensjonen
halvert; 4 vec4-akkumulatorer per tråd.

- **Paritet**: test-wp11-conv uendret grønn med NØYAKTIG samme
  diff-tall som steg 6 (bit-identisk-designet).
- **Ytelse** (microbench_gpu_conv, inkl. CSR-bygg + transfer):
  512ch@12k 216.7→**152.4 ms** (1.42x på toppen av steg 6; **4.31x**
  mot CPU), 256@55k 230.4→146.6 (1.57x; **5.43x**), 128@216k
  266.6→174.3 (1.53x; **5.07x**), up-conv1 512→2048 994.9→644.9
  (1.54x; **3.91x**).
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s9_gpu_smoke.obj`): totalt **86 s** (steg 8: 88,
  CPU: 252 = **2.93x**); decode 19→**17 s** (resten av decode er
  norm/mlp-glue, CSR-bygg og transfers). OBJ BYTE-identisk med steg 8.
- **KØ-NOTAT**: «windowed attention-batching» er STRØKET — slat-DiT-en
  (structured_latent_flow.py:71) bruker `attn_mode='full'` for alle
  blokker, så windowed attention er ikke i 512-runner-stien overhodet.
- **Neste i GPU-køen**: (a) ✅ hel-blokk-residens — GJORT som steg 10
  (under); (b) ✅ fp16-vekter — kjernetid MÅLT FLAT (steg
  11-negativene), 16-bits lagring GJORT som steg 14.

## WP11 steg 10 — hel-blokk-residens (FERDIG 2026-07-11)

`gpu/block.mojo`: `gpu_cross_block_forward` kjører HELE cross-blokken
device-resident — ln+modulate → self-kjede → gate_add → ln(affine) →
cross-kjede → add → ln+modulate → mlp-kjede → gate_add — i ÉN kø med én
x-opplasting, én barrier og én readback per blokk (mot seks transfers +
~28 ms CPU-glue). Dense og sparse cross-blokk DELER orkestratoren
(identisk struktur på flate feats [T, C]); begge dispatcher bak
`_gpu_block_ok` (alle tre kjede-gatene + norm-layout: norm1/3 uten
affine, norm2 med, eps == 1e-6 — LN-kjernens eps er komptime).

- **Arkitektur**: kjedene refaktorert til enqueue-deler
  (`_attn_chain_enqueue`/`_cross_chain_enqueue`/`gpu_mlp_enqueue`) med
  device-buffer inn/ut og UTEN out-bias — host-wrapperne fuser den i
  readbacken som før, blokk-stien folder den inn i
  `gate_add_bias_rows`-kjernen. Glue-constsene (6 par shift/scale/gate
  + norm2-affine + de tre out-biasene) rir i ÉN `bk`-buffer indeksert
  med Int-offset-SKALARER (offset-PEKERE er uprøvd mot
  marshalling-lovene). Cross-kv fikk DEDIKERTE `ckt`/`cvh`-buffere:
  self-kjedens device-pack eier kt/vh i den fusede køen. ALLE
  host-opplastinger (x, consts, faser, kv) skjer FØR enqueue-ene — map
  av host-skrevet buffer committer køen (lov 2), en mid-kø-opplasting
  ville serialisert stille. Nye scratcher: xs/hs/bk (+ ckt/cvh).
- **Paritet**: test-wp11-attn utvidet — 2 hel-blokk-caser (dense
  rms+rope + share_mod, sparse rope-fra-coords; C=1024) + gate-sjekker,
  max|diff| ≤ 3.7e-6 (atol 1e-3; driften er LN-seriell-akkumulering).
- **Ytelse** (microbench_gpu_block, ss-geometri): blokk
  249.2→**211.4 ms** (**1.18x**; glue 27.9 ms CPU → i køen, 4 av 6
  transfers borte).
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s10_gpu_smoke.obj`): totalt **77 s** (steg 9: 86,
  CPU: 252 = **3.27x**); ss-stadiet 45→**40 s**, slat 20→**16 s**,
  decode 17 s. Strukturelt identisk: 948 565 vs 948 577 voxels
  (12 borderline-flips av ~950k — LN-numerikken endret seg i alle
  blokker, gelu-pass-presedens), samme 2369 @32³, bbox lik.
- **Neste i GPU-køen**: se steg 11+12 under.

## WP11 steg 11 — CSR-caching + målte negative resultater (2026-07-11)

**To kø-punkter LUKKET med måling** (microbench_gpu_gemm på ekte
DiT-former): (1) GEMM-registerblokking: 64×128/4×8-variant +6 %,
128×128/8×8 +3 % — kjernen er IKKE register-bundet. (2) bf16-lagret
B-vekt (u16<<16 på shared-fill; EKSAKT for DiT-ene siden
sjekkpunktene ER bf16): FLAT innenfor støy — B-flisene L2-caches, så
bf16/fp16-vektlagring kjøper KUN lastetid/minne. Kjernen ligger på
~2.9 av ~9 teoretiske TF/s (shared-memory-/issue-bundet); uten
simdgroup_matrix i 1.0.0b2 er videre GEMM-tuning lav-ROI.

**CSR-caching**: conv-CSR-sorten avhenger kun av kantene —
`SparseConv3d.forward` spatial-cacher nå den sorterte
(row_start, src, kidx)-tripletten per coords/kernel/dilation (sorten
kjørte per KALL før); `GpuSparseConv.forward` tar forhåndssorterte
lister og fyller int32-packen chunk-parallelt. Bit-identisk
(smoke-OBJ BYTE-identisk med steg 10); e2e 77 s uendret
(~0,5–1 s besparelse er under utskriftsgranulariteten).

## WP11 steg 12 — modellnivå-residens (FERDIG 2026-07-11)

`gpu_cross_block_forward` splittet i primitiver
(`gpu_block_state_upload`/`gpu_cross_block_enqueue`/
`gpu_block_state_readback` + `gpu_block_phases`); blokkene fikk
`_gpu_enqueue_resident` (CPU-prep: mod-chunks + kv/k-rms → enqueue mot
resident xs), og BEGGE DiT-forwardene (ss_flow + slat) holder x
device-resident over ALLE 30 blokker når hver blokk passerer
blokk-gaten: ÉN x-opplasting + ÉN readback per FORWARD (var per
blokk), rope-faser lastes opp én gang per forward. Per-blokk-uploadene
(bk-consts + kv-pack) er inter-blokk-syncene (map committer + venter).

- **Paritet**: BIT-identisk med per-blokk-stien — 2-blokks
  residens-driver vs sekvensiell kjøring i test-wp11-attn:
  max|diff| == **0.0 EKSAKT** (readback/upload-en som droppes var en
  eksakt kopi).
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s12_gpu_smoke.obj`): totalt **72 s** (steg 10/11: 77,
  CPU: 252 = **3.50x**); ss-stadiet 40→**37 s**, slat 16→**14 s**,
  decode 17 s. OBJ BYTE-identisk med steg 11-smoken (`cmp`).
- **Neste**: ✅ bf16-vektlagring — GJORT som steg 14 (under); ellers
  er WP11-GPU-køen høstet. Fordeling nå:
  ss 37 s (last 2 + sampling ~31 + ss-decode ~4), decode 17 s,
  slat 14 s, cond 2 s.

## WP11 steg 13 — sdpa-gulv 1024 + GOLDEN GPU-VERIFISERT (2026-07-11)

Den fulle golden-kjøringen (12 steg + tekstur, shoe_3q seed 42)
avdekket at 1857 slat-tokens falt UNDER det gamle sdpa-gate-gulvet på
2048 — hele slat-DiT-en gikk på CPU (177 s; smoken hadde 2369 og
traff GPU-stien). Målt (microbench_gpu_attn, H12 D128): 1857 self
**3.35x**, 1280 2.14x, 1024 fortsatt **1.81x** på GPU →
`GPU_SDPA_MIN_Q` senket 2048→1024 (test-gate-sjekkene flyttet til
512; under 1024 er umålt). Gulvet mater ALLE gatene (self-/
cross-kjede, hel-blokk, modellnivå), så hele slat-residensen slo inn.

- **Golden GPU-kjøring**: totalt **244 s = 4.1 min** mot CPU-goldenens
  27.4 min = **6.74x** (gammelt gulv: 449 s — slat 177→**48 s**,
  tex-slat 103→**30 s**; ss 138 s, decode 10+11 s). MERK: full
  kjøring gir HØYERE speedup enn smoken (3.50x) fordi DiT-samplingen
  dominerer ved 12 steg.
- **Strukturelt mot CPU-goldenen** (cKDTree/trimesh — samme metode
  som CPU-vs-mac): EKSAKT samme 1857 @32³, 514 604 vs 514 603 @512³,
  V-avvik 0.000 % / F-avvik 0.001 %, bbox-diff 0.0, NN-avstander
  mean 2.9e-08 / p95 4.9e-08 / max 8.8e-04 (< ½ voxel @512;
  CPU-vs-mac-referansen lå på mean 2.1e-3 — GPU-kjøringen er
  geometrisk i praksis SAMME mesh som CPU-goldenen), 0 degenererte
  flater, tex-npz [514 604 × 6] i [-0.011, 1.002]. Artefakter:
  `outputs/shoe_3q_mojo_gpu_seed42.obj` + `_texvoxels.npz`.

## WP11 steg 14 — 16-bits W^T-lagring på device (FERDIG 2026-07-11)

Siste punkt i GPU-køen («bf16-vektlagring kun for lastetid/minne», steg
11-negativene). `GpuLinear` lagrer nå W^T som u16-bits på device når
HVER vekt er bit-eksakt representerbar i 16-bits-formen: bf16 (lav-16
mantissabits null — sant for alle bf16-lastede sjekkpunkter, dvs. alle
DiT-ene; ekspansjon u16<<16 på shared-fill) eller f16 (eksakt rundtur —
fp16-unet-dekoderne; hardware-cast på fill). Klassifiseringen er en
parallell SIMD-or-skann i `try_build` (én cachet lesepass — billigere
enn WC-pakkeskrivingene den halverer); alt annet (tilfeldige
test-vekter, f32-sjekkpunkter som DINOv3) blir f32 som før. ALLE
vekt-GEMM-kallsteder (forward, mlp-kjede, self-/cross-kjedene,
blokk-køen) dispatcher via ny `GpuLinear.enqueue_gemm`;
`allow_16bit=False` på try_build tvinger f32 (test/debug). Ingen nye
env-flagg: ekspansjonen er bit-eksakt, så GEMM-en regner på IDENTISKE
f32-verdier — ikke en numerikk-variant.

- **Paritet**: test-wp11 utvidet — bf16-/f16-kvantiserte vekter velger
  riktig format og er BIT-identiske mot f32-lagring av samme verdier
  (forward + mlp-kjede; kjedene deler enqueue_gemm-dispatchen), blandet
  vekt faller tilbake til f32, allow_16bit=False tvinger f32. Alle
  eksisterende diff-tall UENDRET (f32-stien urørt). Grønn på første
  kompilering. Ekte-vekt-klassifisering verifisert med
  `tests/probe_wfmt_real.mojo` (engangs-probe, ikke pixi-task, leser
  3.3 GB): ss_flow mlp/qkv → bf16, shape_dec ConvNeXt-mlp → f16.
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s14_gpu_smoke.obj`): OBJ **BYTE-identisk** med steg
  12/13-smoken (`cmp`), totalt 71–72 s uendret, stage-tider identiske,
  lastetider uendret på utskriftsgranularitet (skannet koster ~ det
  halverte WC-pakket sparer; +9 s USER-tid, usynlig i veggtid).
- **Minne**: device-W^T for en 1.3B-DiT går fra ~4.8 til ~2.4 GB
  (unified memory — reell RSS-reduksjon i DiT-stadiene). MERK: prosessens
  MAX-RSS er UENDRET (~11–12 GB, målt ±0.5 GB støy) fordi peaken ligger
  i decode-stadiet (948k-voxel-aktiveringer + A/C-scratchene), ikke i
  DiT-vektene.
- **Gjenstående (valgfritt)**: ✅ GpuSparseConv-vekten — GJORT som steg
  15 (under); host-f32-kopien kan IKKE droppes (CPU-fallbacken er
  per-kall på rows-gatene).

## WP11 steg 15 — f16-lagret sparse-conv-vekt (FERDIG 2026-07-11)

Steg 14-mønsteret anvendt på `GpuSparseConv`: vekten [K, Ci, Co] lagres
som f16-bits (hardware-cast per vektlinje-load i gather-kjernen) når
HVER vekt er eksakt f16-representerbar — sant for fp16-dekoder-
sjekkpunktene, som er ENESTE kilde til sparse-conv-vekter (DiT-ene har
ingen sparse conv). bf16 er bevisst usupportert her: 4-peker-
marshallingen gir ingen skalar-binding til et format-flagg, så det er
to kjerner med host-side dispatch — og ingen bf16-conv finnes.
Klassifiseringen deler `wfmt_scan` (refaktorert ut av
GpuLinear.try_build til modulnivå i gpu/linear.mojo — tar Tensor, ikke
peker: mutabilitets-origins gjør peker-signaturer upraktiske på tvers
av borrowed callers). `allow_16bit=False` tvinger f32 som før.

- **Paritet**: test-wp11-conv utvidet — f16-kvantisert vekt velger f16
  og er BIT-identisk mot f32-lagring av samme verdier, blandet vekt
  faller tilbake til f32; alle gamle diff-tall UENDRET. Grønn på første
  kjøring. Ekte shape_dec-conv-vekt klassifiserer f16
  (probe_wfmt_real.mojo utvidet).
- **Ytelse — MERK, ulikt GEMM-en**: der GEMM-ens B-fliser L2-caches
  (bf16 målt FLAT, steg 11), STREAMER gather-kjernen vektlinjene per
  rad-par, så halvert vekttrafikk gir reell gevinst på vekt-tunge
  former (microbench_gpu_conv, nå med f16- vs f32-lagret W på samme
  verdier): 512ch@12k **1.14x**, up-conv1 512→2048 **1.13x**; 256@55k
  1.03x, 128@216k 1.01x (aktivierings-/CSR-bundet). e2e-smoke: decode
  **17→14–15 s**, totalt 71 s (DiT-stadiene uendret).
- **e2e-smoke** (steps 2, --no-tex, seed 42,
  `outputs/wp11s15_gpu_smoke.obj`): OBJ **BYTE-identisk** med steg
  14-smoken (`cmp`) — f16-ekspansjonen er eksakt, ingen
  numerikk-variant.
- **GOLDEN RE-VERIFISERT etter steg 14+15** (12 steg + tekstur, seed
  42, `outputs/wp11s15_golden.*`): BÅDE OBJ OG tex-npz **BYTE-identiske**
  med steg 13-goldenartefaktene (`cmp`) — tekstur-stien (tex-slat-DiT +
  tex-dekoder), som smokene aldri kjører, er dermed også verifisert
  gjennom 16-bits-lagringen. Totalt 247 s (steg 13: 244 s — støy;
  decode 10+11→9+10 s).

## FASE 2 / WP15 — teksturert GLB via vertex-attributter (FERDIG 2026-07-11)

Fase 2 startet på brukerens beslutning; scope i **ADR 0008**
(docs/decisions/0008-phase2-textured-export.md — LES DEN): oppstrøms
`o_voxel/postprocess.py::to_glb` er hardt CUDA-bundet
(cumesh-decimering/UV-unwrap, nvdiffrast-baking, cv2-inpainting) og
finnes ikke på maskinen; uten decimering ligger FDG-verteksene på
voxel-oppløsning, så VERTEX-attributter bærer samme informasjon som
2048²-baking. Runneren skriver nå `prefix.glb` i tillegg til .obj+npz
når tekstur er på.

- **Nytt**: `meshing/vertex_attrs.mojo` (`grid_sample_trilinear` med
  EKSAKT flex_gemm-semantikk — se tabellen — + arealvektede
  vertex-normaler), `io/glb.mojo` (ren-Mojo GLB 2.0-writer,
  `f.write_bytes` probet OK i b2), runner-steget (sampler [T,6]-volumet
  på verteksene i VERDENSKOORDINATER før akse-swappen, som oppstrøms;
  base_color+alpha → COLOR_0, metallic/roughness → globale faktorer =
  gjennomsnitt; full per-voxel-PBR fortsatt i npz-en).
- **Paritet**: `pixi run test-wp15` (I test-all → 16 testfiler):
  3 trilinear-seeds mot ren-torch-reimplementasjon av
  flex_gemm-formelen (max|diff| ≤ 6e-8 — én f32-ulp), tomt volum →
  eksakt 0, normaler ≤ 2.5e-7, akse-swap, GLB-roundtrip (avhengighetsfri
  Python-leser: payload BIT-identisk, materiale/min/max sjekket — MERK:
  JSON-min/max må castes f64→f32 før eksakt sammenligning). Grønn på
  første kjøring.
- **Golden** (12 steg + tekstur, seed 42, `outputs/wp15_tex_golden.*`):
  OBJ OG npz BYTE-identiske med golden-artefaktene (GLB-steget rører
  kun kopier); GLB-steget tar < 1 s på 514k vertekser; GLB-COLOR_0
  matcher en UAVHENGIG vektorisert numpy-referanse (dens volum, samme
  trunkeringssemantikk) på 2.4e-7, materialfaktorer på 1e-11
  (metallic ~0.0001, roughness ~0.885), normaler enhetslengde, trimesh
  (trellis-mac-venvet — IKKE i pixi-env) laster filen rent: 514 604 V /
  1 055 568 F, 33 MB.
- **Gjenstående i fase 2 (bevisst, ADR 0008)**: decimering/remesh +
  UV-baking + inpainting — kun meningsfulle sammen, alle CUDA-bundne
  oppstrøms; per-vertex metallic/roughness har ingen standard
  glTF-kanal (npz-en bærer dem).

## FASE 2 / WP16 — mikrohull-fylling i GLB-eksporten (FERDIG 2026-07-11)

Brukerobserverte mikrohull i skoen — de finnes også i trellis-mac-
outputen: FDG-ekstraksjonen er ikke-vanntett per konstruksjon, og
oppstrøms fyller i CUDA-postprosessen
(`cumesh.fill_holes(max_hole_perimeter=3e-2)`).

**REVIDERT (samme dag)**: første versjon kjedet randkanter til RENE
sykler (åttetalls-splitt + blindvei-revert) og fylte 524/~526 slike på
512-goldenen — men brukeren så FORTSATT hull. CuMesh-kilden viste seg å
være publisert (github.com/JeffreyXiang/CuMesh — src/connectivity.cu +
clean_up.cu), og dens FAKTISKE formulering er en annen:
SAMMENHENGENDE KOMPONENTER av randkanter (union-find over delte
vertekser — ingen kjeding), avvis kun komponenter med
blindvei-vertekser (grad 1; kryss-vertekser grad > 2 er LOV), og fyll
hele komponenten (flettede klynger inkludert) med ÉN centroid
(snitt av kant-MIDTPUNKTER) + trekant (b, a, c) per randkant.
`meshing/postprocess.mojo` er omskrevet til denne formuleringen —
enklere OG riktigere: sykel-vandringen lot alle FLETTEDE
mikrohull-klynger ved FDG-ens non-manifold-kryss stå åpne (målt på
1024-goldenen: 2 204 komponenter / 13 654 randkanter som vandringen
ikke tok). KUN i GLB-eksporten — OBJ/npz forblir rå som oppstrøms
MeshWithVoxel.

- **Test**: `pixi run test-wp16` (I test-all → 17 filer, REN Mojo):
  punktert tetraeder → vanntett med konsistent vinding, terskel
  respekteres, åpen kant står (lukket rand-komponent UTEN blindvei
  avvises kun av perimeter-terskelen), to hull med delt verteks fylles
  som ÉN komponent (cumesh-semantikk), deterministisk, vanntett input
  urørt.
- **Golden-tall (komponent-fyllingen, `outputs/wp16b_*_golden.*`)**:
  512: **929 komponenter** fylt (sykel-vandringen tok 524), 1024:
  **4 103** (vandringen tok 2 080); **0 gjenværende fyllbare
  komponenter** i begge (verifisert med uavhengig numpy-union-find på
  GLB-ene) — gjenværende randkanter (23k @512 / 64k @1024) er
  blindvei-søm-stier som heller ikke cumesh fyller (grad-1-kriteriet).
  OBJ-ene BYTE-identiske med forrige goldens (GPU-determinisme på
  tvers av kjøringer); COLOR_0-sjekkene grønne på begge GLB-ene.
- **REVISJON 2 (samme dag): `repair_non_manifold_edges` portert** —
  brukeren så FORTSATT hull etter komponent-fyllingen. Diagnose på
  1024-goldenen: **100 % av blindvei-verteksene sitter på en
  ikke-mangfoldig kant** — hull-RINGER der én kant deles av 3+ flater
  leses som åpne stier og hoppes over av all fylling (22 087 slike
  komponenter, median 2 kanter). Oppstrøms' svar er å FLETTE
  reparasjon mellom fill-kallene (to_glb: fill → simplify → REPAIR →
  fill → …), og GitHub-issue #105 bekrefter at hull består selv
  oppstrøms uten remeshing. cumesh-reparasjonen (clean_up.cu) er en
  HJØRNE-basert splitting: hvert flatehjørne starter som egen verteks,
  hjørner unioneres kun på tvers av MANGFOLDIGE kanter (nøyaktig 2
  flater, matchet på verteks-id) → ikke-mangfoldige vifter splittes i
  mangfoldige ark med dupliserte posisjoner. På et
  mangfoldig-med-rand-ark er ALL rand lukkede sløyfer → andre
  fill-pass lukker ringene. Portert i `repair_non_manifold_edges`
  (postprocess.mojo, array-union-find over 3F hjørner);
  runner-sekvens fill → repair → fill (decimering/small-components
  hoppes fortsatt over). Test: ring med grad-3-kant (to finner på en
  tetraeder-ringkant) — fill alene 0, repair+fill → rand 0.
  **Golden-tall for sekvensen (`outputs/wp16c_*_golden.*`)**: 512:
  929 + **19 184** komponenter lukket (648 280 V), 1024: 4 103 +
  **53 909** (2 380 344 V); ETTER sekvensen: **0 ikke-mangfoldige
  kanter (var 124k @1024), 0 blindveier (var 46k), 0 fyllbare
  komponenter** — gjenværende randkanter (60k/72k) er utelukkende
  LUKKEDE søm-ringer over 3e-2-terskelen (geometrisk sammenfallende
  ark-grenser, null-bredde). COLOR_0-sjekkene grønne, bounds uendret,
  postprosessen tar 5 s @2.3M V. Hull større enn terskelen er
  modell-generert manglende geometri — oppstrøms' svar på DET er
  remesh-grenen (utenfor scope, ADR 0008).
- **REVISJON 3 (samme dag): `unify_face_orientations` portert** —
  brukeren så FORTSATT det samme, og målingen avslørte hvorfor ALL
  topologi-fiksing var virkningsløs for symptomet: FDG-outputens
  flate-orientering er ~50/50 MYNTKAST (995 684 samme-retning-kantpar;
  23 % korrekt / 23 % feilvendt av avgjørbare flater mot okkupansen).
  Feilvendte flater culles/skyggelegges som HULL, og arealvektede
  vertex-normaler av tilfeldig flippede kryssprodukter kansellerer til
  søppel. Port av cumesh unify_face_orientations (paritets-union-find
  over flater; samme-retning-par = paritet 1) PLUSS en global
  per-ark-avgjørelse — PRØVD I TO VARIANTER OG FJERNET ETTER MÅLING
  (på brukerens instruks om å fjerne det som ikke hjalp): (1) direkte
  okkupans-probing ga 51 % — pipelinens voxels er et overflate-SKALL,
  ikke et solid; (2) flood-fill-avledet inne/ute (256³) ga OGSÅ 51 % —
  FDG-flaten er stedvis FOLDET/dobbel-laget (ytter- og innerhud i samme
  sammenhengende ark), så en konsistent orientering viser nødvendigvis
  begge fortegn til enhver lokal probe: stemmen er et nullsumspill.
  Paritetsdelen VIRKET og beholdes (samme-retning-par 995 684 → 6 845 =
  99.3 %); global retning per ark er udefinerbar for foldede ark, og
  visning garanteres av doubleSided=true i GLB-en (spec-compliant
  viewere culler ikke og flipper normaler for baksider).
  Runner-sekvens: fill → repair → fill → **unify(paritet)** →
  sampling/normaler/GLB. Test: blandet kube → konsistent vinding +
  determinisme. Artefakter: `shoe_512_final.*` / `shoe_1024_final.*`
  (wp16b/c/d/e-mellomversjonene slettet).
- **REVISJON 4 (2026-07-12): Khronos-validatoren fant siste bit** —
  brukeren kjørte gltf-validator på 512-GLB-en: ENESTE feilklasse var
  `ACCESSOR_VECTOR3_NON_UNIT` — normaler med lengde 0, og ALLE lå i
  fyll-CENTROID-verteksene (65 @512 / 379 @1024): flettede/foldede
  komponenters vifter KANSELLERER i arealvekt-summen, og null-normaler
  skygges som svarte prikker som ser ut som... mikrohull.
  `vertex_normals` garanterer nå enhetslengde overalt: kansellerte
  vertekser får snittet av naboenes enhetsnormaler, deretter
  deterministisk +z-fallback (torch-referansen speiler semantikken;
  glTF-krav). Kansellert-vifte-case i test-wp15.
  MERK (2026-07-11): outputs/ ble ryddet — alle avløste
  mellomsteg-artefakter (wp11s*, wp13/14-smokes, wp15/16/16b/16c/17-
  goldens, 3.5 GB → 155 MB) er slettet; kanoniske ankre
  (shoe_3q_mojo_seed42*, shoe_3q_mojo_gpu_seed42*, shoe_3q_mac_seed42)
  + siste wp16d_* beholdt.
- **REVISJON 5 (2026-07-12): `remove_small_connected_components(1e-5)`
  portert** (neste-steg 2 fra prikk-saken). VIKTIG FUNN: cumesh-kilden
  ligger LOKALT i `trellis-mac/deps/mtlmesh/src/{clean_up,
  connectivity}.cu` (mtlmesh = Metal-porten av CuMesh) — ingen
  GitHub-henting nødvendig, og `_remove_faces`/`get_connected_
  components`/`get_manifold_face_adjacency` kunne leses direkte.
  Semantikk: flate-komponenter unioneres KUN over mangfoldige kanter
  (nøyaktig 2 flater — delt verteks eller non-manifold kant kobler
  IKKE), komponentareal = sum av 0.5·|cross(v1−v0, v2−v0)| (f64 her,
  f32 på device hos cumesh — intensjonsport), komponenter < min_area
  mister alle flater, og ureferererte vertekser kompakteres i original
  rekkefølge. Runner-sekvens per oppstrøms to_glb (minus de CUDA-bundne
  simplify-stegene): fill → repair → **remove_small(1e-5)** → fill →
  unify. test-wp16 utvidet (verteks-deling kobler ikke, kant-delende
  par merger med arealsum, grid-patch reddes av SUMMEN, remapping,
  idempotens) — grønn på første kjøring; hele test-all (17 filer)
  grønn. **Golden-tall (2026-07-12)**: @512 fjernes **13 897
  fragment-ark / 60 472 flater** (fyll 929 + 5 287; GLB 648 280 →
  **551 771 V**; 902 ark; 242 s), @1024 **34 597 / 167 997** (fyll
  4 103 + 19 330; GLB → **2 126 855 V**; 81 ark; 861 s). OBJ/npz
  BYTE-identiske med forrige goldens (`cmp` — kun GLB-stien påvirkes);
  check_glb_512/1024 grønne (COLOR_0 2.4e-7/3.0e-7, normaler
  enhetslengde, trimesh rent); uavhengig numpy-union-find: **0
  komponenter < 1e-5 igjen, 0 non-manifold-kanter, 0 blindveier** i
  begge GLB-ene; analyze_boundary @512: 0 fyllbare sløyfer igjen
  (59 794 gjenværende randkanter er sømringer over terskelen som før).
- **REVISJON 6 (2026-07-12): `sew_boundary_seams` — sprekk-syingen**
  (brukerens valg etter A/B-en: «prøv sprekk-sying først», fremfor
  remesh-grenen). EGEN semantikk, IKKE oppstrøms port: etter alle
  fylle-passene er gjenværende randkanter lukkede sømringer der to ark
  møtes med BIT-IDENTISKE duplikat-posisjoner (hjørnesplitten
  duplicerer eksakt, og FDG-ens selvberørende folder emitterer eksakte
  duplikater). Sveiser hver gruppe posisjons-sammenfallende
  RAND-vertekser til én verteks (eksakt f32-bit-likhet, INGEN epsilon
  — nesten-sammenfall er ekte geometri; 63-bits spatial hash av lav-21
  mantissebits + eksakt-likhets-kjede per bøtte, først-sett
  representant = deterministisk), dropper flater som degenererer,
  kompakterer. Sprekk borte topologisk + vertex-normaler midles over
  sømmen (skyggekanten dør); fyll-pass ETTER sveisen tar ringer som
  først da blir lukkbare; unify propagerer vinding over sveisen.
  Runner-sekvens nå: fill → repair → remove_small → fill → **sew →
  fill** → unify. test-wp16 + 5 caser (tvilling-rand 8→6 randkanter,
  1-ulp-avvik urørt, søm-sliver droppes, determinisme, vanntett
  no-op) — grønn på første kjøring; test-all (17 filer) grønn.
  **Golden-tall**: @512 sveises **24 798** rand-vertekser (0
  degenererte), 541 komponenter lukket post-weld; ferdig GLB 527 514 V
  / 1 046 916 F, randkanter **59 794 → 16 456 (−72 %)**. @1024 sveises
  **30 230**, 498 lukket; GLB 2 097 123 V / 4 188 429 F, randkanter
  **72 243 → 19 543 (−73 %)**. OBJ/npz BYTE-identiske begge;
  check_glb-sjekkene grønne (COLOR_0 2.4e-7/3.0e-7, unit-normaler,
  trimesh rent). KJENTE RESTER (bevisste): (1) sveisekryssene
  gjenskaper non-manifold-kanter (6 502 @512 / 8 608 @1024) — ufarlig
  for visning, unify hopper over dem; IKKE kjør repair etter sew (det
  ville splittet sveisene igjen). (2) 145 småsløyfer under
  3e-2-terskelen @512 består: de bor i BLANDEDE randkomponenter med
  vedheftede blindvei-stier, som cumesh-fyllkriteriet avviser — en
  loop-nivå-fylling (fyll lukkede sykler i blandede komponenter) er
  neste kandidat HVIS prikker fortsatt synes.

## YTELSESSPORET (startet 2026-07-12 kveld — brukerens bestilling etter at prikk-saken ble lukket)

Websøk-funn (hva som IKKE var gjort): (1) cache-gjenbruk over
diffusjonssteg (TeaCache/FORA/FasterCache-familien, arXiv:2410.19355);
(2) fp16-aritmetikk i GEMM (Rigel-studien av M4, arXiv:2606.12765:
simdgroup_matrix/MPP gir bare ~1.21x over naiv tiled — IKKE noe stort
uutnyttet tensorspor — men halvpresisjons-ALU dobler gjennomstrømning);
(3) treningsfrie andreordens flow-solvere / færre steg (`--steps`
finnes alt); (4) Mojo nightly har fått innledende Apple-GPU-støtte
(puzzle-nivå, ikke actionable på b2-pinnen). Brukeren valgte
CFG-cache + FORA-blokk-caching.

**CFG-cache IMPLEMENTERT (bak `--cfg-cache`, default AV) — MÅLT
KVALITETSDRIFT, anbefales IKKE som default:** delta-gjenbruk av
guidance-differansen (pos − neg) på annenhver CFG-steg
(flow_euler.mojo::_inference; CFG_CACHE_WARMUP=2 første steg alltid
ekte; aldri to gjenbruk på rad; invariant-tester i
flow-euler-testen: konstant-delta-modell → eksakt innenfor f32-
avrunding, varierende delta → aktiv + deterministisk; test-all
fortsatt 18 grønne). MÅLING @512 golden (--remesh): ss 137→109/113 s,
slat 47→39 s, totalt 256→218/223 s (**~14 %**) — MEN okkupansen
drifter: 1857→1908/1881 @32³ og 514 604→686 617/738 300 (!) @512³
(uten/med warmup), bbox uendret (±2.5 voxler). Diagnose: deltafeilen
forsterkes med (strength−1)=6.5, og guidance-intervallet [0.6,1.0] ER
de tidlige strukturdannende stegene; med bare ~10 CFG-steg av 12 er
per-steg-deltaendringen for stor (FasterCache er utviklet for
30–50-stegs video-DiT-er). Artefakter for visuell dom:
`outputs/shoe_512_cfgcache{,2}.glb` (uten/med warmup).
**Anbefaling videre:** FORA-blokk-caching NEDPRIORITERES (samme
approksimasjonsregime, større flate); neste trygge løft er
**fp16-GEMM** (f16-multiply/f32-akkumulering — ulp-nivå-numerikk, ikke
semantisk approksimasjon; vektene ligger alt i 16 bits på device) og
evt. `--steps 8`-test (veldefinert oppstrøms-skrue).
**BESLUTNING (2026-07-12, brukeren):** cfg-cache-GLB-en viste «par
hull» → cache-sporet DROPPET, FORA STRØKET, og på brukerens
oppfølgingsinstruks er CFG-CACHE-KODEN FJERNET KOMPLETT
(v4-presedensen: fjern det som ikke hjalp) — flow_euler.mojo,
pipelines, runner-flagget og invariant-testene er reverterte til
før-tilstand (verifisert: flow-euler- og wp9-paritet grønne,
runner bygger rent, `grep cfg_cache` = 0 treff). Kun
handover-notatet her dokumenterer forsøket og HVORFOR det røk:
delta-feil ×(strength−1)=6.5 i de tidlige strukturdannende stegene,
514k→686k/738k voxeldrift @512³ for ~14 % fart.

**WP19 fp16-GEMM FERDIG (2026-07-12 — probe-først-disiplinen):**
`microbench_gpu_gemm.mojo` fikk fire prober på ekte DiT-former:
v3 f16-shared-B (+20–25 %), **v4 f16-shared-A+B (+32–40 % — VINNER;
kjernen VAR shared-båndbredde-bundet)**, v5 halv-MULTIPLIKASJON
(TREGERE enn v3/v4 — b2 emitterer ikke lønnsom half-aritmetikk,
DØD) og v6 dobbel-buffring (mye tregere — prefetch-adresseringen
koster mer enn den gjemmer, DØD). Wiring: `gemm_tiled_f16sh`
(gpu/linear.mojo) bak **`TRELLIS2_GPU_F16=1`** (default AV =
bit-eksakt som før): GpuContext fikk f16-felt fra env,
`WFMT_F16SH` velges i try_build for alle 16-bits-kvalifiserte
vekter (bf16-vekter taper kun subnormaler < 6e-8 i f16-castet),
dispatch via `enqueue_gemm`-choke-pointet → forward/mlp-kjede/
attention-kjeder/blokk-kø får kjernen gratis. Paritet: ny case i
test-wp11 (WFMT_F16SH velges med flagg, 3.2e-4 mot CPU på
testdata, avviker fra f32-stien som forventet, av-som-default
verifisert); test-all = 18 filer grønne. **Golden @512 (--remesh,
flagg på): 256→210 s (1.22x)** — ss 137→114 s, slat 47→39 s, tex
30→24 s; EKSAKT samme 1857 @32³, 514 591 vs 514 604 @512³ = **13
borderline-flips** (samme toleranseklasse som gelu-/LN-passene),
bbox lik. Artefakt: `outputs/shoe_512_f16.glb` for visuell sjekk.
GJENSTÅENDE VALG: gjøre flagget til default (krever nye
golden-ankre siden OBJ ikke lenger blir byte-identisk) — brukerens
beslutning.

**WP19 trinn 2 — f16-shared sdpa-GEMM-er (2026-07-12, samme flagg):**
`gemm_z_f16sh` i gpu/attention.mojo (q/k/v/probs castes på shared-
fyllet; f32-matte/akkumulering), dispatch på `g.f16` i
`_enqueue_sdpa_groups` — ALLE sdpa-stier (dense/varlen/self-chain/
cross-chain/blokk-kø) trakter gjennom de to enqueue-ene der. Paritet:
2 nye flaggede caser i test-wp11-attn (2.6e-5 målt, toleranse 5e-3);
alle af-default-tall uendret; test-all 18 grønne. **Ærlig måling**:
kjerne-nivå (/tmp-bench på hele gpu_dense_sdpa-kallet) self-4096
H12 D128 71.4→61.5 ms (**1.16x**), cross 4096×1029 1.09x, 2048² H16
D64 1.12x — MEN e2e drukner i støy: full-f16 215–218 s mot vekt-f16s
210 s (±5 s kjørestøy/termikk; forventet sdpa-bidrag ~3–4 s).
Beholdes bak flagget (aldri tregere utover støy, målbart raskere
isolert). trellis-mac PR #12 (--dit-dtype f16, ~30 % på MPS) VURDERT:
bekrefter f16-retningen uavhengig, ingenting å portere.
f16-SCORES LUKKET MED REGNESTYKKE (2026-07-12, ikke implementert):
scores-trafikken er ~3.2 GB per self-forward, men sdpa-kjeden er
issue-bundet, ikke rent båndbredde-bundet — halveringen ville spart
~3–6 ms/forward ≈ 0.1–0.2 s e2e, mot kirurgi i TRE kjerner
(qk-utgang, softmax, av-inngang) + egen u16-scratch. Ikke verdt det.
--STEPS-TESTEN UTFØRT (2026-07-13, @512 remesh-res 256 f16):
steps 8 → **160 s** (ss 76, slat 26, tex 16; 1931 @32³), steps 10 →
**170 s** (1849 @32³), steps 12-referanse ~210 s. MERK: færre steg er
en LEGITIM oppstrøms-skrue som gir et ANNET (ikke driftet) resultat —
1931/1849 mot 1857 er forskjellige samplingpunkter, ikke feil.
Artefakter for brukerens dom: `outputs/shoe_512_s8.glb` / `_s10.glb`
mot `_r256.glb` (12 steg). Gjenstående kandidat: QEM-decimering
(egen WP, simplify.cu 582 linjer lokalt).

## FASE 2 / WP18 — remesh-grenen: narrow-band dual contouring (FERDIG 2026-07-12, @512)

Brukerens valg etter v7-syingen: «veldig aggressiv metode først,
deretter finjustere» — oppstrøms' remesh-gren, kun 512 inntil videre.
`trellis2_mojo/meshing/remesh.mojo` porterer
`cumesh.remeshing.remesh_narrow_band_dc` (mtlmesh-kilden LEST:
cumesh/remeshing.py + src/remesh/simple_dual_contour.cu): i stedet
for å lappe FDG-topologien ekstraheres OFFSET-flaten **UDF(p) − eps =
0** (eps = band·scale/R ≈ 1 voxel) fra en USIGNERT avstandsfunksjon
rundt originalmeshen — sprekker/hull smalere enn ~2·eps svelges av
skallet PER KONSTRUKSJON, resultatet er lukket med globalt konsistent
vinding (quad-orientering følger felt-fortegnene), og
`project_back=0.9` snapper verteksene 90 % tilbake mot
originalflaten. Runner: `--remesh`-flagget ERSTATTER hele
cleanup-kjeden (fill/repair/small/sew/unify hoppes over) i
GLB-stien; OBJ/npz forblir rå som alltid; fargesampling/normaler er
posisjonsbaserte og virker uendret på den nye topologien.

- **Avvik fra referansen (dokumentert i modul-headeren)**: (1)
  kandidat-voxels stemples direkte fra triangel-AABB-er (superset) og
  filtreres eksakt — oppstrøms' coarse-to-fine-oktre er kun
  akselerasjon, samme voxelsett; (2) avstands-/projeksjonsoppslag via
  uniformt triangel-grid (R/4-celler, CSR, 3³-nabolag, r_max = 4
  voxler — alt DC-en trenger: kryssende kanter har |d| < eps + 1
  voxel) i stedet for cuBVH; (3) f64 internt (f32 på device hos dem);
  (4) **NIENDE oppstrøms-bug funnet** (remeshing.py:220–231):
  split-valgets «align» leser kolonne 1,2,3 av 6-indeks-raden i
  stedet for trekant 2 (kolonne 3,4,5) — sammenligner trekant 1 med
  seg selv (split 1) og en degenerert null-normal (split 2), så
  split 1 velges ALLTID; vi porterer INTENSJONEN (|n(tri1)·n(tri2)|
  per reell splitt, split 1 ved likhet = deres observerbare
  oppførsel på plane quads).
- **Test**: `pixi run test-wp18` (I test-all → **18 filer**): lukket
  kube → vanntett DOBBELT-skall (ytre + indre offset-ark) med
  konsistent vinding OG positivt fortegnsvolum (orienterings-sjekk),
  **punktert kube → FORTSATT vanntett** (sprekk-svelgingen — selve
  egenskapen grenen finnes for), uprojisert skall ligger på ~eps,
  projeksjon trekker til < 0.16·eps, determinisme, tom input. Grønn
  på første fungerende kompilering.
- **Smoke** (steps 2, GPU, --remesh): 948 565 V / 2 055 474 F →
  1 630 426 V / 3 266 276 F; remesh ~25–30 s på toppen av 72
  s-baselinen; **0 randkanter**, 2 202 non-manifold DC-kryss (kjent
  tvetydig-celle-artefakt, usynlig — ingen gap).
- **Golden @512** (12 steg + tekstur, seed 42,
  `outputs/shoe_512_remesh.glb` — EGEN artefakt, sying-finalen
  `shoe_512_final.glb` beholdt for visuell A/B): 514 604 V / 1.06M F
  → **975 708 V / 1 952 036 F, 0 randkanter**, 305 non-manifold
  DC-kryss, positivt fortegnsvolum, unit-normaler, COLOR_0 2.4e-7
  mot numpy-referansen, trimesh rent; totalt 256 s (remesh ~10 s på
  golden-meshen). OBJ/npz BYTE-identiske med goldens (kun GLB-stien).
- **Finjustering (startet samme kveld — brukeren: «mye bedre! enda
  noen få hule» på band-1.0-goldenen)**: runneren fikk
  `--remesh-band B` og `--remesh-project P` (defaults 1.0/0.9;
  scale-oppblåsingen følger band). Genererte varianter for visuelt
  valg: `outputs/shoe_512_remesh_b15.glb` (band 1.5 — svelger
  sprekker < 3 voxler; 969 292 V, 0 randkanter) og
  `..._b20.glb` (band 2.0 — < 4 voxler; 958 123 V, 0 randkanter).
  Avveining: større band svelger bredere hull men glatter mer
  strikke-detalj (projeksjonen henter det meste tilbake;
  skallvolumet vokser 0.00067→0.00102→0.00137 med band).
  Gjenværende skruer: project_back opp (0.95) for skarphet, DC på
  høyere oppløsning enn 512, decimering (utenfor scope). Hull som
  består ved band 2.0 er ekte modell-GROPER (reell geometri).
- **`--remesh-res N` (2026-07-12, brukerens triangel-reduksjons-
  spørsmål)**: DC-gridet frikoblet fra pipeline-oppløsningen —
  triangeltallet skalerer ~N², projeksjonen holder geometrien på den
  ekte flaten (band forblir i DC-voxel-enheter: 1 voxel @256 svelger
  det 2 ville @512). Varianter generert (@512-pipeline, f16-GEMM):
  res 256 → **238 381 V / 477 072 F** (15 MB GLB), res 128 →
  **56 877 V / 113 960 F** (3.6 MB); begge vanntette (0 randkanter),
  unit-normaler. Full kontroll («nøyaktig X flater» + adaptiv
  detaljbevaring) krever QEM-decimering — kilden ligger LOKALT i
  mtlmesh/src/simplify.cu (582 linjer) + driver-løkken i
  cumesh/cumesh.py::simplify; egen WP hvis res-skruen ikke holder.

## FASE 2 / WP17 — 1024-kaskaden + sdpa-hodegruppering (FERDIG 2026-07-11)

Oppstrøms `1024_cascade` (DERES default pipeline_type) wiret i runneren:
`--pipeline 1024`. Flyten (trellis2_image_to_3d.py::run): cond@512 OG
@1024 (4101 tokens), ss@32³ som før, LR-shape-slat med 512-DiT-en →
shape-dekoderens subdivisjons-upsample 4 nivåer → kvantisering til 64³ +
dedupe (`run_cascade_stage`; token-budsjett-løkka er NO-OP for
1024-målet — den bryter alltid på hr==1024) → HR-shape-slat med
1024-DiT-en og cond@1024 → tex-slat med 1024-DiT-en → decode/mesh/npz/
GLB @1024³ (alle stegene var oppløsningsagnostiske; stage-funksjonene
tar nå modellnøkkel). RNG-rekkefølgen LR→HR→tex speiler oppstrøms.

- **SJETTE b2-Metal-felle (probet 2026-07-11)**: kjerneskriv forbi
  4 GiB byte-offset innen ÉN buffer-binding tapes STILLE (allokering
  lykkes, les under grensen er fine) — dokumentert i lovlisten i
  gpu/linear.mojo-headeren. Konsekvens: full-H scores-buffer for
  HR-slaten (T≈12k → 9 GB) kan ALDRI fungere som én binding.
- **Hodegruppert sdpa** (`_enqueue_sdpa_groups`): qk→softmax→av
  enqueues per hodegruppe mot én scores-scratch ≤ 2^28 floats (uendret
  tak); `gpu_sdpa_wants` er nå PER-HODE (mp·lp ≤ 2^28). In-order-køen
  gjør scratch-gjenbruk på tvers av grupper trygt; hg == h reproduserer
  gammel sti eksakt — alle eksisterende paritetstall UENDRET. Ny case i
  test-wp11-attn: self 4160² H16 = 15+1 grupper (tidligere avvist
  form), ≤ 2e-7. Grønn på første kjøring.
- **Smoke @1024** (steps 2, --no-tex, seed 42): HR-slat-sampling
  **698 s (CPU-attention) → 120 s** med GPU = 5.8x; totalt 836→256 s.
  3.26M voxels @1024³ → 6.44M triangler (mot 948k/2.06M @512).
  273 borderline-flips av 3.26M (0.008 %) mellom CPU-/GPU-attention-
  kjøringene — kjent toleranseklasse. Peak RSS 18.4 GB (48 GB-maskin).
- **Golden @1024** (12 steg + tekstur, seed 42,
  `outputs/wp17_1024_golden.*`): totalt **836 s = 13.9 min** (512:
  244 s). 1857 @32³ (identisk med 512-goldenen — samme ss-sti) →
  kaskade 7545 tokens @64³ → **2 058 563 voxels @1024³ = nøyaktig
  4.00x** 512-goldenens 514 604 → 2.06M V / 4.16M F (512: 514k/1.06M).
  Fordeling: ss 138 s, LR-slat 49 s, kaskade 9 s, HR-slat 346 s,
  tex-slat 200 s, decode 33+38 s, mesh 1 s, hullfylling+GLB 2 s
  (2080 mikrohull av 89 321 randkanter). Peak RSS 16.6 GB.
  GLB-sjekk (sparse numpy-referanse — dens 1024³-volum er 26 GB, bruk
  searchsorted): COLOR_0 3.0e-7, materialfaktorer 1e-10, trimesh laster
  rent. MERK: 1024-tex-modellen ga metallic-snitt 0.73 for skoen (512:
  ~0.0001) — modellegenskap, ikke pipeline-avvik (npz-en OG GLB-en
  matcher modellens output eksakt).
- MERK: langvarige kjøringer MÅ startes med nohup/disown — den første
  golden-kjøringen døde med agent-prosessen (dokumentert felle som slo
  til igjen).

## ÅPEN SAK: synlige mørke prikker på strikke-teksturen (per 2026-07-12, revidert etter v6)

Brukeren så små mørke prikker i GLB-en (skjermbilde 2026-07-12:
spredte svarte/mørkerøde prikker KONSENTRERT på den strikkede/humpete
overdelen; sålen og glatte partier er rene; én større lys flekk på
venstre side). Dette er IKKE en regresjon — prikkene har vært synlige
gjennom hele fase 2.

**AVKLART 2026-07-12 (kvantitativ A/B, neste-steg 1)**: trellis-mac-
meshen (oppstrøms MPS-port, samme seed 42, rå OBJ) har SAMME
strukturklasse som vår (`tests/checks/ab_mac_vs_mojo.py`): 26 975
randkanter / 42 180 non-manifold-kanter / 14 857 blindveier / **11 040
fragment-komponenter < 1e-5** mot våre 28 680 / 51 342 / 16 889 /
13 909 — begge har ETT dominant ark (~1.35 areal) + fragmentstøv.
Rå-mesh-patologien er altså MODELLENS NATUR (jf. GitHub-issue #105),
ikke en porteringsartefakt; «fikser» utover cumesh-postprosessen hører
hjemme i remesh-grenen. VISUELL A/B AVGJORT (2026-07-12): trellis-mac
KAN lage teksturert GLB på Mac likevel (`postprocess_cpu.py::to_glb`
= fast_simplification + xatlas + MPS-rasterisering);
`outputs/shoe_3q_mac_seed42_tex.glb` (samme seed; decimert til
113 524 V / 198 681 F, 1024²-bakte PBR-teksturer, DERES fyll/cleanup
via mtlmesh) ble generert, og **brukeren bekreftet: SAMME PRIKKER —
samme resultat** selv etter deres simplify + UV-baking. Saken er
dermed ENDELIG avgjort: prikkene er modellens natur og kan ikke
fjernes med mer mesh-cleanup (verken vår eller deres); eneste
gjenværende fiks-kandidater er sprekk-sying (egen, ikke-oppstrøms
semantikk) eller remesh-grenen (oppstrøms' eget svar — rebygger
topologien).

**Hva som er UTELUKKET/ADRESSERT, med måling (hele mesh-postprosess-sporet):**

| Hypotese | Fiks (alle ren Mojo, cumesh kun LEST som referanse) | Resultat |
|---|---|---|
| Lukkbare mikrohull-ringer | komponent-fylling (WP16 v2) | 929/4103 komponenter lukket @512/@1024 — prikker består |
| Ringer brutt av non-manifold-kanter | hjørne-splitting + refill (v3) | +19k/+54k lukket; 0 blindveier/0 fyllbare igjen — består |
| ~50/50 tilfeldig vinding (culling/normal-søppel) | paritets-unify (v4) | 995 684 → 6 845 gale kantpar — består |
| Global inne/ute-retning | probe-stemme, 2 varianter | MÅLT nullsum (51 %) — flaten er FOLDET; fjernet |
| Null-normaler i fyll-centroids | nabo-snitt + +z-fallback (v5, funnet av brukerens Khronos-validatorrapport) | 65/379 → 0; validator ren — består |
| Småfragment-ark som skygger mørkt | remove_small_connected_components(1e-5) (v6, 2026-07-12) | 13 897/34 597 ark fjernet @512/@1024 — **visuell effekt IKKE verifisert ennå** (brukeren må se på de nye GLB-ene) |
| Sprekker/skyggekanter langs sømringene | sew_boundary_seams (v7, 2026-07-12, EGEN semantikk — brukerens valg) | 24 798/30 230 vertekser sveiset; randkanter −72/−73 %; normaler midles over sømmen — **visuell effekt IKKE verifisert ennå** |

**Gjenstående forklaringer (rangert):**
1. **Sprekker mellom selvberørende folder**: de gjenværende
   randkantene (60k @512 / 72k @1024 i ferdig GLB) er lukkede
   SØMRINGER over terskelen — og strikke-humpene er nettopp
   selvberørende folder. Fra skrå vinkler ses interiøret gjennom
   sprekken = mørk prikk.
2. **Modellgenererte groper**: dype smale hulrom i humpene der
   FDG-overflaten faktisk dipper inn — mørke av skygge/interiørfarge.

**Neste steg (per 2026-07-12 kveld — BÅDE v7-syingen OG WP18-remeshen
er implementert; brukeren valgte «aggressivt først, så finjustere»,
kun 512):**
1. **Visuell sjekk (brukeren), to kandidater @512**:
   (a) `outputs/shoe_512_remesh.glb` — WP18 dual contouring,
   GARANTERT 0 randkanter (hele sprekk-familien død per
   konstruksjon); (b) `outputs/shoe_512_final.glb` — v7-syingen
   (randkanter −72 %, skarpere FDG-detalj). Er remeshen ren nok →
   finjuster; viser den fortsatt prikker, er de ekte modell-GROPER
   (reell geometri — da er teksturen/modellen kilden, ikke meshen).
2. **Finjustering av remeshen** (skruene står i WP18-seksjonen):
   band ned/opp, project_back opp for skarphet, DC på høyere
   oppløsning enn 512, evt. decimering etterpå.
3. 1024-varianten av remeshen når 512 er godkjent (alt er
   oppløsningsagnostisk; kun kjøretid/minne å verifisere).

Verktøy fra øktene, nå i repoet (`tests/checks/`, kjør fra repo-rot
med trellis-mac-venvets python — trimesh er ikke i pixi-env):
`check_glb_512.py` / `check_glb_1024.py` (COLOR_0/materiale/
normal-sjekk mot npz + trimesh-lasting; rediger GLB/NPZ-stiene øverst),
`analyze_boundary.py` (randkant-/sløyfeanalyse på 512-GLB-en) og
`ab_mac_vs_mojo.py` (rå-OBJ-strukturanalyse mojo vs trellis-mac:
randkanter, non-manifold, blindveier, fragment-komponenter). Khronos
gltf-validator (gltf-viewer.donmccurdy.com) er del av
verifiseringsloopen nå: brukerens rapport fant v5-feilen.

## WP9 del 3 — runner med ekte vekter (steg 1–5 FERDIG)

Alt som manglet ligger nå i `~/Documents/testfiler/trellis-mac/`
(github.com/shivampkumar/trellis-mac — en FUNGERENDE MPS-port av TRELLIS.2
som har generert ekte meshes på denne maskinen: `olav_statue_v*.obj`,
2.–6. juli). Kartlagt 2026-07-08:

- **Sjekkpunkter**: HF-cachen (`~/.cache/huggingface/hub/`) har
  `microsoft/TRELLIS.2-4B` (14 GB) med nøyaktig de ckpt-navnene loaderne
  forventer: `ss_flow_img_dit_1_3B_64_bf16`,
  `slat_flow_img2shape_dit_1_3B_{512,1024}_bf16`,
  `slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16`,
  `{shape,tex}_dec_next_dc_f16c32_fp16` (+ `pipeline.json`), pluss
  `microsoft/TRELLIS-image-large` (SS-VAE-dekoderen
  `ss_dec_conv3d_16l8_fp16`), `facebook/dinov3-vitl16-...` (1.1 GB) og
  `briaai/RMBG-2.0` (844 MB). Vektene er bf16/fp16 → kastes til f32 ved
  lasting (v1 er f32); last én modell om gangen (3× 1.3B i f32 ≈ 16 GB).
- **Python-miljø**: `trellis-mac/.venv` (Python 3.11) har safetensors,
  torch 2.12.1, transformers 5.12.1 (DINOv3), trimesh, utils3d, xatlas,
  flex_gemm (Metal-conv). pixi-miljøet her har nå safetensors som
  eksplisitt avhengighet (steg 2); transformers/trimesh mangler fortsatt —
  for steg 3/4: legg til i pixi.toml eller kjør via trellis-mac-venvet.
- **Mesh-ekstraksjon**: `trellis-mac/stubs/o_voxel_override_convert.py` er
  stubben fdg_vae.py leter etter — ren Python/torch
  `flexible_dual_grid_to_mesh` («identical output to the CUDA version for
  inference»). Det finnes også en pakke-stub `trellis-mac/stubs/o_voxel/`
  (convert/io/rasterize). Kopiér eller legg på sys.path.
- **Bildekondisjonering**: pipeline.json → `DinoV3FeatureExtractor`
  (facebook/dinov3-vitl16) + `BiRefNet` (RMBG-2.0) — kjøres i Python via
  interop (transformers ligger i trellis-mac-venvet).
- **✅ Sampler-gap LUKKET (guidance_rescale portert)**: pipeline.json
  bruker `guidance_rescale` (ss: 0.7, shape-slat: 0.5, tex: 0.0/strength
  1.0 → irrelevant der), som porten opprinnelig hoppet over.
  Nå portert i `flow_euler.mojo` (`_rescale_pred` + valgfrie
  `guidance_rescale`/`seg_offsets`-params) med BEGGE oppstrøms
  std-semantikker: dense = torch `.std()` (unbiased, per dim-0-rad),
  varlen = VarLenTensor.std (biased `sqrt(E[x²]−E[x]²)`, per segment over
  tokens×kanaler — aktiveres av seg_offsets). Wiret gjennom
  `sample_sparse_structure`/`sample_slat` (slat utleder offsets fra
  coords). Paritetsverifisert: 2 nye caser i flow-euler-testen (dense +
  varlen med ujevne segmenter, ekte params) og HELE wp9-integrasjonen
  kjører nå med rescale 0.7/0.5. Sampler-params ellers: steps 12,
  strength 7.5, interval [0.6,1.0], rescale_t 5.0 (ss) / 3.0 (slat);
  tex: strength 1.0, interval [0.6,0.9], rescale_t 3.0.
- **Referanse-implementasjon**: `trellis-mac/generate.py` viser hele
  flyten (env-oppsett, backend-valg, pipeline-kall). `trellis-mac/
  TRELLIS.2/` er upstream @75fbf01 med 8 filer Mac-patchet — MERK: deres
  sdpa-patch i sparse full_attn null-padder uten maske (samme stille bug
  som funn 1 her; ufarlig i praksis fordi CFG-grenene kjøres separat med
  B=1). `trellis2/`-kopien i DETTE repoet er fasiten — MEN
  IKKE helt uberørt (oppdaget 2026-07-13 under GitHub-pakkingen): NI
  filer bærer CPU/MPS-kompatibilitets-patcher (CPU-conv/attention-
  backends inkl. tilføyde conv_none.py, ingen harde .cuda()-kall,
  rembg-guard, mesh-stub-hook). Treet er IKKE lenger vendored i
  git — `scripts/fetch_upstream.sh` henter oppstrøms @75fbf01 og
  applikerer `scripts/upstream_mac_compat.patch` (verifisert:
  hentet+patchet er byte-identisk med fasiten suiten er grønn mot).
  Bruk den som fasit, ikke trellis-mac-kopien.
- Runner-retning: Mojo-vert (jf. ADR 0001-hybrid og PyVelocityModel-
  presedensen; Python→Mojo-binding er upraktisk i 1.0.0b2).

Rekkefølge for WP9 del 3 (REVIDERT 2026-07-09 av ADR 0007 — sluttmålet er
ren Mojo-inferens, se `docs/decisions/0007-pure-mojo-inference.md` og
WP12–WP14 i masterplanen):
(1) ✅ guidance_rescale portert + paritetstestet (2026-07-08);
(2) ✅ safetensors→f32-lasting (2026-07-09, se under);
(3) ✅ DINOv3-kondisjonering via interop (2026-07-09): `pipelines/
    conditioning.mojo` (`ImageConditioner.get_cond(path, res)`) +
    `cond_io.py`. Paritet `pixi run test-cond`: BIT-IDENTISK (max|diff|
    0.0) mot originalens DinoV3FeatureExtractor på 128 og 512, preprocess
    pikselidentisk mot pipeline.preprocess_image (alpha-grenen kalles med
    dummy-self — den rører aldri self), RGB/heldekkende-alfa avvises.
    rembg/BiRefNet droppet permanent (ADR 0007): runneren krever RGBA med
    ekte alfakanal — oppstrøms preprocess_image hopper da over rembg
    (trellis2_image_to_3d.py:127). transformers 5.2.0 lå alt i pixi-env;
    torchvision lagt til (brukes KUN av testreferansen — cond_io
    normaliserer manuelt). Merk cascade: cond trengs på både 512
    (1029 tokens) og 1024 (4101 tokens); '512'-pipeline klarer seg med
    512.
(4) ✅ mesh-ekstraksjon i REN Mojo (2026-07-09):
    `trellis2_mojo/meshing/fdg_mesh.mojo` — port av trellis-mac-stubben
    (146 linjer) + `write_obj`. Paritet `pixi run test-mesh` (I test-all):
    3 seeds × begge split-moduser mot vendored stub-kopi
    (`tests/parity/o_voxel_stub.py`, med dokumentert VENDORED-FIX:
    torch.cross får eksplisitt dim=1 — implisitt default velger første
    3-dim og flipper semantikk ved nøyaktig 3 quads), triangler eksakte,
    vertices ≤1.2e-7 (f32-avrunding: stubben regner i f32, porten i f64 →
    f32), degenerater (0 quads → 0 vertices, stubbens early-out) og
    OBJ-roundtrip (Python leser tilbake og sammenligner). Ingen
    to_glb/xatlas (ADR 0007 — selv trellis-mac eksporterer OBJ uten den
    stien);
(5) ✅ ende-til-ende-runner (2026-07-09): `run_image_to_3d.mojo` i repo-
    rot, `pixi run e2e` — se tabellen over og «Steg 5 utført» under.

Deretter ren-Mojo-sporet (WP12–WP14 i 06_MASTER_PLAN.md, med runneren som
regresjonsharness): ✅ WP12, ✅ WP13, ✅ WP14 — ALLE FERDIG 2026-07-09 (se
egne seksjoner under). Sluttilstanden fra ADR 0007 er nådd: Python/torch
kun i tests/parity og benchmarks (+ torch.randn i runneren for
støy-strøm-kompatibilitet).

### ✅ Steg 2 utført: safetensors→f32-lasting (2026-07-09)

- `trellis2_mojo/ckpt_io.py`: HF-cache-oppslag uten nettverk (glob på
  snapshots), `load_state_dict_f32` (safetensors bf16/fp16 → kontiguøs
  f32, cast ved lasting), `load_config`/`pipeline_config`/`model_path`
  (pipeline.json-navn, inkl. kryss-repo-referansen til TRELLIS-image-large
  for ss_dec), + små config-tolker (`is_rope`, `pred_subdiv`).
  `safetensors` er nå eksplisitt pixi-avhengighet.
- `trellis2_mojo/checkpoints.mojo`: `load_sparse_structure_flow()`,
  `load_sparse_structure_decoder()`, `load_slat_flow(key)`,
  `load_unet_decoder(key)` — leser config-JSON og kaller `*_from`-loaderne.
  Shape-dekoderens config er en FlexiDualGridVaeDecoder: out_channels=7 og
  pred_subdiv=True er innbakt i subklassen (ingen egne vekter); ss_dec-
  configen har ingen norm_type → «layer».
- `interop.mojo` omskrevet til peker-kopier (se tabellen) — uten dette tar
  én DiT titalls minutter å konvertere. `UnsafePointer[..., MutAnyOrigin]
  (unsafe_from_address=Int(py=t.data_ptr()))` er 1.0.0b2-formen; husk å
  holde det midlertidige torch-objektet i live gjennom kopien (`_ = Int(
  py=tt.numel())` etterpå — ASAP-destruksjon).
- Paritet: `pixi run test-real` (`torch_ref_real.py`/`parity_real_ckpt_vs_
  torch.mojo`) — torch-siden laster originalmodellene fra SAMME filer
  uavhengig (strict=False som oppstrøms from_pretrained) og kjører seedede
  forwards; Mojo-siden går via checkpoints.mojo. Dekker ss_dec, shape/tex-
  unet-dekoderne (inkl. guided-handover med torch-subs som guide) og alle
  tre 512-DiT-ene med ekte former (ss_flow på 16³=4096 tokens). 1024-slat-
  variantene hoppes over: identisk arkitektur/nøkler som 512, bare andre
  vekter. Observert max|diff| 1e-5–3.5e-4 (atol 5e-4–2e-3 per modell;
  ss_dec-logits er O(200) → atol 1e-3). torch-referansen holder én modell
  om gangen i minnet (~5.3 GB f32 per DiT).

### ✅ Steg 5 utført: ende-til-ende-runner (2026-07-09)

- `run_image_to_3d.mojo` (repo-rot) + pixi-task `e2e`. Speiler
  `Trellis2ImageTo3DPipeline.run(pipeline_type='512')`: sampler-params,
  sigma_min og slat-normalisering leses fra pipeline.json; modellene
  lastes én om gangen i stage-funksjoner (peak ~6 GB RSS — DiT-en frigis
  før neste lastes). Eneste Python i stien (ADR 0007, byttes av
  WP12/WP13): ckpt_io (safetensors/config), cond_io (DINOv3) og
  torch.randn for støy.
- **RNG-avgjørelse**: oppstrøms seeder FØR get_cond, men kondisjonering
  konsumerer ingen RNG ved inferens — bare modell-KONSTRUKSJON trekker
  fra generatoren, og oppstrøms konstruerer ekstraktoren i
  from_pretrained (før seed). Runneren beregner derfor cond først og
  seeder etterpå → trekkerekkefølgen randn(1,8,16³) → randn(N,32) →
  randn(N,32) blir identisk med originalen/trellis-mac, og ss-støyen
  bit-identisk (randn trekkes på CPU også i MPS-kjøringer).
- Mesh: `decode_shape` → koordinater strippes for batch-kolonnen →
  `flexible_dual_grid_to_mesh` (split_weight-stien, aabb [-0.5,0.5]³,
  grid 512) → `write_obj`. Tekstur: `decode_tex` → npz (coords int32,
  attrs f32 [T,6] i pbr-layout base_color/metallic/roughness/alpha,
  origin, voxel_size) — payloaden MeshWithVoxel bærer til baking
  (to_glb/xatlas er utenfor scope per ADR 0007).
- Smoke (steps 2, --no-tex, shoe_3q.png seed 42): grønn hele veien —
  2369 voxels @ 32³ → 948 575 voxels @ 512³ (dekoding 39 s!) → 948 575 V
  / 2 055 480 F på ~10.5 min totalt. trimesh-validering: 0 degenererte
  flater, samme geometri-klasse som referansen (ikke-vanntett FDG).
- **Golden light-verifisert** (shoe_3q.png, seed 42, 12 steg, full
  kjøring med tekstur — 2026-07-09). Mojo (CPU/f32): 1857 voxels @ 32³ →
  514 603 voxels @ 512³ → **514 603 V / 1 055 560 F**, 27.4 min totalt
  (ss-stadiet 20 min = ~17 CFG-forwards à ~2 min på 4096 tokens dense;
  slat-stadiene 4.2/2.5 min; shape-dekoding 20 s; mesh ~0 s). trellis-mac
  (MPS/bf16, `--pipeline-type 512 --no-texture`): **510 511 V /
  1 040 818 F** på 187.7 s. Avvik: V 0.8 %, F 1.4 %; bbox-hjørner matcher
  innenfor 2e-3; nærmeste-nabo-avstand mellom meshene (begge veier,
  cKDTree på vertices) mean 2.1e-3 ≈ 1.1 voxel, p95 6.8e-3, max 2.1e-2;
  0 degenererte flater begge. Tex-npz: 514 603 × 6 attrs i [-0.01, 1.00].
  MERK: MPS/bf16 vs CPU/f32 flipper terskler (occupancy, subdivision), så
  V/F-antall matcher ALDRI eksakt — sammenlign størrelsesorden, bbox og
  avstandsmetrikker, ikke tall. Artefaktene ligger i `outputs/`
  (shoe_3q_mojo_seed42.obj + _texvoxels.npz + mac-referansen).
  Langvarige e2e-kjøringer: start med nohup/disown — bakgrunnsjobber dør
  ellers med agent-prosessen.

## WP12 — safetensors + config-JSON i ren Mojo (FERDIG 2026-07-09)

Hele lastestien er nå ren Mojo: `trellis2_mojo/io/` (json/safetensors/
hf_cache/state_dict — se tabellen) erstatter ckpt_io.py i runner-stien.
(Etter WP13+WP14 er eneste gjenværende Python i runneren torch.randn —
støy-strøm-kompatibilitet med originalen.)

- **Refaktor-grepet**: alle loadere (`loaders.mojo` + `*_from` i
  modellfilene) tar nå `StateDict` i stedet for PythonObject. Fasaden har
  `@implicit`-ctor fra PythonObject, så paritetstestene sender torch-
  state-dicts UENDRET (null test-endringer); runner-stien sender Mojo-
  dicten fra safetensors-leseren. `sd[key]`-oppslag ble `sd.tensor(key)`.
- **macOS-fella**: read() capper på 2 GiB → DiT-filene (2.7 GB) kan ikke
  slurpes. Leseren går per-tensor i offset-rekkefølge (sekvensiell IO,
  `stable_argsort` på data_offsets), validerer kontiguitet, og bruker
  alignment=1-SIMD-laster (data-seksjonen er ualignert).
- **Paritet**: `pixi run test-io` (IKKE i test-all — leser ~14 GB):
  alle 8 sjekkpunktene BIT-identiske mot ckpt_io.py (int32-views,
  7.48 mrd. verdier totalt), model_path/ckpt_base-stier like,
  pipeline.json-flyttallene (sampler-params + 128 normaliseringsverdier)
  eksakt like Pythons json (Int-mantisse + eksakt 10-potens er korrekt
  avrundet for ≤18 sifre), JSON-enhetstester (escapes/surrogatpar/tall/
  nesting). `pixi run test-real` grønn med Mojo-leseren i lastestien
  (samme driftnivåer som før: 1e-5–3.5e-4); hele test-all grønn; runner-
  smoke (steps 2, samme seed) etter byttet ga BIT-identisk OBJ med
  før-WP12-smoken (`cmp` byte-for-byte) — lastingen er eksakt drop-in.
- JSON-parseren er arena-basert (JsonDoc med parallelle per-node-lister,
  noder som indekser) — ingen rekursiv verditype (Mojo-structs kan ikke
  inneholde seg selv uten heap-indireksjon).

## WP13 — DINOv3 ViT-L/16 i ren Mojo (FERDIG 2026-07-09)

Kondisjoneringen kjører nå i ren Mojo: `trellis2_mojo/models/dinov3.mojo`
(se tabellen for arkitekturdetaljene) + `checkpoints.mojo::load_dinov3()`.
`ImageConditioner.get_cond`-signaturen er uendret; cond_io.py er slanket
til preprocess + `pixels()` (transformers er ute av runner-stien —
gjenstår kun for paritetsreferansene).

- **Avvik fra WP-planteksten (verifisert mot transformers-kilden)**:
  `pos_embed_rescale=2.0` er KUN treningsaugmentering — transformers gater
  augment_patches_center_coordinates på `self.training`, så eval bruker
  rå patch-senterkoordinater. Og k-bias-problemet forsvant av seg selv:
  transformers har separate q/k/v-projeksjoner, så porten bruker tre
  `linear`-kall (ingen qkv-fusjonering, ingen nullpadding).
- **Rope-fella**: DINOv3 bruker rotate_half-splitt (NeoX: par (i, i+D/2))
  — IKKE parvis interleave som flow-modellenes rope. Vinkelraden er
  [y·f0..y·f_{q-1}, x·f0..x·f_{q-1}] (q = head_dim/4) og tile(2) gjør at
  lane i og i+D/2 deler vinkel i. Kun patch-tokens roteres (cls + 4
  registre passerer urørt). All fasematte i f32 som originalen.
- **Sluttnorm-fella**: ekstraktoren (og cond_io før WP13) kjører manuell
  lag-løkke + `F.layer_norm` UTEN affine — modellens egen `norm.weight/
  bias` ligger i sjekkpunktet men brukes ALDRI. Porten leser dem ikke.
- **Paritet**: `pixi run test-wp13` (I test-all — liten random-config
  [2 lag, 64 ch, 4 hoder, 3 registre] bygget rett fra transformers-config
  med perturberte params siden _init_weights nuller biases; extractor-stil
  forward; max|diff| ~1.7e-6, atol 2e-5; inkl. ikke-kvadratisk 3×5-grid
  som fanger y/x-transponering). `pixi run test-cond` er nå ekte-vekt-
  referansen: max|diff| 3.4e-5 på både 128 og 512 (atol 5e-4, samme trapp
  som test-real). Merk `model.config._attn_implementation = "eager"` i
  ref-filen.
- **Runner-smoke** (steps 2, --no-tex, shoe_3q.png seed 42, 2026-07-09,
  `outputs/wp13_smoke.obj`): cond-stadiet [1, 1029, 1024] tar ~2 s i ren
  Mojo. 2369 voxels @ 32³ (EKSAKT samme antall som før-WP13-smoken) →
  948 568 voxels @ 512³ → 948 568 V / 2 055 470 F på 627 s. Før WP13:
  948 575 / 2 055 480 — avvik 7 voxels av ~950k (cond-drift 3.4e-5
  flipper borderline-terskler i shape-dekoderen; jf. MERK-en om at V/F
  aldri matcher eksakt på tvers av numerikk-varianter). Strukturelt
  uendret: identisk 32³-okkupans, samme geometri-klasse.

## WP14 — bilde-IO + preprocess i ren Mojo (FERDIG 2026-07-09)

Siste Python-biten ut av runneren: `io/image.mojo` (PAM P7/PPM P6) +
`imaging/{resize,preprocess}.mojo` (se tabellen). Runneren tar nå PAM
P7-filer; PNG→PAM er en dokumentert PIL-énlinjer i README_MOJO.md (som
også fikk et komplett kjøreeksempel — READMEen var forøvrig helt utdatert
og er omskrevet). PNG-dekoder i ren Mojo forble bevisst ugjort.

- **Alt er BIT-eksakt mot PIL** (`pixi run test-wp14`, I test-all):
  rasterlesere, 5 resize-caser (ned/opp/ikke-kvadratisk/én-akse/no-op,
  RGB+RGBA), preprocess på 640 (uten nedskalering) og 1500 (>1024-stien),
  cond_pixels på 128/512 (f32 max|diff| 0.0), + avvisning av RGB/heldekkende
  alfa. Dermed er `pixi run test-cond` sitt driftnivå UENDRET etter byttet
  (3.147e-5/3.386e-5 på 128/512 — nøyaktig samme tall som med PIL-stien),
  og WP14-smoken ga BIT-identisk OBJ med wp13-smoken (`cmp`): se under.
- **PIL-feller funnet (alle verifisert empirisk mot Pillow 12.3)**:
  (1) RGBA-resize går via RGBa — MULDIV255-premultiply (+128-bias),
  resample av premultipliserte bånd, tilbake med TRUNKERENDE divisjon
  (255·v/α, α∈{0,255} passthrough). Uten rundturen bommer ~70 % av
  pikslene. (2) resize til samme størrelse kortslutter til copy() FØR
  RGBa-rundturen. (3) crop-boksen avrundes med Pythons round-half-even
  (PIL `map(int, map(round, box))`) — relevant fordi bbox-senteret kan
  ligge på .5. (4) Fixed-point-numerikken (PRECISION_BITS=22, clip8,
  u8-kvantisering MELLOM passene) gjør at en flyttallsimplementasjon
  ALDRI kan matche PIL bit-eksakt — porten speiler heltallsstien.
- **Interpass-detalj**: Pillow trimmer temp-bildet til radene vertikalpasset
  leser (ybox_first/last) — ren arbeidsbesparelse; porten prosesserer alle
  rader og får identisk resultat. Koeffisientsummene holder seg < 2³¹, så
  64-bits Int matcher Pillows INT32.
- **Runner-smoke etter WP14** (steps 2, --no-tex, /tmp/shoe_3q.pam seed
  42): `outputs/wp14_smoke.obj` er BYTE-IDENTISK med wp13-smoken
  (`cmp` — PAM-dekoding+Mojo-preprocess ga eksakt samme pikselbuffer som
  PIL-stien, og resten av kjeden er deterministisk).

## Kjente begrensninger (bevisste, dokumentert i tracker)

- CPU + float32 only i v1 (dtype er strukturparameter; fp16/GPU er WP11+).
- Naiv ytelse overalt — korrekthet først, per masterplanen.
- Trening, texturing-pipeline, renderers, elastic mixin: utenfor scope, se
  tracker. o-voxel: mesh-ekstraksjonsstien er portert til ren Mojo (ADR
  0007); resten (postprocess/rasterize, QEF/CUDA) er utenfor scope.
- WP0 (golden outputs fra ekte modell) krever CUDA-maskin — eneste reelle
  eksterne avhengighet for slutt-verifisering av WP8/WP9.
