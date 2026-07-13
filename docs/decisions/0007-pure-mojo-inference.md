# ADR 0007: Ren Mojo-inferens som sluttmål

**Skrevet:** 2026-07-09 (WP9 del 3 steg 2 ferdig).

**Beslutning:** ADR 0001s «Full pure later» konkretiseres nå: sluttmålet er en
runner der HELE inferens-stien er Mojo — Python/torch beholdes kun i
paritetstestene (som fasit) og benchmarks. Hybrid-glue som finnes i dag
(`ckpt_io.py`) og som kommer i steg 3 (DINOv3 via interop) er midlertidige
stillaser som byttes ut komponent for komponent, med den fungerende runneren
som regresjonsharness.

**Rekkefølge (begrunnelse: korrekthet-først, baseline før utbytting):**
1. WP9 del 3 fullføres FØRST som hybrid (steg 3 kondisjonering via interop,
   steg 5 runner) — raskeste vei til ende-til-ende-baseline mot trellis-mac.
   Unntak: steg 4 (mesh) gjøres direkte i ren Mojo — stubben er 146 linjer
   ren torch, det er dobbeltarbeid å wire den via Python først.
2. Deretter ren-Mojo-sporet WP12→WP14 (se 06_MASTER_PLAN.md): safetensors/
   JSON-lasting (WP12), DINOv3-port (WP13), bilde-IO + preprocess (WP14).
   Hver utbytting verifiseres mot Python-stillaset den erstatter (bit-identisk
   der det er mulig) + `test-real`/runner-baseline grønn.

**Bevisste avgrensninger:**
- **BiRefNet/RMBG porteres IKKE.** Oppstrøms `preprocess_image` hopper over
  rembg når input er RGBA med ekte alfakanal (verifisert i
  `trellis2/pipelines/trellis2_image_to_3d.py:127`) — runneren krever RGBA-
  input med utklippet objekt. Bakgrunnsfjerning er preprosessering, ikke
  3D-generering.
- **xatlas/UV-unwrap/tekstur-baking utenfor scope.** Selv den fungerende
  trellis-mac-referansen eksporterer OBJ uten `o_voxel.postprocess.to_glb`-
  stien (generate.py fanger at stubben mangler .postprocess). v1 eksporterer
  geometri (+ vertex-farger fra tekstur-slat via voxel-oppslag hvis det viser
  seg enkelt — avklares i steg 4/5).
- **PNG-dekoding i ren Mojo er et eget beslutningspunkt** (krever zlib
  inflate, ~400 linjer + filtre). WP14 starter med PAM/PPM-input (P7 har
  alfa) + dokumentert konverteringskommando; PNG-dekoder tas kun hvis
  friksjonen viser seg å være reell.

**Konsekvens:** `modules/image_feature_extractor.py` «forblir Python» fra
WP8-planen og «pipelines forblir Python» fra ADR 0001 er opphevet for
inferens-stien. ADR 0004 («keep o-voxel as FFI») overstyres delvis:
`flexible_dual_grid_to_mesh`-inferensstien porteres til Mojo (stubben i
trellis-mac beviser at ren-tensor-implementasjonen er ekvivalent for
inferens); resten av o_voxel (postprocess/rasterize, QEF/CUDA) forblir
utenfor scope som før. Trening, renderers og texturing-pipelinen er fortsatt utenfor
scope (ADR 0001/tracker).
