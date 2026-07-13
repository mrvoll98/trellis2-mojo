# ADR 0008: Fase 2 — teksturert eksport via vertex-attributter (ikke UV-baking)

**Skrevet:** 2026-07-11 (WP11 steg 2–15 ferdig, GPU-køen tom; fase 2 startet
på brukerens beslutning).

**Kontekst:** v1 (ADR 0007) leverer OBJ + tekstur-voxels-npz. Oppstrøms
postprosess (`o_voxel/postprocess.py::to_glb`) gjør mesh-en om til en
teksturert GLB, men HELE stien er CUDA-bundet og finnes ikke på denne
maskinen: `cumesh` (hullfylling, decimering til 1M, non-manifold-reparasjon,
UV-chart-unwrap, cuBVH, vertex-normaler), `nvdiffrast` (UV-roms-rasterisering
for baking), `flex_gemm.grid_sample_3d` (CUDA-hashmap) og `cv2.inpaint`
(sømfylling). Selv den fungerende trellis-mac-MPS-porten hopper over stien
(generate.py fanger at stubben mangler `.postprocess`).

**Beslutning:** Fase 2 v1 eksporterer en teksturert, direkte viewbar **GLB
med vertex-attributter** i ren Mojo, i stedet for å portere
UV-baking-pipelinen:

1. `grid_sample_3d`-trilineær-sampling porteres til ren Mojo med EKSAKT
   flex_gemm-semantikk (8 naboer via TRUNKERING av p±0.5 — ikke floor;
   vekt = prod(1 − |nabo + 0.5 − p|); manglende voxels vekt 0;
   renormalisering med clamp_min(1e-12)) og brukes til å sample
   PBR-attributtvolumet på mesh-VERTEKSENE.
2. En ren-Mojo GLB 2.0-writer (`io/glb.mojo`) skriver POSITION / NORMAL /
   COLOR_0 / indices + ett pbrMetallicRoughness-materiale.
   Oppstrøms akse-konvensjon beholdes (y,z → z,−y ved eksport).
   base_color+alpha rir i COLOR_0 (multipliseres med baseColorFactor=1 av
   glTF-spesifikasjonen); metallic/roughness har ingen standard
   per-vertex-kanal og eksporteres som GLOBALE faktorer (gjennomsnitt av de
   samplede verdiene). Full per-voxel-PBR ligger fortsatt i npz-en.
3. OBJ + npz beholdes uendret; GLB-en skrives i tillegg når tekstur er på.

**Hvorfor dette er nesten tapsfritt:** oppstrøms baking eksisterer for å
overleve DECIMERING (514k→~naboskapet av 1M er allerede under målet, men
generelt ned mot 1M flater fra flere millioner) — teksturen bevarer
voxel-detaljen når geometrien forenkles. Vår mesh decimeres IKKE:
FDG-verteksene ligger per konstruksjon på voxel-oppløsningen (512³), så
per-vertex-sampling bærer samme informasjonsmengde som en 2048²-baking av
samme volum (4.2M texels mot 514k vertekser ved ~1 sample per
overflate-voxel begge veier; bakingens ekstra texels er interpolasjon).

**Bevisste avgrensninger (fortsatt utenfor scope, dokumentert her):**
- Decimering/remeshing/non-manifold-reparasjon (cumesh) — krever en hel
  mesh-prosesseringsbibliotek-port; mesh-en brukes som FDG-en leverer den
  (0 degenererte flater i golden-kjøringene).
- ~~Hullfylling~~ **TILLEGG 2026-07-11 (WP16, revidert samme dag):**
  hullfylling er LØFTET INN i scope etter brukerobservasjon av synlige
  mikrohull (de finnes også i trellis-mac-outputen — FDG er
  ikke-vanntett per konstruksjon, og oppstrøms fyller i
  cumesh-postprosessen). Første versjon antok «hull = ren sykel» og
  kjedet randkanter — men brukeren så fortsatt hull: FLETTEDE klynger
  ved non-manifold-kryss forble åpne. CuMesh-kilden viste seg å være
  PUBLISERT (github.com/JeffreyXiang/CuMesh), og
  `meshing/postprocess.mojo` porterer nå dens faktiske formulering:
  union-find-KOMPONENTER av randkanter, avvis kun komponenter med
  grad-1-vertekser (blindveier), fyll hele komponenten med én centroid
  (snitt av kant-midtpunkter) + (b, a, c)-trekant per randkant. Kun i
  GLB-eksporten (OBJ/npz forblir rå som oppstrøms MeshWithVoxel).
  Gjenværende randkanter er blindvei-søm-stier som heller ikke cumesh
  fyller.
- UV-unwrap (xatlas/cumesh-charts) + rasterisert baking + inpainting — kun
  meningsfullt SAMMEN med decimering, jf. over.
- `alphaMode` settes OPAQUE som oppstrøms gjør for image_to_3d-stien.

**Verifisering:** paritetstest mot en ren-torch-reimplementasjon av
flex_gemm-formelen (CUDA-hashmapen erstattes av dict-oppslag — samme matte,
inkl. trunkerings-quirken der p<0.5 gir duplikat-naboer som begge teller);
GLB-roundtrip via trimesh (les tilbake, sammenlign posisjoner/indekser/
farger/normaler bit-for-bit mot det som ble skrevet); golden-kjøring med
tekstur der GLB-ens COLOR_0 sammenlignes mot referanse-sampling av npz-en
på de samme verteksene.
