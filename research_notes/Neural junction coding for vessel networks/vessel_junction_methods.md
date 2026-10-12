# Vessel junction detection, crossing-vs-bifurcation classification and topology recovery in 2-D vascular images

Scope note: these notes cover 2-D projection imaging (colour fundus, SLO, fluorescein/X-ray angiography, microscopy and microscopy video). Each method record gives: modality/dataset, how junctions are found, how crossing vs bifurcation is decided, how connectivity/topology is recovered, numbers, learned (L) vs hand-designed (H), and a citation. Numbers from different papers are almost never directly comparable. Papers use different inputs (manual binary segmentation vs raw image), different ground truth (each group's own graders), different tolerances (5 px vs 10 px) and different metrics (detection F1 vs type accuracy vs A/V accuracy). The caveats are given inline.

User context for comparison: the pipeline gets ~0.75–0.80 junction-detection F1, ~90% bifurcation accuracy and ~50–67% crossover accuracy on synthetic data, and the crossover score plateaus even as noise goes to zero.

---

## 1. Biologically inspired junction methods (COSFIRE; Favali/Citti/Sarti cortical spectral clustering; Abbasi-Sureshjani/Bekkers/Duits/ter Haar Romeny orientation scores)

### Takeaway
Cortex-inspired methods have made strong claims, but most report only limited quantitative junction results:
- **COSFIRE** detects bifurcations very well on binary segmentations (~97–98.5% recall at ~96–97% precision), but it was designed for and evaluated on bifurcations, not on crossing-vs-bifurcation classification.
- **Orientation-score methods** (BICROS, ETOS tracking, crossing-preserving vesselness) separate crossing vessels into different orientation layers. BICROS reaches only ~0.61–0.67 junction-detection F1 against independent ground truth.
- **Fokker–Planck/V1-kernel spectral clustering** (Favali 2016; Abbasi-Sureshjani 2018) is validated mostly qualitatively, on synthetic images and retinal patches.

### Cited Findings

**COSFIRE (Azzopardi & Petkov) — H (configured from prototypes, not trained by gradient descent)**
- Modality/data: binary (segmented) fundus images from DRIVE and STARE. Junction finding: COSFIRE keypoint detectors are configured on prototype bifurcations and then find similar patterns in segmented retinal images. Results: recall 97.88% at precision 96.94% on the 40 manually segmented DRIVE images; recall 97.02% / precision 96.53% on 10 automatically segmented DRIVE images; recall 97.32% / precision 96.04% on 20 manually segmented STARE images. The abstract motivates identifying "bifurcations and crossovers", but the method and results concern bifurcations; no crossover results are reported. Citation: G. Azzopardi, N. Petkov (2013), "Automatic detection of vascular bifurcations in segmented retinal images using trainable COSFIRE filters," *Pattern Recognition Letters* 34(8):922–933, doi:10.1016/j.patrec.2012.11.002 — [Azzopardi publication list (abstract)](https://www.cs.rug.nl/~george/my-publications/?tgid=44); [OpenAlex/Semantic Scholar metadata](https://api.openalex.org/works/doi:10.1016/j.patrec.2012.11.002)
- Precursor (CBMS 2012): a filter configured from a single example bifurcation, built from selected Gabor channels and combined by a geometric mean of blurred, shifted responses, with rotation, scale and reflection invariance. Five filters gave ≈98.57% recall and 95.37% precision on the 40 DRIVE binary images, which contain >5000 bifurcations — [UM repository](https://um.edu.mt/library/oar/handle/123456789/26336); [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/26336/1/ConferencePaper_Detection%20of%20retinal%20vascular%20bifurcations%20by%20rotation-_scale-_and%20reflection-invariant%20COSFIRE%20filters.pdf)
- TPAMI version ("Trainable COSFIRE filters for keypoint detection and pattern recognition"): 98.50% recall and 96.09% precision for retinal bifurcations on DRIVE; crossovers not mentioned — [Azzopardi publication list](https://www.cs.rug.nl/~george/my-publications/?tgid=44)
- Inhibition-augmented COSFIRE (Guo et al., *Machine Vision and Applications* 2016, doi:10.1007/s00138-016-0781-7). It adds inhibitory prototypes for "the exclusive detection of vascular bifurcations (i.e., without crossovers)" and uses a T-junction vs crossover example as motivation. The listing gives no recall/precision numbers. This is the closest COSFIRE-family work to crossing-vs-bifurcation discrimination: it relies on push-pull inhibition, where the response to a crossing configuration suppresses the bifurcation detector — [Azzopardi publication list](https://www.cs.rug.nl/~george/my-publications/?tgid=44); [Springer link](https://link.springer.com/article/10.1007/s00138-016-0781-7)
- Hamad et al. (2014) note that Azzopardi et al.'s Gabor/COSFIRE landmark method "did not proceed to distinguish between bifurcations and crossovers" — [Hamad et al. 2014, Image Anal Stereol](https://ias-iss.org/ojs/IAS/article/view/1101)

**Favali, Abbasi-Sureshjani, ter Haar Romeny, Sarti (2016) — cortically inspired spectral clustering — H (fixed kernel)**
- Modality: colour fundus patches. Method: a connectivity kernel, the fundamental solution of the Fokker–Planck equation (the "usual" model of multi-orientation V1 horizontal connectivity), is combined with an intensity feature into a large local affinity matrix. Spectral clustering then splits vessels into a hierarchy of "individual perceptual units". There is no centerline extraction. The method exploits continuity of orientation and intensity of the artery and of the vein across an arteriovenous crossing. Quantitative results: none in the abstract; it claims vessels are "identified successfully". Citation: M. Favali, S. Abbasi-Sureshjani, B. M. ter Haar Romeny, A. Sarti (2016), "Analysis of Vessel Connectivities in Retinal Images by Cortically Inspired Spectral Clustering," *J Math Imaging Vis* 56(1):158–172, doi:10.1007/s10851-016-0640-1, arXiv:1512.06559 — [arXiv](https://arxiv.org/abs/1512.06559); [search summary of arXiv PDF](https://arxiv.org/pdf/1512.06559)

**Abbasi-Sureshjani, Favali, Citti, Sarti, ter Haar Romeny (2018) — 5-D curvature kernel — H**
- Lifts the image to multi-orientation scores plus a contextual affinity matrix. Connectivity is a 5-D kernel, the Fokker–Planck fundamental solution in the space of position × orientation × curvature × intensity, followed by self-tuning spectral clustering. Validated on artificial images and real retinal patches. It is stated to solve grouping "also for blood vessels with high curvature", which earlier approaches could not. The stated motivation: segmentation and tracking still fail "when two or more curvilinear structures cross or bifurcate", at interruptions, and for highly curved vessels. No junction-level F1/accuracy was found in the abstract. Citation: *IEEE Trans Image Process* 27(2):606–621 (2018), doi:10.1109/TIP.2017.2761543, arXiv:1608.08049 — [arXiv](https://arxiv.org/abs/1608.08049v3); [TU/e record](https://research.tue.nl/en/publications/curvature-integration-in-a-5d-kernel-for-extracting-vessel-connec/)

**Abbasi-Sureshjani, Zhang, Duits, ter Haar Romeny (2017) — learned V1-like co-occurrence kernel — L (statistics) / H (model form)**
- Line co-occurrence statistics measured from retinal vessel centerlines closely match the probabilistic V1 connectivity model. The best fit, a symmetrized direction process on the projective line bundle, has least-squares error <2%. Statistics from different datasets are highly similar ("independent of the dataset"). The trained kernel was used in spectral clustering to group interrupted vessels around junctions, with qualitative "successful results". Citation: "Retrieving challenging vessel connections in retinal images by line co-occurrence statistics," *Biological Cybernetics* (2017), doi:10.1007/s00422-017-0718-x — [Semantic Scholar record/abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s00422-017-0718-x?fields=title,abstract); [Springer](https://link.springer.com/article/10.1007/s00422-017-0718-x)

**BICROS — Abbasi-Sureshjani, Smit-Ockeloen, Bekkers, Dashtbozorg, ter Haar Romeny (ISBI 2016) — hybrid: H orientation-score lifting + L supervised junction classifier**
- Method: lifts the 2-D image to SE(2) (positions × orientations) so "vessels disentangle based on their orientations". Junctions are then discriminated on their geometric properties "in a supervised manner". It is described as inspired by V1 pinwheels — [Abbasi profile (thesis summary)](https://manet.dm.unibo.it/fellow.php?f=abbasi)
- Citation: "Automatic detection of vascular bifurcations and crossings in retinal images using orientation scores," ISBI 2016, pp. 189–192, doi:10.1109/ISBI.2016.7493241 — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ISBI.2016.7493241); [DBLP](https://dblp1.uni-trier.de/pid/164/7671.html)
- Independent re-evaluation by Uslu & Bharath (2018), junction detection (P/R/F1, 5-px tolerance). DRIVE: 0.75 / 0.61 / 0.67 (vessels segmented by Soares) and 0.75 / 0.61 / 0.66 (vessels segmented by Abbasi-Sureshjani). IOSTAR: 0.47 / 0.60 / 0.52 (Soares) and 0.67 / 0.57 / 0.61 (Abbasi-Sureshjani) — [Uslu & Bharath, arXiv:1806.03175](https://ar5iv.labs.arxiv.org/html/1806.03175)
- I could not retrieve BICROS's own reported bifurcation-vs-crossing classification accuracy (see Gaps).

**Bekkers, Duits, Berendschot, ter Haar Romeny (2014) — ETOS tracking in invertible orientation scores — H**
- Method: tracking runs in invertible (cake-wavelet) orientation scores, as a curve optimization in SE(2), with left and right edges tracked as a pair. At a crossing the score separates the two vessels into different orientation layers, and the tracker follows the layer matching its current direction. A faint edge at a crossing can still be followed through its partner edge. Junctions are flagged when an orientation column at an edge shows an extra local maximum between the two vessel directions; nearby candidates are clustered in position and orientation and merged — [ar5iv full text summary](https://ar5iv.labs.arxiv.org/html/1212.3530)
- Results: on a complex-junction test (HRFI sub-images, 27 seeded vessels), ETOS on cake-wavelet scores failed on 1 vessel, ETOS on Gabor scores on 3, and multi-scale Gabor centerline tracking (CTOS) on 5. CTOS sometimes merged parallel small vessels. Width measurement on REVIEW: success 99.83–100%; width-error σ 0.36 (KPIS), 0.53 (CLRIS, lower than the observers' 0.57–0.70), 0.80 (VDIS), 0.45 (HRIS). This is comparable to ESP, Graph and ARIA — [ar5iv](https://ar5iv.labs.arxiv.org/html/1212.3530)
- Citation: *J Math Imaging Vis* 49(3):583–610 (2014), doi:10.1007/s10851-013-0488-6, arXiv:1212.3530 — [arXiv](https://arxiv.org/abs/1212.3530)

**Hannink, Duits, Bekkers (2014) — crossing-preserving multi-scale vesselness — H**
- Multi-scale Frangi vesselness "cannot properly cope with crossings or bifurcations since it only looks for elongated structures". Applying vesselness inside multi-scale invertible orientation scores performs "considerably better at enhancing vessels throughout crossings and bifurcations". The comparison is qualitative in the abstract. Citation: "Crossing-preserving multi-scale vesselness," MICCAI 2014; arXiv:1402.4963 — [arXiv](https://arxiv.org/abs/1402.4963); [FAU CRIS](https://cris.fau.de/publications/109977164?lang=en_GB)

### Inferences
- The cortex-inspired literature closest to the user's pipeline mainly solves **enhancement and continuation through a crossing**: orientation layers separate the two vessels. It does not solve **deciding whether a junction is a crossing**. Where a decision is made (BICROS), it is a supervised classifier on top of fixed orientation-score features. A fully fixed-weight decision stage would be novel, but there is no published fixed-weight benchmark to beat.
- The COSFIRE numbers (~97% P/R) are for bifurcation *presence* on clean binary maps. They should not be compared with the user's F1, which includes type classification.
- The Abbasi-Sureshjani 2017 result (empirical vessel co-occurrence ≈ V1 association-field model, <2% error, dataset-independent) gives the strongest published justification for using fixed, cortex-derived good-continuation kernels on vessels.
- Inhibition-augmented COSFIRE suggests a mechanism the user's rule-based reader could borrow: an explicit inhibitory "crossing template" that vetoes the bifurcation response (a push-pull arrangement), rather than a single score with a threshold.

### Gaps
- BICROS's own ISBI 2016 numbers for classifying crossing vs bifurcation could not be retrieved (IEEE paywall; the thesis PDF was not found).
- No quantitative junction-level evaluation of Favali 2016 or Abbasi-Sureshjani 2018 was found. Both papers appear to report qualitative or patch-level grouping results.
- The full ETOS junction-resolver results on HRFI were in an unread part of the paper.
- The Hannink 2014 numbers (dataset, accuracy) were not in the abstract.

---

## 2. Graph-based topology / artery-vein reconstruction and junction classifiers (hand-designed vs learned)

### Takeaway
Hand-designed geometric rule systems report the highest crossing scores (≈87–93% recall/precision; Hamad 2014), but only on **manual segmentations with their own annotations**. On raw images with independent ground truth, learned detectors reach only ~0.6–0.7 junction F1. When junction type is evaluated directly, crossover accuracy is strikingly close to the user's plateau:
- **De et al. 2016:** 0.51 (DRIVE) and 0.43 (STARE), graph label propagation with angle-only weights.
- **Pratt et al. 2018:** CNN patch-level crossing recall ≈0.41–0.67.

The biggest documented gains come from **global constraints**. In Estrada et al., adding tree-level priors (arteries cross only veins, growth model, colour) raised A/V accuracy from 0.54–0.79 (local orientation search) to 0.91–0.94.

### Cited Findings

#### 2a. Hand-designed geometric/graph junction classifiers

**Hamad, Tegolo & Valenti (2014) — H, unsupervised rules on a skeleton**
- Data: DRIVE and STARE, 20 test images each, using the first *manual* segmentation. Ground truth: one ophthalmologist labelled junction type. DRIVE test: 2167 bifurcations, 520 crossings. STARE: 1146 bifurcations, 370 crossings. The optic disc was excluded — [Hamad et al. 2014 PDF text](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- Junction finding: skeleton pixels with ≥3 neighbours, giving graph vertices — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- Crossing vs bifurcation rules, all thresholds set empirically:
  - **Direct crossings** are degree-4 vertices. Opposite segments pair up. Width ratio ρ = min over pairs of min/max width must be ≥0.15, and width difference δ = max(|w1−w3|, |w2−w4|) <6.2 px. Angles: 0 ≤ α1 ≤ α2 ≤ 173°, 35° ≤ β1 ≤ β2 ≤ 180°, and the deflection Δ between each segment and its continuation ≤90°.
  - **Non-direct crossings** are two degree-3 vertices joined by a skeleton edge d ≤ 25 px. Candidates are sorted by edge length and the shortest is checked first. Stricter rules apply: ρ ≥ 0.25, δ ≤ 3 px, and angle bounds (α1 ≤ α2 ≤ 166°, α1+α2 ≤ 260°, α2−α1 ≤ 135°, β2−β1 ≤ 113°, 88° ≤ Δ1 ≤ Δ2 ≤ 267°).
  - **Connectivity rule:** in a true crossing, at least one segment of each crossing pair must connect onward to other vessels (blood-supply argument). Otherwise the configuration is read as two branches of one vessel, subject to a segment-length threshold.
  - **5-segment crossing** (two vessels crossing plus a daughter branch) is handled; more complex crossings are not.

  — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- Results (RE/PR %): DRIVE bifurcations 94.9 / 84.5, crossings 86.8 / 87.6. STARE bifurcations 95.9 / 89.6, crossings 88.4 / 93.3. Jaccard index: DRIVE 80.2 (bifurcations) / 76.5 (crossings); STARE 84.3 / 80.7. Abstract averages: bifurcations 95.4 / 87.1, crossovers 87.6 / 90.5 — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- **Key structural fact:** of the 518 crossings the algorithm found in DRIVE, 326 were "non-direct" (split into two degree-3 nodes) and 192 were direct. In STARE, 252 of 349 were non-direct — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- Stated failure modes: two vessels that intertwine more than once; spurious or missing segments in the binary image; unusual vessel structure; misclassified crossings default to bifurcations, which reduces the crossing count — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- Comparison table, as reported by Hamad (DRIVE, RE/PR %):
  - Al-Diri et al.: bifurcations RE 71; crossings RE 48 / PR 69.
  - Aibinu et al.: bifurcations 94.8 / 51.9; crossings 4.6 / 85.3.
  - Fathi et al. 2013: bifurcations 80.8 / 88.4; crossings 85.9 / 78.9.

  STARE: Aibinu: crossings 6.1 / 87.5. Fathi: bifurcations 88.2 / 83.8, crossings 82.9 / 85.8 — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- Citation: H. Hamad, D. Tegolo, C. Valenti (2014), "Automatic detection and classification of retinal vascular landmarks," *Image Analysis & Stereology* 33:189–200, doi:10.5566/ias.1101 — [journal page](https://ias-iss.org/ojs/IAS/article/view/1101)
- Fathi et al. (2013): a circular "local vessel pattern operator" applied to the thinned centerline, H. Citation: A. Fathi, A. Naghsh-Nilchi, F. Mohammadi, "Automatic vessel network features quantification using local vessel pattern operator," *Comput Biol Med* (2013), doi:10.1016/j.compbiomed.2013.01.011 — [Hamad et al. description](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952); [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.compbiomed.2013.01.011)

**Calvo, Ortega, Penedo, Rouco (2011) — H**
- Two steps: (i) filters and morphological operations extract the vessel structure (crease-based); (ii) each feature point's surroundings are analysed to classify it as crossover or bifurcation — [Hamad et al. description](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952). A secondary summary reports "93.6% of accuracy in the identified locations" — [search summary citing Hamad et al.](https://staff-old.najah.edu/2521/published-research/unsupervised-recognition-retinal-vascular-junction-points) (not verified against the primary paper).
- Citation: *Comput Methods Programs Biomed* 103(1):28–38 (2011), doi:10.1016/j.cmpb.2010.06.002 — [UDC portal](https://portalinvestigacion.udc.gal/documentos/609c22921aec1f036bb1dcfd)

**Sánchez, Barreira, Penedo, Cancela (2015) — H, minimal paths**
- Segments the tree, analyses the area around each interest point, and computes the **minimal path between vessel points** in that area to find which segments are connected. This separates bifurcations from crossovers. Accuracy 90% on DRIVE + VICAVR. Motivation: detecting arteriovenous nicking. Citation: "Automatic identification of vessel crossovers in retinal images," Proc. SPIE (2015), doi:10.1117/12.2181376 — [Semantic Scholar abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1117/12.2181376); [unpaywall](https://unpaywall.org/10.1117%2F12.2181376)

**Tsai, Stewart, Tanenbaum, Roysam (2004) — ERPR, H, model-based junction localisation**
- "Exclusion region and position refinement" (ERPR): vessel geometry inside the junction region is excluded, and the junction position and orientations are refined from the model. Median location error fell from 2.04 px to 1.1 px, median spread (repeatability) from 2.09 to 1.05 px, and vessel-orientation error from 7.2° to 3.8°. Applies to branching and crossover landmarks. Citation: *IEEE Trans Inf Technol Biomed* 8(2):122–130, doi:10.1109/TITB.2004.826733 — [OpenAlex abstract](https://api.openalex.org/works/doi:10.1109/TITB.2004.826733)

**Al-Diri, Hunter, Steel, Habib (2010) — H/L-lite (SOFM-learned cost functions)**
- Builds a vessel graph from segmented vessels by analysing the potential connectivity of segment ends. Self-organizing feature maps model implicit cost functions for junction geometry. These costs "resolve the configuration of local sets of segment ends". There are "specialized algorithms to handle overlapping vessels". Tested on DRIVE junctions. Citation: "Automated analysis of retinal vascular network connectivity," *Comput Med Imaging Graph* 34(6):462–470, doi:10.1016/j.compmedimag.2009.12.013 — [Semantic Scholar abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.compmedimag.2009.12.013)

**Qureshi, Hunter, Al-Diri (2014) — Bayesian junction configuration — L (shallow probabilistic model)**
- Builds a probabilistic model of joint types (terminals, **bridges**, bifurcations) and their configuration within a junction, then selects the most likely configuration by MAP. The model was fitted on 3010 joints from the DRIVE reference set and evaluated on 3435 DRIVE test joints: **95.2% accuracy**. Citation: "A Bayesian Framework for the Local Configuration of Retinal Junctions," CVPR 2014, doi:10.1109/CVPR.2014.397 — [Semantic Scholar abstract](https://api.semanticscholar.org/graph/v1/paper/search/match?query=A%20Bayesian%20framework%20for%20the%20local%20configuration%20of%20retinal%20junctions); [CVF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Qureshi_A_Bayesian_Framework_2014_CVPR_paper.pdf)

**Lau, Lee, Hsu, Wong (2013) — H (combinatorial optimisation)**
- Models the segmented vasculature as a vessel-segment graph and finds the optimal forest under constraints, i.e. it identifies whole "true vessels" through junctions. Evaluated on 2,446 retinal images: 98.9% pixel precision and 98.7% recall of true vessels on clean segmentations, and "robust" on noisy ones. Citation: "Simultaneously Identifying All True Vessels From Segmented Retinal Images," *IEEE TBME* 60(7):1851–1858, doi:10.1109/TBME.2013.2243447 — [OpenAlex abstract](https://api.openalex.org/works/doi:10.1109/TBME.2013.2243447)

**Dashtbozorg, Mendonça, Campilho (2014) — graph node-type analysis + A/V — H graph analysis + L intensity classifier**
- Intersection points are graph nodes and vessel segments are links. The method decides the type of each node (bifurcation, crossing, or meeting point in the 2012 RecPad version), then assigns one of two labels to each link. Final A/V combines this graph labelling with intensity features. A/V accuracy: 88.3% (INSPIRE-AVR), 87.4% (DRIVE), 89.8% (VICAVR). Citation: "An Automatic Graph-Based Approach for Artery/Vein Classification in Retinal Images," *IEEE TIP* 23(3):1073–1083, doi:10.1109/TIP.2013.2263809 — [OpenAlex abstract](https://api.openalex.org/works/doi:10.1109/TIP.2013.2263809); [RecPad 2012 version](https://repositorio-aberto.up.pt/handle/10216/85679)
- Hamad et al. cite Dashtbozorg et al. for the observation that crossings with more than four segments are "quite rare in fundus images" — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)

**Hu, Abràmoff, Garvin (MICCAI 2013; J Med Imaging 2015) — overlapping-tree separation — H topology + L A/V pixel SVM**
- Builds an over-connected "vessel potential connectivity map" (VPCM). Segment ends are linked within a search radius set by vessel length, tortuosity and alignment with the main arcades, and nearby ends are grouped into "neighbourhoods" (candidate landmarks) — [Hu et al. 2015, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785)
- Crossing vs bifurcation: each neighbourhood's candidate configurations are scored with local costs:
  - width: the parent must be wider than its children;
  - intensity profiles across six colour channels;
  - end-point direction (cosine);
  - balance of child angles, plus a bifurcation evaluator on child angles and lengths.
  
  Curvature is not used. A hard rule states that **arteries cross only veins and veins cross only arteries** — [PMC4652785](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785)
- Topology: segments are vertices and neighbourhoods are clusters of equality/inequality constraints. A/V separation becomes 2-colouring without conflict cycles, solved by a metaheuristic. The global cost penalises distance to the optic disc, bifurcation geometry, and inconsistency with the dominant blood-flow orientation — [PMC4652785](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785)
- Results on RITE: pixel-wise A/V accuracy 88.15% (manual vessel input) and 86.11% (automatic), against a pixel-classification-only baseline of 81.90% (automatic). The paper reports no separate crossing/bifurcation accuracy — [PMC4652785](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785)
- Citations: Q. Hu, M. D. Abràmoff, M. K. Garvin, "Automated Separation of Binary Overlapping Trees in Low-Contrast Color Retinal Images," MICCAI 2013, LNCS 8150:436–443, doi:10.1007/978-3-642-40763-5_54 — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/978-3-642-40763-5_54); [MICCAI LNCS page](https://iacl.ece.jhu.edu/proceedings/miccai2013/html/8150/81500436.htm). "Automated construction of arterial and venous trees in retinal images," *J Med Imaging* 2(4):044001 (2015), doi:10.1117/1.JMI.2.4.044001 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785)

**Estrada, Allingham, Mettu, Cousins, Tomasi, Farsiu (2015) — topology estimation + A/V — H model with fitted weights**
- Crossings vs branches are not decided by local rules. A directed tree hypothesis determines which projected vertices are branch points and which are crossings, and a global likelihood scores the hypothesis:
  - growth term: a one-inflated Poisson on child count (a vessel usually continues rather than splits) plus von Mises distributions on parent–child angles;
  - overlap term: the fraction of crossings that join an artery and a vein (A–A and V–V crossings are rare);
  - colour term: arteries are brighter.

  Weights on WIDE were 0.43 / 0.29 / 0.29. Width and explicit curvature are not used. The search contracts and expands vertex partitions over valid trees, visiting 2,500–14,600 trees per graph — [Estrada et al. 2015, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4685460)
- Graphs were extracted semi-automatically (Gabor tracks, then manual editing) — [PMC4685460](https://pmc.ncbi.nlm.nih.gov/articles/PMC4685460)
- Results (balanced A/V accuracy): WIDE 0.910 (second human rater 0.974); AV-DRIVE 0.935; CT-DRIVE 0.917 (Dashtbozorg 0.874 on the same set); AV-INSPIRE 0.909 (second rater 0.971). The **local-only "orientation search" baseline** gave only 0.776 (WIDE), 0.537 (AV-DRIVE), 0.559 (CT-DRIVE) and 0.793 (AV-INSPIRE). Junction-level accuracy is not reported — [PMC4685460](https://pmc.ncbi.nlm.nih.gov/articles/PMC4685460)
- Citation: *IEEE TMI* 34(12):2518–2534, doi:10.1109/TMI.2015.2443117 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4685460); [Duke PDF](https://people.duke.edu/~sf59/estradaTMI15.pdf)
- Theory (Estrada, Tomasi, Schmidler, Farsiu, "Tree Topology Estimation," *IEEE TPAMI* 2015, doi:10.1109/TPAMI.2014.2382116): a 3-D tree projects to a planar graph with crossings, so recovering tree topology from that graph is **ill-posed**. Under a generative tree-growth model, reconstruction takes linear time if each edge's direction (toward or away from the root) is known, but is **NP-hard if directions are unknown**. Tested on retinal vessels, plant roots and synthetic trees — [Semantic Scholar abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TPAMI.2014.2382116)

**Zhao Y. et al. (2020, TMI) and Xie et al. (ISBI 2018; CVPR 2019) — dominant-set clustering — H (hand-crafted similarity)**
- TMI 2020: the graph comes from segmentation, skeletonisation and identification of significant nodes. Edge weight = inverse Euclidean distance between the two end points in a feature space of **intensity, orientation, curvature, diameter and entropy**. Topology estimation and A/V are posed as pairwise (dominant-set) clustering. A/V accuracies: 95.1% (INSPIRE), 94.2% (IOSTAR), 93.8% (VICAVR), 91.1% (DRIVE), 91.0% (WIDE). The group also released manual topology annotations. Citation: "Retinal Vascular Network Topology Reconstruction and Artery/Vein Classification via Dominant Set Clustering," *IEEE TMI* 39(2) (2020), doi:10.1109/TMI.2019.2926492 — [Semantic Scholar abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TMI.2019.2926492); [Cranfield repository](https://dspace.lib.cranfield.ac.uk/handle/1826/14428)
- ISBI 2018 version: accuracies 0.915 (IOSTAR), 0.928 (INSPIRE), 0.889 (VICAVR). It names crossover handling as the key bottleneck — [search summary of Nottingham record](https://research.nottingham.edu.cn/en/publications/retinal-vascular-topology-estimation-via-dominant-sets-clustering)
- CVPR 2019: a curvilinear structural-similarity measure combining intensity and geometry, locally and globally, guides dominant-set clustering to "group curvilinear objects at crossover points into different connected branches". Evaluated on retinal vessels, plant roots and neuronal networks. Citation: J. Xie, Y. Zhao, et al., "Topology Reconstruction of Tree-Like Structure in Images via Structural Similarity Measure and Dominant Set Clustering," CVPR 2019, pp. 8505–8513, doi:10.1109/CVPR.2019.00870 — [CVF page](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Topology_Reconstruction_of_Tree-Like_Structure_in_Images_via_Structural_Similarity_CVPR_2019_paper.html); [Semantic Scholar abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/CVPR.2019.00870)

**De, Cheng, Zhang, Lin, Li, Ong, Yu, Yu, Ahmed (2016) — label propagation on digraphs — L segmentation + H/algebraic tracing**
- Neuronal (2-D fluorescence) and retinal (DRIVE, STARE) images. Pipeline: segmentation, skeleton, then a digraph whose nodes are filaments and whose edges join filaments meeting at a junction. Roots are optic-disc boundary pixels or DAPI nuclei. Hand rules convert cliques at junctions into directed edges. Tracing is label propagation by the matrix-forest theorem — [De et al. TMI 2016 manuscript](https://www.ece.ualberta.ca/~lcheng5/papers/DeEtAl_TMI16.pdf)
- **Direct crossover evaluation, as accuracy over filament pairs at junctions:** DRIVE 0.505 (405 crossovers, 849 pairs); STARE 0.426 (121 crossovers, 263 pairs); neuronal NeuB1 0.546 and NeuB2 0.536, against MetaMorph NX 0.449 and 0.180. The authors name two error sources: segmentation errors that alter topology, and a **weight matrix in which "just angular property of neighbouring filaments in a junction is considered"** — [De et al.](https://www.ece.ualberta.ca/~lcheng5/papers/DeEtAl_TMI16.pdf)
- Overall DIADEM tracing score (De method): DRIVE 0.79, STARE 0.42 — [De et al.](https://www.ece.ualberta.ca/~lcheng5/papers/DeEtAl_TMI16.pdf)
- Citation: "A Graph-Theoretical Approach for Tracing Filamentary Structures in Neuronal and Retinal Images," *IEEE TMI* (2016), doi:10.1109/TMI.2015.2465962 — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TMI.2015.2465962)

#### 2b. Learned junction detectors/classifiers (for contrast)

**Pratt et al. (2018) — two ResNet-18 CNNs on 21×21 patches along the skeleton of a *manual* segmentation — L**
- C1 decides junction vs non-junction; C2 decides bifurcation vs crossing. DRIVE split 30 train / 10 test; training used 101,416 junction patches and 216,756 other patches; ~11.2 M parameters. Image-level evaluation allows ±10 px. IOSTAR was resized to DRIVE scale and used unseen — [Pratt et al., J Imaging PDF](https://cgi.csc.liv.ac.uk/~frans/PostScriptFiles/joiPratt_2017.pdf)
- Patch detection accuracy ≈0.75–0.81 on DRIVE and 0.63–0.73 on IOSTAR. Image-level detection: sensitivity 0.85–0.99, specificity 0.71–0.81, accuracy 0.72–0.81 — [Pratt et al.](https://cgi.csc.liv.ac.uk/~frans/PostScriptFiles/joiPratt_2017.pdf)
- Patch classification (bifurcation vs crossing) accuracy: 0.80 (grader 1), 0.70 (grader 2) and 0.71 (grader 3) on DRIVE, against 0.65, 0.55 and 0.59 on IOSTAR. From the confusion matrices (true BF/CR rows), DRIVE G1A2 gives BF 13289 correct / 2068 wrong and CR 4801 correct / 2379 wrong. IOSTAR (trained on G1) gives BF 12375 / 1907 and CR 5527 / 7821 — [Pratt et al.](https://cgi.csc.liv.ac.uk/~frans/PostScriptFiles/joiPratt_2017.pdf)
- Uslu & Bharath's re-evaluation of Pratt at 5-px tolerance on GT vessel maps: P/R/F1 0.74 / 0.57 / 0.64 (DRIVE) and 0.52 / 0.54 / 0.53 (IOSTAR) — [Uslu & Bharath](https://ar5iv.labs.arxiv.org/html/1806.03175)
- Citation: H. Pratt, B. M. Williams, J. Y. Ku, C. Vas, E. McCann, B. Al-Bander, Y. Zhao, F. Coenen, Y. Zheng, "Automatic Detection and Distinction of Retinal Vessel Bifurcations and Crossings in Colour Fundus Photography," *J Imaging* 4(1):4 (2018), doi:10.3390/jimaging4010004 — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/jimaging4010004)

**Uslu & Bharath (MICCAI 2018) — multi-task network for junction detection — L (detection only, no crossing/bifurcation type)**
- A fully connected 5-hidden-layer network (RBM-pretrained) outputs vessel interior, centerline, edge and junction label patches. Candidates come from Hessian blobs on the junction map and are verified by counting ≥3 branches on rings of radius 3–6 px in the centerline map. Results (P/R/F1, 5-px tolerance): on fundus images 0.65 / 0.69 / 0.67 (DRIVE) and 0.52 / 0.67 / 0.59 (IOSTAR); on GT vessel maps 0.63 / 0.77 / 0.69 (DRIVE) and 0.74 / 0.82 / 0.78 (IOSTAR). Trained on DRIVE only. The authors state the method "is not designed to differentiate crossings from bifurcations" and that wide crossing vessels cause errors. Citation: F. Uslu, A. A. Bharath, "A Multi-task Network to Detect Junctions in Retinal Vasculature," MICCAI 2018, arXiv:1806.03175 — [ar5iv](https://ar5iv.labs.arxiv.org/html/1806.03175)

**Zhao H., Sun, Li (2020) — Faster-RCNN junction proposal + refinement + classification networks on raw colour images — L**
- No segmentation or skeleton. F1 ≈70% on DRIVE and ≈60% on IOSTAR, stated to beat the prior state of the art by 4.5 and 1.7 points. Citation: H. Zhao, Y. Sun, H. Li, "Retinal vascular junction detection and classification via deep neural networks," *Comput Methods Programs Biomed* 183:105096 (2020), doi:10.1016/j.cmpb.2019.105096 — [Semantic Scholar abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.cmpb.2019.105096); [BIT Pure](https://pure.bit.edu.cn/en/publications/retinal-vascular-junction-detection-and-classification-via-deep-n/)

### Inferences
- **The user's plateau matches the literature.** Independent direct measurements of crossover handling cluster around the user's 50–67%:
  - De et al., graph propagation with angle-only junction weights: 0.43–0.55.
  - Pratt et al., CNN crossing recall computed from the confusion matrices: DRIVE ≈ 4801/7180 = 0.67 (G1), ≈0.49 (G2), ≈0.51 (G3); IOSTAR ≈ 5527/13348 = 0.41. Bifurcation recall from the same matrices is ≈0.80–0.87.

  The asymmetry (bifurcation ≈85–90%, crossing ≈50–67%) is the same as the user's.
- **Why the plateau does not improve as noise → 0:** Estrada's TPAMI result shows that recovering tree topology from a 2-D projection is ill-posed and becomes tractable only with edge directions or global priors. Local-only search in Estrada 2015 gave 0.54–0.79 A/V accuracy, against 0.91–0.94 with global terms. A purely local junction reader is therefore limited by **structural ambiguity, not noise**. Removing noise cannot fix configurations whose local geometry is genuinely compatible with both readings.
- **Skeleton artefact is the most actionable finding.** In Hamad's data, ~63% of detected DRIVE crossings (326/518) and ~72% of STARE crossings (252/349) appear as two degree-3 nodes joined by a short edge (≤25 px) rather than as one degree-4 node. A reader that classifies each degree-3 node independently will call most crossings "bifurcations". This failure would persist at zero noise and fits the user's "thin crosses thick at shallow angle" failure, since shallow angles lengthen the split. Pairing nearby degree-3 nodes and testing continuation across the pair is a published, hand-designed fix.
- Two more noise-independent cues the user's rules may lack:
  - **The connectivity/blood-supply rule** (Hamad): a true crossing's arms must continue to other vessels; a short dead-end arm implies branching.
  - **The exclusion region** (Tsai ERPR): measure arm directions and widths outside the junction core, where the skeleton and widths are corrupted. This halved orientation error, 7.2° → 3.8°.
- **Global consistency constraints** add the largest documented gains and are hand-designable:
  - arteries cross veins, never their own type (Hu; Estrada);
  - a vessel usually continues rather than splits (Estrada's one-inflated Poisson);
  - the parent is wider than its children (Hu);
  - flow/root direction (Hu's dominant flow orientation; De's root and terminal rules).

  On synthetic data without A/V identity, the analogue is "each tree's members never cross each other", if the generator obeys that.
- High hand-designed numbers (Hamad ~87–93% crossings; Qureshi 95.2%; Sánchez 90%) are on **manual segmentations with their own labels**. Learned methods on raw images with independent labels reach ~0.6–0.7 F1. The user's 0.75–0.80 detection F1 on synthetic data is in a competitive range for detection, but synthetic data limits the comparison.

### Gaps
- Dashtbozorg 2014: the exact node-type decision rules and any node-classification accuracy could not be retrieved (paywall); only the A/V numbers are cited.
- Calvo 2011: the primary abstract and numbers were not retrieved; the 93.6% figure is second-hand.
- Xie CVPR 2019: per-dataset quantitative crossing/topology numbers were not retrieved (CVF PDF blocked, 403).
- Zhao CMPB 2020: separate bifurcation vs crossing F1 was not retrieved (server 503); only the overall F1 is cited.
- Lin et al. (Kalman tracking with minimum-cost matching at junctions, continuity of curvature, width and intensity) is mentioned only via Hamad's description; no primary numbers were retrieved.
- Pratt et al.'s patch counts look inconsistent. The text reports 67,650 crossing vs 33,766 bifurcation training patches, yet says bifurcation patches were "substantially more prominent". Treat their class balance with caution.

---

## 3. Datasets with junction annotations and inter-observer agreement

### Takeaway
Junction-type ground truth is scarce, group-specific and rarely reported with inter-observer statistics:
- **DRIVE and IOSTAR** junction annotations come from the Abbasi-Sureshjani/TU/e group and from Pratt et al.'s three graders.
- **Hamad et al.** used one ophthalmologist's DRIVE and STARE labels.
- **A/V topology sets** (RITE, AV-DRIVE, CT-DRIVE, WIDE, AV-INSPIRE, VICAVR) encode crossings implicitly.

No formal kappa for crossing vs bifurcation labelling was found. Pratt's cross-grader results (classification accuracy ~0.70–0.80 within DRIVE) suggest grader inconsistency is itself a ceiling.

### Cited Findings
- **DRIVE:** 40 colour images (Canon CR5, 45° FOV, 768×584 px). Uslu & Bharath report ≈130 junctions (bifurcations plus crossings) per image, with junction labels cited to Abbasi-Sureshjani et al. (BICROS, ISBI 2016) — [Uslu & Bharath](https://ar5iv.labs.arxiv.org/html/1806.03175); [Pratt et al.](https://cgi.csc.liv.ac.uk/~frans/PostScriptFiles/joiPratt_2017.pdf)
- **IOSTAR:** 24 SLO images (EasyScan, i-Optics; 1024×1024 px, 14 µm/px, 45° FOV). Vessels, bifurcations and crossings were annotated and corrected by two experts. ≈78 junctions per image — [Pratt et al.](https://cgi.csc.liv.ac.uk/~frans/PostScriptFiles/joiPratt_2017.pdf); [Uslu & Bharath](https://ar5iv.labs.arxiv.org/html/1806.03175)
- **Pratt et al. grading:** three expert graders, two of them with repeat annotations (G1A1/A2, G2A1/A2, G3A1/A2). Graders clicked as close as possible to the junction centre. Networks trained on one grader and tested on another grader performed similarly to within-grader tests: detection accuracy ≈0.72–0.81; classification accuracy 0.73–0.74 when tested on G1A1. The authors attribute limits partly to intra-grader variability and propose studying "intra and inter-observer variability" in future — [Pratt et al.](https://cgi.csc.liv.ac.uk/~frans/PostScriptFiles/joiPratt_2017.pdf)
- **Hamad et al. ground truth:** one expert ophthalmologist on the 20 test images. DRIVE: 2167 bifurcations, 520 crossings. STARE: 1146 bifurcations, 370 crossings — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- **Uslu & Bharath:** DRIVE junction labels not corresponding to the GT vessel maps were ignored; IOSTAR labels were not refined. This implies some misalignment between junction labels and vessel maps — [Uslu & Bharath](https://ar5iv.labs.arxiv.org/html/1806.03175)
- **A/V and topology sets used by Estrada et al.:**
  - WIDE: Optos 200Tx, 30 images, edge labels; second rater 0.974.
  - AV-DRIVE: Qureshi et al.
  - CT-DRIVE: Dashtbozorg et al., centerline labels. AV-DRIVE and CT-DRIVE agree on about 96% of pixels.
  - AV-INSPIRE: second rater 0.971.

  — [Estrada et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4685460)
- **RITE** (from DRIVE; A/V labels including an explicit "overlap" class and "uncertain") was used by Hu et al. — [Hu et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785)
- **VICAVR** was used for crossing identification (Sánchez et al., 90%) and A/V (Dashtbozorg 89.8%; Zhao 93.8%) — [Sánchez et al.](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1117/12.2181376); [Dashtbozorg et al.](https://api.openalex.org/works/doi:10.1109/TIP.2013.2263809); [Zhao et al. 2020](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TMI.2019.2926492)
- **Zhao et al. (TMI 2020)** report making manual topology annotations available (abstract truncated) — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TMI.2019.2926492)
- **COSFIRE ground truth:** the 40 DRIVE binary images contain >5000 bifurcations (Azzopardi's own annotation) — [CBMS 2012 record](https://um.edu.mt/library/oar/handle/123456789/26336)

### Inferences
- **Class prior:** crossings are about 19% of labelled junctions in DRIVE (520/2687) and about 24% in STARE (370/1516), per Hamad's ophthalmologist. An "always bifurcation" classifier therefore already scores ~76–81% junction-type accuracy. Crossing recall and balanced accuracy or F1 per class are the metrics to report and compare.
- Human agreement on A/V labels is high (≈0.97, Estrada). Human agreement on junction *type* and position is plausibly much lower: Pratt's cross-grader numbers are ~0.7–0.8, though they mix network error with grader differences. A synthetic benchmark with exact ground truth (as in the user's setup) avoids this confound, but its results will not translate one-to-one to clinical labels.

### Gaps
- No paper found reports Cohen's kappa or percentage agreement for crossing vs bifurcation labelling between graders.
- Azzopardi's DRIVE/STARE junction ground-truth files, and whether they mark crossings, were not verified.
- The VICAVR crossing annotation contents were not verified.

---

## 4. Photometric cues (overlap darkening/additive absorption), width/caliber rules (Murray's law, continuity) and angle rules — and how effective they are

### Takeaway
Width continuity and angle rules are the backbone of hand-designed crossing detection (Hamad, Hu, Al-Diri, Tsai). Intensity/colour continuity is used as an affinity term (Favali, Zhao, Hu, Estrada). Angle-only junction weights give only ~43–55% crossover accuracy (De et al.). No retinal paper was found that isolates and quantifies an "additive absorption at the overlap" cue. Murray's law appears only implicitly as "parent wider than children". Explicit physical additivity of overlapping vessels is used in X-ray angiography, not in fundus junction classification.

### Cited Findings
- **Width rules (Hamad):** opposite arms of a crossing must have similar widths. Min/max ratio ≥0.15 and difference <6.2 px for direct crossings; ≥0.25 and ≤3 px for split (non-direct) crossings. Combined with angle rules, these give crossing RE/PR of 86.8 / 87.6 (DRIVE) and 88.4 / 93.3 (STARE) on manual segmentations — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- **Width measured off-junction:** Hamad measures width perpendicular to the local centerline, with a rotational method for short segments. Directions are estimated from "the nearest portion of the vessel to the junction" using SVD on segment pixels — [Hamad et al.](https://ias-iss.org/ojs/IAS/article/viewFile/1101/952)
- **Caliber hierarchy (Hu):** in bifurcation configurations the parent must be wider than its children, and the bifurcation evaluator also uses child-angle balance and lengths. Intensity profiles across six colour channels compare candidate continuations — [Hu et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785)
- **Angle priors (Estrada):** von Mises distributions on parent–child angles, plus a one-inflated Poisson child-count prior. Width and curvature were not used. Colour: arteries brighter than veins — [Estrada et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4685460)
- **Angle-only weights are insufficient (De et al.):** junction weights using "just angular property" were named as a main error source; crossover accuracy was 0.505 (DRIVE) and 0.426 (STARE) — [De et al.](https://www.ece.ualberta.ca/~lcheng5/papers/DeEtAl_TMI16.pdf)
- **Orientation accuracy at junctions (Tsai ERPR):** excluding the junction core cut orientation error from 7.2° to 3.8° and location error from 2.04 to 1.1 px — [OpenAlex abstract](https://api.openalex.org/works/doi:10.1109/TITB.2004.826733)
- **Intensity continuity across crossings (Favali):** the method uses "continuity in orientation and intensity of the artery and vein respectively" at arteriovenous crossings, with low local variation of orientation and intensity along individual vessels — [search summary of arXiv 1512.06559](https://arxiv.org/pdf/1512.06559)
- **Multi-feature similarity (Zhao 2020):** edge weights combine intensity, orientation, curvature, diameter and entropy — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TMI.2019.2926492)
- **Expert cue set in neuron tracing (Gala et al.):** anatomists resolve topological ambiguities by combining distances between branches, branch orientations, intensities, calibers, tortuosities, colours, and spines/boutons. Encoding these features in a classifier gave <1% generalisation error with few training examples (active learning) — [Gala et al. 2014](https://api.semanticscholar.org/graph/v1/paper/search/match?query=Active%20learning%20of%20neuron%20morphology%20for%20accurate%20automated%20tracing%20of%20neurites)
- **Physical additivity at overlaps (X-ray):** projection imaging stacks 3-D structures as "semi-transparent layers that overlap each other". A modelling patent describes the overlap region's value as the sum of gray values contributed by each intercepted structure along the ray. This is the physical basis of the darker crossing in transmission imaging — [patent search summary](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12694481); [patent search summary](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6501848)
- **X-ray layer separation:** Ma, Hoogendoorn, Regar, Niessen, van Walsum (2017), "Automatic online layer separation for vessel enhancement in X-ray angiograms for percutaneous coronary interventions," *Medical Image Analysis*, doi:10.1016/j.media.2017.04.011. It separates the vessel layer from overlapping background layers — [TU Delft record](https://research.tudelft.nl/en/publications/automatic-online-layer-separation-for-vessel-enhancement-in-x-ray/); [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.media.2017.04.011)
- **Artery/vein crossing rule (topological "colour" cue):** arteries cross only veins (Hu, a hard constraint); A–A and V–V crossings are rare (Estrada, a soft overlap term) — [Hu et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785); [Estrada et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4685460)

### Inferences
- The user's cue set (good continuation, width similarity, optical-density additivity, amodal completion) overlaps heavily with published hand-designed systems. The extra elements that published systems with good crossing scores add are:
  - **pairing split junctions** (Hamad's non-direct crossings);
  - **onward-connectivity checks** (Hamad);
  - **off-junction measurement** (Tsai);
  - **global tree/A-V constraints** (Hu; Estrada).
- **Optical-density additivity is physically sound** for transmission imaging (X-ray, and broadly fundus/microscopy absorption). However, it is not a validated discriminator in the retinal junction literature. It may fail for a thin vessel over a thick one: the thin vessel's added optical density is small relative to the thick vessel's, so the overlap barely darkens. That is exactly the user's hardest case. Treat it as a weak cue whose weight should scale with the expected optical density of the thinner vessel.
- For thin-over-thick shallow crossings, the most discriminative cues are probably:
  - continuation of the thin vessel on both sides, with matched width and orientation, measured outside the thick vessel's footprint (an exclusion region the size of the thick vessel's width);
  - the absence of a caliber change in the thick vessel across the junction. A real bifurcation should reduce the parent's caliber (Murray-like), while a crossing leaves it unchanged.

  The second cue is an inference from the parent-wider-than-children rule, not a published result.

### Gaps
- No paper found quantifies the benefit of an overlap-darkening (additive optical density) cue for crossing vs bifurcation in fundus or microscopy images.
- No paper found uses Murray's law (cube law/junction exponent) as an explicit classifier feature with reported accuracy. Only "parent wider than children" (Hu) and width-similarity thresholds (Hamad) were found.
- I did not verify the details of Ma et al.'s layer model (e.g., whether it is explicitly log-domain/Beer–Lambert additive); only title and venue were confirmed.
- Retinal anatomy priors on which vessel lies on top at A/V crossings (artery anterior in a majority of normal crossings, per BRVO case-control abstracts at ~54–65% of control crossings) came only from aggregator snippets and are not verified ([qxmd aggregator](https://www.qxmd.com/r/8460014)).

---

## 5. Methods outside the retina: neuron tracing (crossing neurites), microvascular microscopy (capillaroscopy, SDF/IVM) and video/flow-based topology

### Takeaway
Neuron tracing faces the same crossover bottleneck and uses the same toolbox:
- graph label propagation (De 2016: crossover accuracy 0.54–0.55 on 2-D neuronal data);
- path classifiers with integer programming that allow loops (Türetken 2016);
- learned feature-based topology scoring (Gala 2014: <1% error with active learning);
- recent pixel-embedding networks.

Common 2-D microvascular tools (AngioTool, REAVER) count skeleton branchpoints and do not separate crossings from bifurcations. Video/flow is used mainly for velocity: space-time diagrams in SDF, RBC tracking in intravital video, and fluorescein filling dynamics separating arteries, veins and capillaries. I found no 2-D method that uses flow to resolve crossing vs bifurcation at individual junctions.

### Cited Findings

**Neuron tracing**
- **De et al. 2016:** 2-D neuronal fluorescence datasets NeuB1/NeuB2 (roots = DAPI-tagged nuclei). Crossover accuracy 0.546 (NeuB1) and 0.536 (NeuB2), against 0.449 and 0.180 for the commercial MetaMorph NX. The authors say "local and global contextual information is crucial to resolve the crossover issue" — [De et al.](https://www.ece.ualberta.ca/~lcheng5/papers/DeEtAl_TMI16.pdf)
- **Gala, Chapeton, Jitesh, Bhavsar, Stepanyants (2014):** neurites appear broken (imperfect labelling) or fused (limited optical resolution). A classifier scores alternative topologies from branch distances, orientations, intensities, calibers, tortuosities and colours, and is trained by active learning with a confidence measure: <1% generalisation error with few examples. Citation: "Active learning of neuron morphology for accurate automated tracing of neurites," *Front Neuroanat* 8:37 (2014), doi:10.3389/fnana.2014.00037 — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4032887/)
- **Türetken, Benmansour, Andres, Głowacki, Pfister, Fua (2016):** a graph of candidate paths is weighted by discriminatively trained path classifiers, and integer programming selects the subset under topological constraints. The model allows loops, penalises spurious junctions and early terminations, and outperformed the state of the art on aerial roads and neural micrographs. Citation: "Reconstructing Curvilinear Networks using Path Classifiers and Integer Programming," *IEEE TPAMI* 38(12) (2016), doi:10.1109/TPAMI.2016.2519025 — [Harvard VCG](https://vcg.seas.harvard.edu/publications/20160211-reconstructing-curvilinear-networks-using-path-classifiers-and-integer-programming); [EPFL PDF](https://infoscience.epfl.ch/record/201670/files/turetken16_1.pdf?version=1)
- **Pixel-embedding neurite segmentation (arXiv 2507.23359, 2025):** fiber crossing and self-entanglement are named as one of two main obstacles. Two-class semantic segmentation networks "frequently" merge unrelated fibers at crossings. Proposed fix: a discriminative embedding loss plus graph post-processing at crossing and discontinuity sites — [arXiv](https://arxiv.org/pdf/2507.23359)
- **2022 review of light-microscopy neuron tracing:** dense arbors may artificially intersect in images, producing crossover structures in reconstructions; "fiber missing and crossing are common errors" — [PMC9750132](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9750132/)

**Microvascular (2-D microscopy, SDF/IVM)**
- **AngioTool** (Zudaire et al., *PLoS ONE* 2011) reports vessel area, length, density, lacunarity and a "branching index" (branch points per unit area), validated on embryonic hindbrain, postnatal retina and allantois explants. Junctions are skeleton branch points, with no crossing/bifurcation distinction — [PMC3217985](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3217985/)
- **REAVER** (Corliss et al., *Microcirculation* 2020) gives high accuracy and precision for vessel length density, area fraction, mean diameter and branchpoint count against manual analysis of 2-D fluorescent murine images. Segment count is "very similar to branchpoint count" — [PMC7507177](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7507177/); [PubMed 32173962](https://pubmed.ncbi.nlm.nih.gov/32173962/)
- **SDF automated analysis:** space-time diagrams are acquired automatically along each vessel centerline, corrected for curvature, and the orientation of structures in the diagram is converted to RBC velocity. Bright bands mark plasma gaps or white cells, dark bands red cells — [US patent 8526704, "Measurement of functional microcirculatory geometry and velocity distributions using automated image analysis"](https://patents.google.com/patent/US8526704)
- **Bezemer, Dobbe, et al. (2011):** "Rapid automatic assessment of microvascular density in sidestream dark field images," *Med Biol Eng Comput* 49(11):1269–1278 — [Amsterdam UMC record](https://pure.amsterdamumc.nl/en/publications/erratum-rapid-automatic-assessment-of-microvascular-density-in-si/)
- **Fraser, Goldman, Ellis (2012, *Microcirculation*):** protocol and software that reconstruct (3-D) capillary networks from in vivo video of microvascular flow. From the same video they measure RBC velocity, hematocrit, RBC supply rate and SO2, and index these onto specific vessels in the network map — [PMC3393846](https://pmc.ncbi.nlm.nih.gov/articles/PMC3393846)
- **Video intravital microscopy artefacts:** a 2022 review notes that intravital microscopy shows a flat microcirculatory network, but artefacts such as vessel movement and **superposition** affect velocity estimates — [Frontiers in Physiology 2022](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.886675/pdf)

**Video/flow-based A/V or connectivity**
- **Fluorescein video angiography (rat, 30 fps):** pixel-by-pixel temporal analysis (PCA; half-rise and half-fall times) "clearly delineates differences in the temporal profiles of arteries, veins and capillaries". Citation: "Quantitative Spatial and Temporal Analysis of Fluorescein Angiography Dynamics in the Eye," *PLoS ONE* 9(11):e111330 (2014) — [PMC4218721](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4218721/)
- **Ultra-widefield angiography:** arterial and venous filling times overlap, so timing alone is not a clean A/V separator — [search summary, IJO](https://www.ijo.cn/gjyken/article/pdf/20210711)
- **Erythrocyte-mediated angiography (arXiv 2606.01006, 2026):** vessel masks guide cell tracking, and invisible capillaries are inferred with a minimum spanning tree. This is a flow-guided topology analogue — [arXiv](https://arxiv.org/pdf/2606.01006)
- **Hu et al.** use a "dominant blood-flow orientation" consistency term in the global cost. This is static and inferred from the arcade geometry, not measured from video — [PMC4652785](https://pmc.ncbi.nlm.nih.gov/articles/PMC4652785)

**3-D (brief)**
- Invertible orientation scores have been extended to 3-D to enhance complex vasculature, handling bifurcations better — [arXiv 1707.02191](https://arxiv.org/pdf/1707.02191). Fraser et al. reconstruct networks in 3-D from video — [PMC3393846](https://pmc.ncbi.nlm.nih.gov/articles/PMC3393846)

### Inferences
- In microscopy video, temporal signals (RBC/plasma-gap streaks in space-time diagrams, or dye-front arrival) are a candidate cue for crossing vs bifurcation. At a true bifurcation the streak velocity and flux should split between daughters (flux conservation). At a crossing, each vessel's streak pattern should pass through unchanged. This is an inference: no paper found tests it at the junction level.
- Neuron-tracing results reinforce the retinal ones: local angle-only rules give about 50% crossover accuracy (De). Large improvements come from multi-feature topology scoring (Gala) or global optimisation with structural constraints (Türetken), both learned or optimised. The fixed-weight analogue is a global consistency stage (e.g., a recurrent constraint-satisfaction layer) rather than richer local filters.
- Common 2-D microvascular analysis tools treat every skeleton branchpoint as a junction (AngioTool, REAVER). Published 2-D microvascular "branchpoint" statistics may therefore include crossings and contact artefacts. A crossing-aware junction reader would directly improve such metrics.

### Gaps
- No method was found for nailfold capillaroscopy junction or crossing handling, or for SDF/IDF videos that classify crossings vs bifurcations.
- No 2-D method was found that uses kymographs, space-time diagrams or optical flow specifically to resolve junction topology. The flow-based work found addresses velocity or A/V labelling, not junction type.
- The Dobbe et al. 2008 journal paper (presumed source of the SDF space-time method) was not retrieved; only the patent of the same title was seen, and its inventors were not confirmed.
- The 2-D neuronal crossing literature beyond De, Gala, Türetken and the 2025 preprint (e.g., NeuronStudio, NeuroGPS-Tree, APP2 crossing handling) was not searched in depth because of the time budget.
