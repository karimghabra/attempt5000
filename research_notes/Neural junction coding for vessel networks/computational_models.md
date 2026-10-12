# Neurally inspired computational models of junction, corner and crossing detection and classification

Scope note: these notes cover hand-designed or minimally parameterized models inspired by visual cortex or its geometry. The end use is a fixed-weight network that identifies and classifies vessel junctions (bifurcation versus crossing). Each model entry gives the architecture, what it detects, how it was evaluated, its limitations and a full citation. DOIs were checked against Crossref. Some DOIs contain parentheses, and in those links the parentheses are percent-encoded. Where only an abstract or a secondary description could be read, the bullet says so.

---

## 1. End-stopped and keypoint operators (Heitger et al. 1992; Heitger & von der Heydt 1993; Würtz & Lourens 2000; Dobbins, Zucker & Cynader; descendants)

### Takeaway
The Heitger et al. (1992) cascade is a fixed-weight, fully specified operator chain:
1. even/odd Gabor "S-operators";
2. a quadrature energy "C-operator";
3. first and second spatial derivatives of C along the preferred axis, giving single and double end-stopped cells;
4. tangential and radial inhibition;
5. local maxima, which are the keypoints.

It marks line ends, corners and high-curvature points. It does not respond to the middle of straight contours, and by design it gives a weak response at X-crossings. The closely related Dobbins/Zucker "difference" end-stopped cells add coarse curvature magnitude and sign. For vessels this family is the natural "termination detector". A bifurcation or T-junction contains terminating branches, whereas an ideal crossing contains none.

### Cited Findings
**Heitger, Rosenthaler, von der Heydt, Peterhans & Kübler (1992)**
- Citation: Heitger F, Rosenthaler L, von der Heydt R, Peterhans E, Kübler O (1992). "Simulation of neural contour mechanisms: from simple to end-stopped cells." *Vision Research* 32(5):963–981. DOI 10.1016/0042-6989(92)90039-L — [DOI](https://doi.org/10.1016/0042-6989%2892%2990039-L); [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/1604865/)
- Architecture as stated in the abstract — [PubMed](https://pubmed.ncbi.nlm.nih.gov/1604865/):
  1. convolution with even- and odd-symmetric orientation-selective filters (S-operators);
  2. combination of even and odd outputs into a local energy measure (C-operator);
  3. "differentiation" of the C-operator maps along the respective orientation, giving single and double end-stopped operators;
  4. determination of local maxima ("key-points") of the combined end-stopped activity.
- What it detects: "S- and C-operators are optimised for the representation of 1-D features such as edges and lines", while "end-stopped operator responses at the key-points make explicit 2-D signal variations such as line ends, corners and segments of strong curvature". Testing was on grey-valued images and was qualitative, as far as the abstract states — [PubMed](https://pubmed.ncbi.nlm.nih.gov/1604865/)

**Heitger et al. (1998), follow-up**
- Citation: Heitger F, von der Heydt R, Peterhans E, Rosenthaler L, Kübler O (1998). "Simulation of neural contour mechanisms: representing anomalous contours." *Image and Vision Computing* 16:407–421. DOI 10.1016/S0262-8856(97)00083-8 — [DOI](https://doi.org/10.1016/S0262-8856%2897%2900083-8)
- It extends the model to occluding and illusory contours with a grouping operator that links collinear termination features (line-ends, corners). This is from an abstract-level description — [ScienceDirect abstract page via search](https://www.sciencedirect.com/science/article/abs/pii/S0262885697000838)

**Heitger & von der Heydt (1993)**
- Citation: Heitger F, von der Heydt R (1993). "A computational model of neural contour processing: figure-ground segregation and illusory contours." *Proc. 4th IEEE ICCV*, pp. 32–40. DOI 10.1109/ICCV.1993.378238 — [DOI](https://doi.org/10.1109/ICCV.1993.378238)
- An extended version appeared at PerAc '94, pp. 181–192 (Heitger, von der Heydt, Kübler), DOI 10.1109/FPA.1994.636103 — [DOI](https://doi.org/10.1109/FPA.1994.636103)

**Würtz & Lourens (1997, 2000): explicit equations for a Heitger-type implementation**
- Citations:
  - Würtz RP, Lourens T (2000). "Corner detection in color images through a multiscale combination of end-stopped cortical cells." *Image and Vision Computing* 18(6–7):531–541. DOI 10.1016/S0262-8856(99)00061-X — [DOI](https://doi.org/10.1016/S0262-8856%2899%2900061-X)
  - Conference version: Würtz & Lourens (1997), ICANN'97, LNCS 1327:901–906, DOI 10.1007/BFb0020267 — [PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf)
- Simple and complex cells: "Simple cells" are the real and imaginary parts of DC-corrected Gabor functions. Complex cells C_{σ,θ}(x,y) are the modulus. The authors "follow the model of [Heitger et al. 1992] and use the responses of complex cells as input for the end-stopped cells" — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf)
- Single end-stopped operator (first derivative of C), with c = cosθ and s = sinθ: S_{σ,θ}(x,y) = C_{σ,θ}(x+ds, y−dc) − C_{σ,θ}(x−ds, y+dc) — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf)
- Double end-stopped operator (second derivative): D_{σ,θ}(x,y) = C_{σ,θ}(x,y) − ½C_{σ,θ}(x+2ds, y−2dc) − ½C_{σ,θ}(x−2ds, y+2dc) — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf)
- Choice of the offset d: d is critical. If d is too large, a line is "eliminated in the middle" and becomes two segments. If d is too small, corners are mislocated. They chose d = 1.8 — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf)
- False responses are removed by two inhibition operators, each half-wave rectified [·]⁺ — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf):
  - tangential inhibition: I_t = Σ_i [−w_t C_{σ,θ_i} + C_{σ,θ_i}(x + d c_i, y + d s_i)]⁺
  - radial inhibition: I_r = Σ_i [C_{σ,θ_i}(x,y) − w_r C_{σ,(i+N/2)}(x + ½d c_i, y + ½d s_i)]⁺
- Inhibition and corner-operator parameters — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf):
  - w_t = 1 and w_r = 4, which "work well" for N ∈ {8, 16} orientations and σ ∈ [1, 10];
  - the final operator is X'_{σ,θ} = [X_{σ,θ} − g(I_t + I_r)]⁺ with g = 2;
  - the threshold is T = 5, followed by local-maximum thinning.
- Multiscale combination: Small scales localize well but are noise-sensitive. Large scales are imprecise. Corner strength varied by ≤20% across scales. The best combination was the plain average of uniformly sampled scales (S = 15, σ_min = 1.18, σ_max = 9.43, N = 8). An extension to colour builds "double-opponent" simple/complex/end-stopped cells for red–green and blue–yellow channels — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf)
- Evaluation and limitations: The evaluation is qualitative, on natural images with texture, noise, rounded and low-contrast corners. The authors claim it "compares favorably with known corner detectors", but no quantitative benchmark table was found. Parameters were tuned on one test image with hand-marked corners. Single-scale end-stopped models "performed poorly, especially with background texture and noise" — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf)

**Dobbins, Zucker & Cynader (1987, 1989): curvature-selective end-stopping**
- Citations:
  - Dobbins A, Zucker SW, Cynader MS (1987). "Endstopped neurons in the visual cortex as a substrate for calculating curvature." *Nature* 329:438–441. DOI 10.1038/329438a0 — [DOI](https://doi.org/10.1038/329438a0)
  - Dobbins, Zucker, Cynader (1989). "Endstopping and curvature." *Vision Research* 29(10):1371–1387. DOI 10.1016/0042-6989(89)90193-4 — [DOI](https://doi.org/10.1016/0042-6989%2889%2990193-4)
- Secondary description: end-stopped receptive profiles are modelled as a weighted difference of two co-located, co-oriented simple receptive profiles of different sizes, with the smaller field's positive weight exceeding the larger field's negative weight. Note that this source cites an unnamed reference [24] for the construction — [arXiv 1803.05783](https://arxiv.org/pdf/1803.05783)
- Limitation noted in secondary literature: the end-stopped model of curve detection "detects curvature with a low degree of accuracy" (Yu & Niitsuma 2007, as summarized in search results) — [J-STAGE](https://www.jstage.jst.go.jp/article/tjsai/22/6/22_6_595/_article)

**Rodríguez-Sánchez & Tsotsos (2011, 2012): a concrete implementation of Dobbins-style difference end-stopped cells**
- Citations:
  - Rodríguez-Sánchez AJ, Tsotsos JK (2011). "The importance of intermediate representations for the modeling of 2D shape detection: Endstopping and curvature tuned computations." *CVPR 2011*, pp. 4321–4326. DOI 10.1109/CVPR.2011.5995671 — [PDF](https://iis.uibk.ac.at/public/antonio/Rodriguez-SanchezCVPR2011.pdf)
  - Journal version: *PLoS ONE* 7:e42058 (2012), DOI 10.1371/journal.pone.0042058 — [DOI](https://doi.org/10.1371/journal.pone.0042058)
- Model details — [PDF](https://iis.uibk.ac.at/public/antonio/Rodriguez-SanchezCVPR2011.pdf):
  - The end-stopped cells explicitly follow A. Dobbins' (1991) thesis "Difference models of visual cortical neurons".
  - Simple cells are difference-of-Gaussians, with 12 orientations × 4 sizes, giving 48 types.
  - A complex cell is the sum of 5 laterally displaced simple cells.
  - The end-stopped response is RES = Φ[c_c φ(R_c) − (c_d1 φ(R_d1) + c_d2 φ(R_d2))]: an excitatory centre simple cell minus two inhibitory displaced complex cells, shifted by ½ receptive-field size, with gains c_d ∈ {0.7, 0.8, 1, 2}.
- Curvature sign and classes: "direction" end-stopped neurons, made by adding components rotated by ±45°, give the sign of curvature. Combining n end-stopped types with sign gives 2n curvature classes; for example, 4 types give 8 classes. The authors describe this as "a coarse curvature estimation" — [PDF](https://iis.uibk.ac.at/public/antonio/Rodriguez-SanchezCVPR2011.pdf)

**Rodrigues & du Buf (2006): multiscale keypoints**
- Citation: Rodrigues J, du Buf JMH (2006). "Multi-scale keypoints in V1 and beyond: object segregation, scale selection, saliency maps and face detection." *BioSystems* 86:75–90. DOI 10.1016/j.biosystems.2006.02.019 — [DOI](https://doi.org/10.1016/j.biosystems.2006.02.019)
- Abstract-level description: retinotopic keypoint maps are built from end-stopped cells at many scales, then used for grouping and object segregation, automatic scale selection and saliency maps. The 2005 predecessor (LNCS, DOI 10.1007/11565123_21) reports that keypoint detection works best with non-classical receptive-field inhibition, and credits Heitger et al. for the single-scale single- and double-stopped cells — [repository record](https://sapientia.ualg.pt/entities/publication/89af786d-a2c3-4433-8fa9-2b6fa9660ca9/full); [LNCS DOI](https://doi.org/10.1007/11565123_21)

### Inferences
- **Directly reusable.** The Würtz/Lourens equations, with d = 1.8, w_t = 1, w_r = 4, g = 2, N = 8–16 and 15 averaged scales, are a complete fixed-weight recipe for a "termination/keypoint" layer. For vessels, the C-operator should use line (even-symmetric, ridge) energy rather than edge energy, because vessels are bars. Scales should span the vessel-width range.
- **Branches, crossings and fills.** End-stopped responses are expected at the stems of bifurcations and T-junctions, where branches start or end. Ideal X-crossings, where both vessels continue, should give little end-stopped response. This asymmetry is the core cue for separating branching from crossing at the operator level. Weidenbacher & Neumann (2009) state this explicitly (see section 2). One caution: vessel width changes at bifurcations and gaps in segmentation produce spurious "terminations". So does central light reflex, a bright stripe along the vessel centreline that can make one vessel look like two parallel edges.
- **Curvature sign and magnitude.** The curvature classes from Rodríguez-Sánchez & Tsotsos give a cheap fixed-weight estimate of curvature sign and magnitude. Section 4 shows curvature can help separate high-curvature branching from low-curvature crossing.

### Gaps
- Full text of Heitger et al. (1992) could not be retrieved, so exact filter parameters and figures are known only through the Würtz/Lourens re-implementation.
- The occlusion and figure-ground labelling rules of Heitger & von der Heydt (1993) — for example how keypoint "direction" labels are assigned at T-junctions — could not be verified. Only the title and DOI are confirmed.
- No quantitative ROC or precision–recall numbers were found for Heitger-type or Würtz–Lourens keypoint detectors on a public benchmark.
- Full text of Dobbins et al. (1987) was not obtained. The exact size ratio and weights of the difference model are unverified.

---

## 2. Recurrent V1–V2 models of junctions (Hansen & Neumann 2004; Neumann & Sepp 1999; Hansen & Neumann 2008; Weidenbacher & Neumann 2009; Grossberg BCS/FACADE/3D LAMINART)

### Takeaway
The Ulm group (Neumann, Hansen, Weidenbacher) represents junctions implicitly as multi-orientation activity in a hypercolumn after recurrent long-range collinear grouping. Hansen & Neumann (2004) read junctions out with a circular-variance measure. Weidenbacher & Neumann (2009) go further and classify them explicitly with multiplicative AND-combinations of cell pools:
- T = V2 bipole × V1 end-stop at different orientations;
- X = bipole × bipole;
- L = end-stop × end-stop.

This is almost exactly a crossing-versus-branching classifier and is the most reusable design found. Grossberg's FACADE/3D LAMINART has no explicit junction detectors. T-junction sensitivity there emerges from bipole cooperation plus short-range competition, which creates "end gaps" in the T stem. The same competition can break one branch of an X-junction if the branches are unequal, which is a design hazard for vessel crossings.

### Cited Findings
**Hansen & Neumann (2004): circular-variance junction readout**
- Citation: Hansen T, Neumann H (2004). "Neural mechanisms for the robust representation of junctions." *Neural Computation* 16(5):1013–1037. DOI 10.1162/089976604773135087 — [DOI](https://doi.org/10.1162/089976604773135087)
- Core idea: "junctions are implicitly characterized by high activity for multiple orientations within a cortical hypercolumn. A local measure of circular variance is suggested to extract junction points from this distributed representation." The orientation measurements are first made coherent by "a model of V1 utilizing mechanisms of collinear long-range integration and recurrent interaction". Activities "consistent within a more global context are enhanced while inconsistent activities are suppressed" — [Crossref abstract / DOI](https://doi.org/10.1162/089976604773135087)
- Evaluation: Junction detection with the recurrent model was compared with a feedforward complex-cell model. The recurrent model improved "localization accuracy and positive correctness in the detection of generic junction configurations such as L- and T-junctions". ROC analysis on synthetic and camera images showed "superior performance of the new approach" — [DOI](https://doi.org/10.1162/089976604773135087)

**Hansen & Neumann (2002): conference precursor**
- Citation: Hansen T, Neumann H (2002). "A biologically motivated scheme for robust junction detection." *Biologically Motivated Computer Vision (BMCV 2002)*, LNCS 2525, pp. 16–26. DOI 10.1007/3-540-36181-2_2 — [dblp](https://dblp.org/pid/71/4468); [chapter page](https://mijn.bsl.nl/chapter/10.1007/3-540-36181-2_2)
- Evaluation: In ROC comparisons against Gaussian-curvature and structure-tensor junction detectors on artificial and natural images, "the new approach performs superior to the standard schemes" (abstract-level) — [chapter page](https://mijn.bsl.nl/chapter/10.1007/3-540-36181-2_2)
- Related papers:
  - Hansen T, Neumann H (2004). "A simple cell model with dominating opponent inhibition for robust image processing." *Neural Networks* 17:647–662, DOI 10.1016/j.neunet.2004.04.002 — [DOI](https://doi.org/10.1016/j.neunet.2004.04.002)
  - Hansen T, Sepp W, Neumann H (2001). "Recurrent long-range interactions in early vision." LNCS, DOI 10.1007/3-540-44597-8_9 — [DOI](https://doi.org/10.1007/3-540-44597-8_9)

**Neumann & Sepp (1999) and Hansen & Neumann (2008): the underlying grouping models**
- Citations:
  - Neumann H, Sepp W (1999). "Recurrent V1–V2 interaction in early visual boundary processing." *Biological Cybernetics* 81:425–444. DOI 10.1007/s004220050573 — [DOI](https://doi.org/10.1007/s004220050573)
  - Hansen T, Neumann H (2008). "A recurrent model of contour integration in primary visual cortex." *Journal of Vision* 8(8):8. DOI 10.1167/8.8.8 — [DOI](https://doi.org/10.1167/8.8.8)
  - Neumann H, Mingolla E (2001). "Computational neural models of spatial integration in perceptual grouping." In *From Fragments to Objects* (Advances in Psychology 130), pp. 353–400, DOI 10.1016/S0166-4115(01)80032-7. This is a review of bipole, association-field and related grouping kernels — [DOI](https://doi.org/10.1016/S0166-4115%2801%2980032-7)

**Weidenbacher & Neumann (2009): explicit T/L/X junction classification by cell-pool AND-gates**
- Citation: Weidenbacher U, Neumann H (2009). "Extraction of surface-related features in a recurrent model of V1–V2 interactions." *PLoS ONE* 4(6):e5909. DOI 10.1371/journal.pone.0005909 — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- Feedforward stage: model V1 has simple, complex, end-stop and bipole cells per orientation, arranged as hypercolumns. Complex cells pass through divisive or shunting inhibition across orientation. This weakens responses where activity is spread over many orientations (texture, noise) and strengthens single-orientation responses — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- V1 bipole cells are "two additively connected elongated Gaussian subfields", "narrowly tuned to the preferred orientation", about 2× the complex-cell receptive-field length — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- V1 end-stop cells have "an elongated excitatory subfield and an inhibitory isotropic counterpart". They are direction-selective over 0–360°, and opposite directions are summed. They "respond to edges or lines that terminate within their RF … also corners or junctions … However, at positions along contours or at X-junction configurations, end-stop cells do not respond" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- V2 long-range bipole cells integrate V1 bipole and end-stop outputs over a larger extent and respond to illusory contours — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- Feedback (reusable gain-control motif):
  - V2→V1 feedback is modulatory. It multiplies feedforward complex-cell responses by (feedback + constant unit bias), so "feedback signals generate an enhancing gain factor which cannot on its own generate any activity at positions where no initial feeding input responses are present". This implements a "no-strong-loop" rule.
  - It must be paired with subsequent shunting lateral inhibition, "otherwise, the feedback signal would lead to uncontrolled growth".
  - "Few iterations" reach a stable state.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- Junction readout rules (Figure 6 of the paper):
  - "To extract a T-junction signal orientation specific V2 bipole cell activities are combined pairwisely with orientation specific V1 end-stop cell activities. Pairs of cells that correspond to same orientations are not considered … The combination … is done multiplicatively such that both cells have to be active."
  - "X-, and L-junction maps are extracted similarly except that pairwise combination is based on V2 bipole cells (X-junctions) and V1 end-stop cells (L junctions)."
  - Each map is summed over orientation pairs to give an orientation-invariant saliency map. The multiplicative connection "acts like an AND-gate which extracts only those V2 bipole responses that have a bimodal activity distribution in the orientation domain".
  - Centre-surround inhibition then "helps to suppress multiple junctions in a small neighbourhood which could be induced by fine texture or noise", and the junction maps "locally compete".
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- Context-dependent reclassification: an L-junction that becomes completed by an illusory contour "turn[s] into perceptual T-junctions" over the recurrent iterations. Y- and W-junctions are discussed as 3D-corner cues — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- Evaluation: ROC on a corner test image adapted from Smith & Brady, with ground-truth positions for L, T and X junctions. The recurrent model's ROC "lies well above the Harris corner detector curve and the initial feedforward model curve". The comparison was detection-only, because Harris cannot classify junction types. No per-class classification accuracy is reported in the passages read — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
- Limitations stated:
  - V2 bipoles cannot bridge large gaps; Kanizsa figures with large gaps fail.
  - The model "does not account for different image scales" (single scale; the authors suggest a pyramid).
  - It "does not explain how occlusion-based junctions can be distinguished from textu[re]".
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)

**Thielscher & Neumann (2008): T-junction cues for depth ordering**
- Citation: Thielscher A, Neumann H (2008). "Globally consistent depth sorting of overlapping 2D surfaces in a model using local recurrent interactions." *Biological Cybernetics* 98:305–337. DOI 10.1007/s00422-008-0211-7. Content was not read beyond the title; Weidenbacher & Neumann cite this line of models as using T-junction cues for layered surfaces — [DOI](https://doi.org/10.1007/s00422-008-0211-7); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)

**Grossberg BCS/FCS, FACADE and 3D LAMINART**
- Citations:
  - Grossberg S (1997). "Cortical dynamics of three-dimensional figure–ground perception of two-dimensional pictures." *Psychological Review* 104(3):618–658. DOI 10.1037/0033-295X.104.3.618 — [DOI](https://doi.org/10.1037/0033-295X.104.3.618)
  - Kelly F, Grossberg S (2000). "Neural dynamics of 3-D surface perception: figure-ground separation and lightness perception." *Perception & Psychophysics* 62:1596–1618. DOI 10.3758/BF03212158 — [DOI](https://doi.org/10.3758/BF03212158)
  - Grossberg S, Mingolla E (1985). "Neural dynamics of perceptual grouping: textures, boundaries, and emergent segmentations." *Perception & Psychophysics* 38:141–171. DOI 10.3758/BF03198851 — [DOI](https://doi.org/10.3758/BF03198851)
- T-junction mechanism (summarized in Grossberg's 2016 review, reprinting Grossberg 1997):
  - Bipole cells drive long-range recurrent excitatory horizontal connections (cooperation) and short-range inhibitory interneurons (competition).
  - Near a T-junction, "the top of the T activates bipole cells along the top of the T much more than bipole cells are activated along the T stem". Horizontal bipoles receive input from both sides of the stem.
  - So "the stem boundary gets inhibited by the short-range inhibitory signals from the horizontal bipole cells, whereas the top boundary does not receive comparable inhibition". The result is an end gap in the stem.
  - This "boundary ownership process is sensitive to image T-junctions, but there are no explicit T-junction detectors in the network."
  - [Grossberg 2016, PMC4726768](https://pmc.ncbi.nlm.nih.gov/articles/PMC4726768/)
- Simulations: Grossberg (1997) explains, and Kelly & Grossberg (2000) simulate, figure-ground percepts such as Kanizsa stratification and bistable transparency — [PMC4726768](https://pmc.ncbi.nlm.nih.gov/articles/PMC4726768/)
- X-junctions: when attention spreading along a Necker-cube line reaches an X-junction, "by strengthening the boundary of one branch of the X-junction, it breaks the boundary of the other branch near the intersection of the lines, causing end gaps". FACADE (Grossberg 1994; Grossberg & Yazdanbakhsh 2005) uses end gaps to trigger figure-ground separation and 3D neon spreading — [PMC4726768](https://pmc.ncbi.nlm.nih.gov/articles/PMC4726768/)
- Laminar and transparency follow-ups:
  - Grossberg S, Yazdanbakhsh A (2005). "Laminar cortical dynamics of 3D surface perception: stratification, transparency, and neon color spreading." *Vision Research* 45:1725–1743. DOI 10.1016/j.visres.2005.01.006 — [DOI](https://doi.org/10.1016/j.visres.2005.01.006)
  - Grossberg S, Raizada R (2000). "Contrast-sensitive perceptual grouping and object-based attention in the laminar circuits of primary visual cortex." *Vision Research* 40:1413–1432. DOI 10.1016/S0042-6989(99)00229-1 — [DOI](https://doi.org/10.1016/S0042-6989%2899%2900229-1)
  - Grossberg S, Swaminathan G (2004). "A laminar cortical model for 3D perception of slanted and curved surfaces and of 2D images…" *Vision Research* 44:1147–1187. DOI 10.1016/j.visres.2003.12.009. Here "angle cells" combined with straight bipole cells code tilt — [DOI](https://doi.org/10.1016/j.visres.2003.12.009); [PMC4726768](https://pmc.ncbi.nlm.nih.gov/articles/PMC4726768/)

### Inferences
- **Proposed vessel mapping of the Weidenbacher rules.** These are untested mappings:
  - Crossing ≈ "X": two continuous (bipole-supported) orientations and no terminations.
  - T-type branch ≈ "T": one continuous orientation plus one terminating orientation.
  - Symmetric Y-bifurcation: all three branches end at the junction, so end-stop × end-stop pairs are active ("L/Y-like"), possibly with weak bipole support across the obtuse angle.
  - In each case only pairs with different orientations are combined, products are summed over orientation pairs, and the junction maps compete.
  - A fixed-weight network can implement this directly as quadratic (product) units on orientation-indexed V1/V2 maps.
- **Circular variance readout.** A plausible fixed readout for Hansen & Neumann-style junction maps is CV(x) = 1 − |Σ_k a_k e^{i2θ_k}| / Σ_k a_k over orientation channels, gated by total activity. This is the standard circular-variance definition. Check the paper for its exact normalization. Circular variance detects multi-orientation points but does not by itself say whether the extra orientation continues (crossing) or terminates (branch). The end-stop/bipole factorization supplies that.
- **Crossing hazard.** In Grossberg-style bipole competition, a crossing between a thick and a thin vessel could produce an "end gap" in the thinner vessel and turn an X into a pseudo-T. To avoid this, a vessel network should either confine competition within orientation channels or make cross-orientation inhibition symmetric and weak. Otherwise crossings will be misread as occlusion T-junctions or branch points.
- **Feedback motif.** The modulatory feedback form, r = input × (1 + feedback) followed by shunting normalization, is a safe fixed-weight recurrence that cannot hallucinate vessels where there is no feedforward evidence.

### Gaps
- Full texts of Hansen & Neumann (2004, 2008) and Neumann & Sepp (1999) were not accessible. The exact kernel parameters (bipole length and opening angle, circular-variance threshold), the ROC numbers and the iteration counts could not be quoted.
- No per-class classification accuracy for T versus L versus X was found in Weidenbacher & Neumann (2009). Their quantitative comparison is detection-only.
- Grossberg (1997) and Kelly & Grossberg (2000) full texts were not read directly. Equations, parameters and quantitative fits are not reported here, and the mechanism description comes from the 2016 review.

---

## 3. Border-ownership and grouping-cell models (Craft et al. 2007; Zhaoping 2005; Sakai & Nishimura; Mihalas et al. 2011; descendants)

### Takeaway
These models assign which side of a contour is "figure". They do so with either:
- lateral V2 connections that encode convexity and T-junction consistency, without explicit junction detectors (Zhaoping 2005); or
- annular "grouping cells" (G) that pool border-ownership (B) cells and feed back modulatorily (Craft et al. 2007; Mihalas et al. 2011; Hu et al. 2019).

They are fixed-weight and image-computable, but they target occlusion and figure-ground rather than vessel topology. Their main reusable ideas are:
- size-independent latency through grouping cells rather than chains of horizontal links;
- connection rules that make "T-top/T-stem" pairs facilitate or suppress depending on ownership.

Hu et al. (2019) keep only the single best orientation per pixel, which cannot represent crossings.

### Cited Findings
**Zhaoping (2005): V2 lateral-connection model**
- Citation: Zhaoping L (2005). "Border ownership from intracortical interactions in visual area V2." *Neuron* 47(1):143–153. DOI 10.1016/j.neuron.2005.04.005 — [UCL Discovery full text](https://discovery.ucl.ac.uk/id/eprint/13395/); [DOI](https://doi.org/10.1016/j.neuron.2005.04.005)
- Units: for each location i and orientation θ there are two neurons with opposite preferred border-ownership (BOWN) sides. Interactions use excitatory connections J and disynaptic inhibitory connections W between neurons with nearby, not necessarily overlapping, classical receptive fields — [UCL PDF](https://discovery.ucl.ac.uk/id/eprint/13395/)
- Connection rules — [UCL PDF](https://discovery.ucl.ac.uk/id/eprint/13395/):
  - Co-aligned segments facilitate if they prefer the same figure side and inhibit otherwise.
  - A convexity bias gives stronger facilitation when the border turns toward the figure.
  - The T-junction rule: "while the model does not have neurons explicitly detecting T junctions, two neurons for two border segments which are plausibly the top and stem, respectively, of a T junction facilitate each other if the neuron for the T top has a BOWN preference consistent with an occluding surface, otherwise, suppressive connections are designed between the two neurons". A stated consequence is that "cells should manifest end-stopping properties".
  - Connection strength decays with distance, and no single contextual bar can drive a cell without receptive-field input.
- Two notable design choices — [UCL PDF](https://discovery.ucl.ac.uk/id/eprint/13395/):
  - The model needs "many links between cells tuned to very different, e.g., orthogonal, orientations", unlike V1-style iso-orientation links.
  - Conduction latency is fixed at 8–10 ms regardless of distance.
- Result: BOWN signals emerge 2–3 membrane time constants (≈20–30 ms) after onset for squares 1×, 2× and 3× the longest connection length (10 units). The signal is weaker for larger squares, matching Zhou et al. (2000), and it rises faster at corners — [UCL PDF](https://discovery.ucl.ac.uk/id/eprint/13395/)
- Critique: later work argues that propagation through local V2 interactions predicts latencies growing with figure size. This is the motivation for Craft et al. (2007) — [Craft et al. abstract, PubMed](https://pubmed.ncbi.nlm.nih.gov/17442769/)

**Craft, Schütze, Niebur & von der Heydt (2007): grouping-cell model**
- Citation: Craft E, Schütze H, Niebur E, von der Heydt R (2007). "A neural model of figure–ground organization." *Journal of Neurophysiology* 97(6):4310–4326. DOI 10.1152/jn.00203.2007 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/17442769/)
- Abstract: border ownership is produced by "dedicated neural circuits for contour grouping that produce border-ownership assignment and also provide handles for mechanisms of selective attention". The model "makes predictions about the hypothetical grouping circuits and the role of feedback between cortical areas" — [PubMed](https://pubmed.ncbi.nlm.nih.gov/17442769/)

**Mihalas, Dong, von der Heydt & Niebur (2011): attention through grouping neurons**
- Citation: Mihalas S, Dong Y, von der Heydt R, Niebur E (2011). "Mechanisms of perceptual organization provide auto-zoom and auto-localization for attention to objects." *PNAS* 108:7583–7588. DOI 10.1073/pnas.1014655108 — [DOI](https://doi.org/10.1073/pnas.1014655108)
- Key features: "grouping neurons integrating local features into coherent tentative objects, excitatory feedback to the same local feature neurons that caused grouping neuron activation, and inhibition between incompatible interpretations both at the local feature level and at the object representation level" — [DOI](https://doi.org/10.1073/pnas.1014655108)

**Hu, von der Heydt & Niebur (2019): image-computable descendant**
- Citation: Hu B, von der Heydt R, Niebur E (2019). "Figure-ground organization in natural scenes: performance of a recurrent neural model compared with neurons of area V2." *eNeuro* 6(3):ENEURO.0479-18.2019. DOI 10.1523/ENEURO.0479-18.2019 — [PDF](https://www.eneuro.org/content/eneuro/6/3/ENEURO.0479-18.2019.full.pdf)
- Architecture — [eNeuro PDF](https://www.eneuro.org/content/eneuro/6/3/ENEURO.0479-18.2019.full.pdf):
  - S cells cover 8 orientations × 2 contrast polarities, plus colour-opponent channels.
  - Each S cell feeds a pair of B cells with opposite side-of-figure preference.
  - G cells integrate B cells "in a roughly annular fashion". The kernel is v_θ(x,y) ∝ exp(cos(atan(y/x) − θ + π/2)) · exp(−(√(x²+y²) − R0)) / (2π I0(·)), a von Mises-like angular profile on a ring of radius R0 = 2 px.
  - The image pyramid spans 5 octaves in √2 steps, so the effective G radius grows with level.
  - G→B feedback is modulatory: facilitatory from the preferred side and suppressive from the non-preferred side. Opposing B cells compete indirectly through G.
  - 2–3 feedforward/feedback iterations suffice.
- Results — [eNeuro PDF](https://www.eneuro.org/content/eneuro/6/3/ENEURO.0479-18.2019.full.pdf):
  - BSDS-500 contour F-score 0.64 (0.65 with the per-image best scale). The cited state-of-the-art models reach 0.73 and humans 0.80.
  - Figure-ground mean accuracy 71.5%, against human 83.9%, SRF 74.7%, Global-CRF 68.9% and 2.1D-CRF 69.1%.
  - The timing and consistency of border-ownership signals agree with V2 recordings.
- Limitation relevant to crossings: "Only the signal strength at the optimal orientation at each spatial location is used as input to the network". This simplification discards the second orientation present at crossings — [eNeuro PDF](https://www.eneuro.org/content/eneuro/6/3/ENEURO.0479-18.2019.full.pdf)

**Sakai & Nishimura (2006): surround-modulation model**
- Citations:
  - Sakai K, Nishimura H (2006). "Surrounding suppression and facilitation in the determination of border ownership." *Journal of Cognitive Neuroscience* 18(4):562–579. DOI 10.1162/jocn.2006.18.4.562 — [DOI](https://doi.org/10.1162/jocn.2006.18.4.562)
  - Nishimura H, Sakai K (2005). *Neurocomputing* 65–66:77–83. DOI 10.1016/j.neucom.2004.10.057 — [DOI](https://doi.org/10.1016/j.neucom.2004.10.057)
- Findings: border ownership arises from asymmetric surround regions relative to the classical receptive field. In a population study of hundreds of surround organizations, "most of the asymmetric, iso-orientation suppressive regions, regardless of position or size, lead to surprisingly high consistency in the direction of ownership". Combining iso-orientation suppression with cross-orientation facilitation gave "both high robustness and consistency". The model reproduced the variety, size invariance and contrast co-selectivity of V2 border-ownership neurons — [DOI](https://doi.org/10.1162/jocn.2006.18.4.562)

**Other ownership models**
- Sajda P, Finkel LH (1995). "Intermediate-level visual representations and the construction of surface perception." *Journal of Cognitive Neuroscience* 7(2):267–291. DOI 10.1162/jocn.1995.7.2.267. A network that represents surfaces through "ownership", a selective binding of contours and regions, interacting with depth, transparency and completion — [DOI](https://doi.org/10.1162/jocn.1995.7.2.267)
- Kogo N, Strecha C, Van Gool L, Wagemans J (2010). "Surface construction by a 2-D differentiation–integration process: a neurocomputational model for perceived border ownership, depth, and lightness in Kanizsa figures." *Psychological Review* 117:406–439. DOI 10.1037/a0019076. Only the title and DOI were verified — [DOI](https://doi.org/10.1037/a0019076)

### Inferences
- **Limited direct use for vessels.** Border-ownership models are mostly not directly useful for vessel junction classification. Vessels are thin bars, not region borders, and "occlusion" in a 2D fundus image is an X-crossing of two transparent-looking tubes. Zhaoping's T-junction rule, however, is a precise template for a fixed-weight cross-orientation facilitation/suppression rule. Its sign depends on whether one segment "terminates against" another (stem) or passes (top). This is the same logic needed to tell a T-type branch from a crossing.
- **Size-independent pooling.** The annular G-cell idea can be repurposed for vessels. A G cell centred on a candidate junction, with an annulus of radius ≈ 2–3 vessel widths, could count how many oriented vessel segments enter the junction and from which directions. Three inputs suggests a bifurcation; four, in two collinear opposite pairs, suggests a crossing. Pooling over a ring rather than propagating gives latency independent of size.
- **Keep all orientations.** The "single optimal orientation per pixel" simplification in Hu et al. (2019) must be avoided in any crossing-aware design. All orientation channels must be kept at each position, as orientation scores do.

### Gaps
- The Craft et al. (2007) full text could not be retrieved; the publisher returned 403 and no repository PDF was found. Its exact G-cell radii, scale set, and any treatment of T-junctions or end-stopping are unverified. The architecture above comes from its descendant, Hu et al. (2019).
- No border-ownership model was found that is evaluated on junction detection or junction classification benchmarks.

---

## 4. Geometry of the visual cortex as SE(2) / orientation scores (Zucker; Petitot; Citti & Sarti; Duits & Franken; Bekkers & Duits; Abbasi-Sureshjani et al.)

### Takeaway
"Lifting" an image to positions × orientations (SE(2)), and optionally curvature, is the most principled way to keep crossing curves separate. At a crossing, the two vessels occupy different orientation layers, so diffusion, tracking or grouping kernels that move along the lifted curve ("left-invariant", "sub-Riemannian") pass straight through.

The most vessel-specific evidence comes from the Eindhoven group:
- Bekkers et al. (2014) detect junctions as extra peaks in a directional orientation column and classify bifurcation versus crossing. In 495 junctions from 9 HRF images, 109/114 crossings (95.6%) and 290/381 bifurcations (76.1%) were correct.
- Abbasi-Sureshjani et al. (2018) add curvature as a fifth dimension. Junctions show high curvature (abrupt orientation change), whereas crossings keep low curvature. On 274 patches, 76.0% were grouped correctly.

Invertible orientation scores built from cake wavelets, and closed-form SE(2) kernels, are fixed and reusable.

### Cited Findings
**Zucker and colleagues: curve inference and good continuation**
- Citation: Parent P, Zucker SW (1989). "Trace inference, curvature consistency, and curve detection." *IEEE TPAMI* 11(8):823–839. DOI 10.1109/34.31445 — [DOI](https://doi.org/10.1109/34.31445)
- Method, from secondary descriptions: relaxation labeling over discrete tangent (orientation) and curvature labels, with compatibilities from co-circularity and curvature consistency — [secondary summary (Hacettepe report)](https://ee.hacettepe.edu.tr/~eyuksel/Publications/InferenceReport.pdf); [co-circularity description, US patent 7130484](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7130484)
- Citation: Zucker SW, Dobbins A, Iverson L (1989). "Two stages of curve detection suggest two styles of visual computation." *Neural Computation* 1(1):68–81. DOI 10.1162/neco.1989.1.1.68
  - Stage 1 is a local description with hypotheses "represented explicitly and coarsely in a fixed, preconfigured architecture".
  - Stage 2 is global, with hypotheses "represented implicitly and more finely in a dynamically constructed architecture".
  - [DOI](https://doi.org/10.1162/neco.1989.1.1.68)
- Citation: Iverson LA, Zucker SW (1995). "Logical/linear operators for image curves." *IEEE TPAMI* 17(10):982–996. DOI 10.1109/34.464562. Only bibliographic details were verified — [DOI](https://doi.org/10.1109/34.464562)
- Citation: Ben-Shahar O, Zucker SW (2004). "Geometrical computations explain projection patterns of long-range horizontal connections in visual cortex." *Neural Computation* 16(3):445–476. DOI 10.1162/089976604772744866
  - Long-range projection fields are derived from differential geometry, with curvature(s) entering "in a fundamental way".
  - The model "quantitatively predicts the … spread in projection distribution, its nonmonotonic variance, and the differences found among individual neurons".
  - It argues that texture or shading continuation is an alternative function of these connections.
  - [DOI](https://doi.org/10.1162/089976604772744866)

**Petitot (2003) and Citti & Sarti (2006): sub-Riemannian models of V1**
- Citations:
  - Petitot J (2003). "The neurogeometry of pinwheels as a sub-Riemannian contact structure." *Journal of Physiology–Paris* 97:265–309. DOI 10.1016/j.jphysparis.2003.10.010 — [DOI](https://doi.org/10.1016/j.jphysparis.2003.10.010)
  - Citti G, Sarti A (2006). "A cortical based model of perceptual completion in the roto-translation space." *Journal of Mathematical Imaging and Vision* 24(3):307–326. DOI 10.1007/s10851-005-3630-2 — [DOI](https://doi.org/10.1007/s10851-005-3630-2)
- Citti–Sarti model, as summarized by the same authors in a 2008 follow-up:
  - Completion "is fully performed in the rototranslation group R² × S¹, allowing the simultaneous reconstruction of occluding and occluded objects".
  - The image is lifted to a surface, an occlusion is treated as a hole, and the hole is completed with a minimal surface. Its projection recovers the Masnou–Morel elastica level lines.
  - It extends Petitot & Tondut's (1999) Heisenberg-group model.
  - [Sanguinetti, Citti, Sarti 2008, VISAPP PDF](https://www.scitepress.org/Papers/2008/10758/10758.pdf)
- A later extension models illusory contours as sub-Riemannian geodesics in a metric modulated by simple-cell responses (Franceschiello, Mashtakov, Citti, Sarti 2019, *Differential Geometry and its Applications* 65:55–77, DOI 10.1016/j.difgeo.2019.03.007) — [DOI](https://doi.org/10.1016/j.difgeo.2019.03.007)

**Invertible orientation scores and cake wavelets (Duits et al.)**
- Citation: Duits R, Felsberg M, Granlund G, ter Haar Romeny B (2007). "Image analysis and reconstruction using a wavelet transform constructed from a reducible representation of the Euclidean motion group." *IJCV* 72:79–102. DOI 10.1007/s11263-006-8894-5 — [DOI](https://doi.org/10.1007/s11263-006-8894-5)
- Cake-wavelet construction (as used in Bekkers et al. 2014):
  - The wavelets are built in the Fourier domain from B-spline "pieces of cake" in the angular direction and windowed spatially by a Gaussian.
  - They are quadrature filters. The real part responds to lines and the imaginary part to edges.
  - They satisfy M_ψ ≈ 1 inside a disk of radius ϱ near Nyquist, so the transform is stably invertible ("proper wavelets").
  - Typically N_o = 36 orientations are used.
  - [arXiv 1212.3530](https://arxiv.org/abs/1212.3530)
- Single-sided (directional) wavelets: to "distinguish between π symmetries and 2π symmetries … and to be able to handle bifurcations", the score is split into forward and backward parts, U = U⁺ + U⁻. This uses ψ± = w(±x)ψ with an erf-shaped window w(x) = ½ + ½ erf(x) — [arXiv 1212.3530](https://arxiv.org/abs/1212.3530)

**Franken & Duits: crossing-preserving coherence-enhancing diffusion (CED-OS)**
- Citations:
  - Franken E, Duits R (2009). "Crossing-preserving coherence-enhancing diffusion on invertible orientation scores." *IJCV* 85(3):253–278. DOI 10.1007/s11263-009-0213-5 — [DOI](https://doi.org/10.1007/s11263-009-0213-5)
  - Companion: Duits R, Franken E (2010). "Left-invariant parabolic evolutions on SE(2) and contour enhancement via invertible orientation scores, Part II: nonlinear left-invariant diffusions on invertible orientation scores." *Quarterly of Applied Mathematics* 68:293–331. DOI 10.1090/S0033-569X-10-01173-3 — [DOI](https://doi.org/10.1090/S0033-569X-10-01173-3)
- CED-OS method: coherence-enhancing diffusion on the orientation score with left-invariant diffusion on the Euclidean motion group. The diffusion is steered by local features: curvature, deviation from horizontality and orientation confidence. Two explicit finite-difference schemes are given, with stability analysis. Standard image-domain CED "struggles" with crossings — [search summary of Springer abstract](https://link.springer.com/doi/10.1007/s11263-009-0213-5)
- Part II: these diffusions "lead to clear improvements of the celebrated standard 'coherence enhancing diffusion' equations on images as they can enhance images with crossing contours", using "optimized local coordinate systems attached to an orientation score, allowing us to include local features such as adaptive curvature" — [QAM abstract](https://doi.org/10.1090/S0033-569X-10-01173-3)
- Related:
  - Franken E, Duits R, ter Haar Romeny B (2007). "Curvature estimation for enhancement of crossing curves." *ICCV 2007*. DOI 10.1109/ICCV.2007.4409145 — [DOI](https://doi.org/10.1109/ICCV.2007.4409145)
  - A 3D extension (CEDOS on SE(3)) appeared in *J Math Imaging Vis* (2018), open access — [PMC6413631](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6413631/)

**Hannink, Duits & Bekkers (2014): crossing-preserving multiscale vesselness**
- Citation: Hannink J, Duits R, Bekkers E (2014). "Crossing-preserving multi-scale vesselness." *MICCAI 2014*, LNCS 8674:603–610. DOI 10.1007/978-3-319-10470-6_75 — [arXiv 1402.4963](https://arxiv.org/pdf/1402.4963)
- Method: the Frangi filter "cannot properly enhance vessels throughout crossings or bifurcations". Crossings are disentangled into orientation layers, and a Frangi-equivalent is applied on SE(2) or the scale-orientation group, with 5 scales {1.5, 2.4, 3.8, 6.0, 9.5} px and 12 orientations — [arXiv](https://arxiv.org/pdf/1402.4963)
- Results on the HRF dataset (15 images each for healthy, DR and glaucoma), at threshold t = 0.05:

  | Group | Sensitivity | Specificity | Accuracy |
  |---|---|---|---|
  | Healthy | 0.786 | 0.988 | 0.969 |
  | Diabetic retinopathy | 0.811 | 0.963 | 0.953 |
  | Glaucoma | 0.641 | 0.988 | 0.960 |

  The method was "considerably better" than multi-scale Frangi — [arXiv](https://arxiv.org/pdf/1402.4963)

**Bekkers, Duits, Berendschot & ter Haar Romeny (2014): ETOS tracking with junction detection and classification**
- Citation: Bekkers E, Duits R, Berendschot T, ter Haar Romeny B (2014). "A multi-orientation analysis approach to retinal vessel tracking." *Journal of Mathematical Imaging and Vision* 49(3):583–610. DOI 10.1007/s10851-013-0488-6 — [arXiv 1212.3530](https://arxiv.org/abs/1212.3530)
- Tracking: the ETOS (Edge Tracking based on Orientation Scores) tracker runs in the invertible orientation score. Each iteration optimizes centre position, orientation and scale. Parameters: step 2 px, scan line 40 px, N_o = 36, edge-envelope σ = 3 — [arXiv](https://arxiv.org/abs/1212.3530)
- Junction detection rule: "At a point on a straight vessel the modulus of the directional orientation column ideally contains two local maxima; one in the positive direction, say θ+, and one (with a π phase difference) in the negative direction, say θ−. A candidate vessel junction point at the left edge is detected whenever there exists another local maximum in between θ+ and θ−", and similarly on the right edge — [arXiv](https://arxiv.org/abs/1212.3530)
- Grouping candidates: candidates are clustered by position (distance < mean width), then by orientation (number of histogram peaks), then merged — [arXiv](https://arxiv.org/abs/1212.3530)
- Classification: "To classify between junctions and bifurcations we check for alignment of local orientations in SE(2) via the sub-Riemannian metric". Final labels also use proximity to other junctions, orientation and width. A crossing is found as overlapping tracked segments — [arXiv](https://arxiv.org/abs/1212.3530)
- Results (Table 2): 9 HRF images (healthy, diabetic, glaucoma) gave 495 junctions (381 bifurcations, 114 crossings):
  - Bifurcations: 290 (76.12%) correct; 71 (18.64%) were actually part of a crossing (E1); 20 (5.25%) were false vessels (E2).
  - Crossings: 109/114 (95.61%) correct.
  - By group, crossings were 100% correct in healthy and glaucoma images and 88.10% in diabetic images; bifurcations were 81.82%, 73.03% and 74.79% respectively.
  - The text gives the precision rates as 76.54% and 96.03%, which differ slightly from the table.
  - "Each miss-classified crossing introduces two E1 errors."
  - [arXiv](https://arxiv.org/abs/1212.3530)
- Comparison: invertible (cake) scores beat non-invertible Gabor scores. A fast multiscale centreline variant (CTOS) "is less stable at critical vessel points (crossings and parallel vessels)" — [arXiv](https://arxiv.org/abs/1212.3530)

**Data-driven sub-Riemannian geodesics and related**
- Citation: Bekkers EJ, Duits R, Mashtakov A, Sanguinetti GR (2015). "A PDE approach to data-driven sub-Riemannian geodesics in SE(2)." *SIAM Journal on Imaging Sciences* 8(4):2740–2770. DOI 10.1137/15M1018460 — [DOI](https://doi.org/10.1137/15M1018460)
- Method: a wavefront propagation (HJB viscosity solution) computes the SR distance in SE(2) with an image-based cost C ∈ [δ, 1]. Backtracking gives geodesics, and Maxwell and cusp points are computed. "Trackings of elongated structures in retinal and synthetic images show that our line tracking generically deals with crossings" — [DOI](https://doi.org/10.1137/15M1018460)
- Related papers:
  - Sanguinetti G, Bekkers E, Duits R, Janssen M, Mashtakov A, Mirebeau J-M (2015). "Sub-Riemannian fast marching in SE(2)." LNCS (CIARP), DOI 10.1007/978-3-319-25751-8_44 — [DOI](https://doi.org/10.1007/978-3-319-25751-8_44)
  - Duits R, Meesters S, Mirebeau J-M, Portegies J (2018). "Optimal paths for variants of the 2D and 3D Reeds–Shepp car with applications in image analysis." *JMIV* 60:816–848, DOI 10.1007/s10851-018-0795-z — [DOI](https://doi.org/10.1007/s10851-018-0795-z)
  - Duits R, Janssen M, Hannink J, Sanguinetti G (2016). "Locally adaptive frames in the roto-translation group and their applications in medical imaging." *JMIV* 56:367–402, DOI 10.1007/s10851-016-0641-0 — [DOI](https://doi.org/10.1007/s10851-016-0641-0)
  - Zhang J, Dashtbozorg B, Bekkers E, Pluim J, Duits R, ter Haar Romeny B (2016). "Robust retinal vessel segmentation via locally adaptive derivative frames in orientation scores." *IEEE TMI* 35:2631–2644, DOI 10.1109/TMI.2016.2587062 — [DOI](https://doi.org/10.1109/TMI.2016.2587062)

**Abbasi-Sureshjani, Favali, Citti, Sarti & ter Haar Romeny (2018): 5D position–orientation–curvature–intensity kernel**
- Citation: Abbasi-Sureshjani S, Favali M, Citti G, Sarti A, ter Haar Romeny BM (2018). "Curvature integration in a 5D kernel for extracting vessel connections in retinal images." *IEEE Transactions on Image Processing* 27(2):606–621. DOI 10.1109/TIP.2017.2761543 — [arXiv 1608.08049](https://arxiv.org/abs/1608.08049)
- Lifting:
  - A lifted curve is γ(t) = (x, y, θ, f, κ), with (ẋ, ẏ) = (cos θ, sin θ) and θ̇ = κ.
  - The horizontal vector fields are Y1 = (cos θ, sin θ, κ, 0, 0), Y4 = ∂_f (intensity) and Y5 = ∂_κ (curvature).
  - Connectivity is the fundamental solution of the Fokker–Planck equation ∂_t p = (σ5²/2) ∂²_κ p + (σ4²/2) ∂²_f p − Y1 p.
  - That is, transport along Y1 with diffusion in curvature (which sets the rate of θ change) and in intensity.
  - [arXiv](https://arxiv.org/abs/1608.08049)
- Kernel and grouping:
  - The kernel is w5 = exp(−(κ−κ′)²/σ_κ²) · exp(−(f−f′)²/σ_int²) · ½[Γ_κ + Γ_κ′]. The 3D sections at fixed κ are estimated by Monte-Carlo with n = 100,000 random paths.
  - The affinity matrix A_ij = w5(·,·) feeds self-tuning spectral clustering (Zelnik-Manor & Perona), which chooses the number of clusters automatically.
  - Parameters: n_θ = 18 orientations, curvature step 0.05, σ_κ = 1.
  - Orientation comes from the invertible orientation score. Curvature at every (x, θ) comes from exponential-curve fits on the Gaussian Hessian in SE(2), taken at the scale of maximum vesselness. This avoids curve fitting at junctions, where "it is not possible to fit a curve".
  - [arXiv](https://arxiv.org/abs/1608.08049)
- Crossing versus junction mechanism: "the bifurcations are grouped with the parent vessel, but at crossovers with small crossing angles … the vessels are totally separated. The main reason is that the curvature at junction points is high (because of sudden change of orientation), while for crossings the orientation for individual vessels changes only slightly" — [arXiv](https://arxiv.org/abs/1608.08049)
- Evaluation: phantom images (201×201) and IOSTAR SLO patches (images downsampled to 512×512) in categories A (crossings), B (bifurcations), C (close parallel vessels), D (bifurcations + crossings) and E (multiple nearby bifurcations), with hard versions A1–E1. Correct Detection Rate (CDR) per category (Table I) — [arXiv](https://arxiv.org/abs/1608.08049):

  | Category | CDR |
  |---|---|
  | A | 81.25% |
  | B | 71.43% |
  | C | 85.42% |
  | D | 77.42% |
  | E | 73.68% |
  | A1 | 66.67% |
  | B1 | 65.22% |
  | C1 | 85.37% |
  | D1 | 81.48% |
  | E1 | 72.22% |
  | **All (274 patches)** | **76.02%** (Q_clust 0.9942) |

  Mean times per patch: kernel 43 s, affinity 18 s, clustering 0.9 s.
- Limitations stated — [arXiv](https://arxiv.org/abs/1608.08049):
  - Very high curvature combined with co-circularity and co-linearity defeats the clustering.
  - A bifurcation almost orthogonal to its parent was clustered as a crossing.
  - It fails when "the measured curvature or orientations are not accurate enough".
  - Only small patches were analysed; the authors suggest label propagation for full images.
  - It needs a binary segmentation to define the points of interest.
- Predecessor (4D kernel, no curvature): Favali M, Abbasi-Sureshjani S, ter Haar Romeny B, Sarti A (2016). "Analysis of vessel connectivities in retinal images by cortically inspired spectral clustering." *JMIV* 56:158–172. DOI 10.1007/s10851-016-0640-1 — [DOI](https://doi.org/10.1007/s10851-016-0640-1)
- Related junction paper: Abbasi-Sureshjani S, Smit-Ockeloen I, Bekkers E, Dashtbozorg B, ter Haar Romeny B (2016). "Automatic detection of vascular bifurcations and crossings in retinal images using orientation scores." *IEEE ISBI 2016*, pp. 189–192. DOI 10.1109/ISBI.2016.7493241 — [DOI](https://doi.org/10.1109/ISBI.2016.7493241)

**Learned descendant (for contrast only)**
- Smets B, Portegies J, Bekkers E, Duits R (2022/2023). "PDE-based group equivariant convolutional neural networks." *JMIV* 65:209–239. DOI 10.1007/s10851-022-01114-x. A trainable network built from SE(2) convection, diffusion and morphological PDE layers; only bibliographic details were verified — [DOI](https://doi.org/10.1007/s10851-022-01114-x)

### Inferences
- **Recommended fixed-weight backbone.** Use a cake-wavelet orientation score with N_o ≈ 12–36. Use the single-sided split U⁺/U⁻ so the network knows which direction a branch leaves the junction.
- **Junction detection in the orientation column.** Count peaks in the directional orientation column. A straight vessel shows 2 opposite peaks (θ, θ+π). A bifurcation shows 3 peaks with no requirement of opposition. A crossing shows 4 peaks forming 2 opposite pairs. This is a direct reading of the Bekkers peak rule, extended by me to the full column.
- **Classification by alignment.** At each candidate, check whether each incoming direction has an "aligned" outgoing partner under the SE(2)/sub-Riemannian distance (low curvature continuation, as in Bekkers 2014). Two aligned pairs means a crossing; one aligned pair plus one extra means a T-branch; no aligned pairs means a Y-bifurcation.
- **Curvature as a discriminator.** Curvature, from fixed exponential-curve fits or the Rodríguez-Sánchez end-stopped curvature classes, separates small-angle crossings (low κ continuations) from branches (high κ at the junction).
- **Fixed propagation kernels.** CED-OS and sub-Riemannian geodesic tracking show that propagation in the lifted space passes through crossings without leaking between orientation layers. Fixed SE(2) Green's functions (diffusion, convection or Fokker–Planck kernels) are therefore suitable fixed-weight lateral connections. Leakage at small crossing angles is controlled by angular resolution, so N_o must exceed π divided by the smallest crossing angle of interest, with a safety factor.
- **Error pattern to anticipate.** Bekkers' error profile suggests the hard case is bifurcations near crossings (E1 errors, 18.6%), not crossings themselves. Crossings were 95.6% correct.

### Gaps
- Full text of Franken & Duits (2009) IJCV was not accessible; the publisher redirected to login. No quantitative results (for example on collagen or retinal images) are quoted, only the abstract-level method description.
- The results of Abbasi-Sureshjani et al. (2016, ISBI) on bifurcation and crossing detection with orientation scores could not be retrieved. The IEEE page returned no content.
- Parent & Zucker (1989) was described only through secondary sources. How their relaxation labeling handles two orientations at one location (a crossing) could not be confirmed.
- Petitot (2003) and Citti & Sarti (2006) full texts were not read. Neither reports junction or crossing classification results; they model completion.

---

## 5. Tensor voting and stochastic completion fields

### Takeaway
Tensor voting is fixed-weight. Its only free parameter is scale. Junctions appear as the isotropic ("ball") part of the accumulated tensor, which is high where votes with many orientations meet. First-order (polarity) voting adds endpoint detection and enables T/L/endpoint labelling (Mordohai & Medioni 2004).

Stochastic completion fields compute the probability that a random-walk contour passes through each position and orientation. They do this as the product of a source field and a sink field, each a convolution on (x, y, θ). Because they live in the lifted space, two crossing curves can coexist at one pixel in different θ.

Neither framework was found evaluated on vessel junctions.

### Cited Findings
**Tensor voting**
- Citations:
  - Medioni G, Lee M-S, Tang C-K (2000). *A Computational Framework for Segmentation and Grouping.* Elsevier — [chapter DOI](https://doi.org/10.1016/B978-044450353-4/50003-2)
  - Medioni G, Tang C-K, Lee M-S (c. 2000). "Tensor voting: theory and applications." The retrieved PDF does not state the venue — [PDF](https://www.cse.unr.edu/~bebis/CS773C/ObjectRecognition/Papers/Medioni00.pdf)
- Encoding and saliencies:
  - A token without orientation is encoded as a unit isotropic tensor (a "ball"); an oriented token is encoded as a "stick".
  - The accumulated tensor decomposes as S = (λ1−λ2) ê1ê1ᵀ + (λ2−λ3)(ê1ê1ᵀ + ê2ê2ᵀ) + λ3(ê1ê1ᵀ + ê2ê2ᵀ + ê3ê3ᵀ), i.e. stick, plate and ball.
  - In 3D: point-ness (junction) saliency = λ3; curve-ness saliency = λ2−λ3 with orientation ê3; surface-ness = λ1−λ2.
  - "In 2-D, there is no surface-ness, and curve-ness is expressed by ê1 for the tangent orientation, and by λ1-λ2 for curve saliency."
  - The dense tensor map is "decomposed into a junction map and a curve map". Junctions are local maxima of the junction saliency map (JMap).
  - [Medioni et al. 2000 PDF](https://www.cse.unr.edu/~bebis/CS773C/ObjectRecognition/Papers/Medioni00.pdf)
- Voting field:
  - A stick vote is cast "normal to the smoothest circular path connecting the voter and receiver". Its strength decays with distance along the path and with the path's curvature.
  - The scale ρ "essentially controls the size of the voting neighborhood and the strength of the votes" and is the only free parameter.
  - Limitations: one scale for all the data causes problems with uneven density, "a trade-off between unwanted smoothing in dense areas and weak enforcement of continuity in sparse areas"; multiscale voting was "under investigation".
  - A first-order voting scheme "can detect region boundaries … and endpoints of curves".
  - [Mordohai, CVonline tensor-voting page](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORDOHAI/tensorvoting.html)
- Junction classification: Mordohai P, Medioni G (2004). "Junction inference and classification for figure completion using tensor voting." *CVPR Workshops 2004*, p. 56. DOI 10.1109/CVPR.2004.376
  - The framework works from tokens or binary images and labels junctions as T, L and endpoints using first- and second-order tensor voting. "The first-order component … makes the inference of endpoints and the labeling of junctions possible".
  - It chooses between modal and amodal completion. This is abstract-level only.
  - [mlanthology](https://mlanthology.org/cvpr/2004/mordohai2004cvpr-junction); [DOI](https://doi.org/10.1109/CVPR.2004.376)
- Other tensor-voting papers:
  - Guy G, Medioni G (1996). "Inferring global perceptual contours from local features." *IJCV* 20:113–133. DOI 10.1007/BF00144119 — [DOI](https://doi.org/10.1007/BF00144119)
  - Tang C-K, Medioni G (1998). "Integrated surface, curve and junction inference from sparse 3-D data sets." *ICCV 1998*, pp. 818–824, DOI 10.1109/ICCV.1998.710812 — [DOI](https://doi.org/10.1109/ICCV.1998.710812)
  - Mordohai P, Medioni G (2006). *Tensor Voting: A Perceptual Organization Approach to Computer Vision and Machine Learning*, Synthesis Lectures — [DOI](https://doi.org/10.1007/978-3-031-02242-5_2)

**Stochastic completion fields**
- Citation: Williams LR, Jacobs DW (1997). "Stochastic completion fields: a neural model of illusory contour shape and salience." *Neural Computation* 9(4):837–858. DOI 10.1162/neco.1997.9.4.837 — [DOI](https://doi.org/10.1162/neco.1997.9.4.837)
- Model: the prior on completion shape is "a random walk in a lattice whose points are positions and orientations in the image plane (i.e., the space that one can reasonably assume is represented by neurons of the mammalian visual cortex)". The probability that a path joining two fragments passes through a given (x, y, θ) "can be computed directly as the product of two vector-field convolutions". No relaxation is used. Maximum-likelihood paths are least-energy curves (elastica) — [DOI](https://doi.org/10.1162/neco.1997.9.4.837)
- Citation: Williams LR, Jacobs DW (1997). "Local parallel computation of stochastic completion fields." *Neural Computation* 9(4):859–881. DOI 10.1162/neco.1997.9.4.859
  - It gives a finite-difference scheme for the Fokker–Planck equation identified by Mumford, implementable in a sparse, locally connected network (search-result summary).
  - [DOI](https://doi.org/10.1162/neco.1997.9.4.859)
  - Mumford D (1994). "Elastica and computer vision." In *Algebraic Geometry and its Applications*, pp. 491–506. DOI 10.1007/978-1-4612-2628-4_31 — [DOI](https://doi.org/10.1007/978-1-4612-2628-4_31)
- Extensions:
  - Thornber & Williams (NeurIPS 1998) compute a scale-invariant distribution of closed contours — [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/1998/file/109d2dd3608f669ca17920c511c2a41e-Paper.pdf)
  - Zweck J, Williams LR (2004). "Euclidean group invariant computation of stochastic completion fields using shiftable-twistable functions." *JMIV* 21:135–154, DOI 10.1023/B:JMIV.0000035179.47895.bc — [DOI](https://doi.org/10.1023/B:JMIV.0000035179.47895.bc)

### Inferences
- **2D ball saliency for junctions.** For a 2D vessel map, ball saliency (λ2 of the 2×2 accumulated tensor) is a fixed-weight junction detector. It is the 2D analogue of the 3D JMap and is my inference, since the sources read give only the 3D definition. It signals "multiple orientations meet here" but, like circular variance, does not separate a crossing from a branch.
- **Polarity for classification.** First-order (polarity) votes measure whether support comes from one side (an endpoint or branch stem) or from both sides (continuation). Combining polarity per orientation with ball saliency gives a fixed rule set analogous to Weidenbacher's end-stop × bipole products.
- **Second-order collapse at crossings.** Second-order tensor voting collapses all orientations at a pixel into one 2×2 tensor. At an X-crossing the two orientations are averaged into a ball, so the individual tangents are lost. Orientation-score (SE(2)) representations keep them separate, which matters for tracing through crossings.
- **SCFs pass through crossings.** Stochastic completion fields, like CED-OS, act on (x, y, θ), so a straight path through a crossing keeps its own θ layer. A source field from a vessel end on one side and a sink field from the other side would "complete" through a crossing without being captured by the crossing vessel, except at small crossing angles where θ layers overlap. This is my inference; no SCF paper read here evaluates crossings.

### Gaps
- Full text of Mordohai & Medioni (2004) was not retrieved. The exact polarity-based rules for T versus L versus endpoint, and any quantitative results, are unknown.
- The exact tensor-voting decay formula, usually written exp(−(s² + cκ²)/σ²), could not be quoted from an accessible source. The CVonline page shows it only as an image.
- No tensor-voting or SCF study was found that evaluates vessel junction or crossing classification.

---

## 6. Incremental grouping and curve-tracing network models (Roelfsema 2006; growth-cone model; Jolicoeur; learned tracing networks)

### Takeaway
Roelfsema's framework separates fast feedforward "base grouping" from slow "incremental grouping". In incremental grouping, enhanced activity (an attention label) spreads along a curve through horizontal and feedback connections.

The "growth-cone" model is fitted to V1 recordings. It holds that the label spreads at about 50 ms per receptive field, in whichever cortical level has receptive fields that fit between the target and the distractor. Spread is slow when curves are close and fast when they are far apart.

None of the papers read specifies how propagation is gated at a crossing versus a branch point. Gestalt good continuation (collinearity) is the implied gating rule. A fixed-weight label-propagation layer in SE(2) would implement it naturally.

### Cited Findings
**Roelfsema (2006): base versus incremental grouping**
- Citation: Roelfsema PR (2006). "Cortical algorithms for perceptual grouping." *Annual Review of Neuroscience* 29:203–227. DOI 10.1146/annurev.neuro.29.051605.112939 — [DOI](https://doi.org/10.1146/annurev.neuro.29.051605.112939)
- Base groupings "are coded by single neurons tuned to multiple features" and computed rapidly by feedforward selectivity. Incremental grouping "enhances the responses of neurons coding features that are bound in perception, but it takes more time … because it relies also on horizontal and feedback connections". The enhancement is the neural correlate of object-based attention — [DOI](https://doi.org/10.1146/annurev.neuro.29.051605.112939)

**Pooresmaeili & Roelfsema (2014): growth-cone model**
- Citation: Pooresmaeili A, Roelfsema PR (2014). "A growth-cone model for the spread of object-based attention during contour grouping." *Current Biology* 24(24):2869–2877. DOI 10.1016/j.cub.2014.10.007 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/25456446/)
- Findings from V1 of monkeys tracing a target curve among distractors:
  - Response enhancement was "delayed for neurons with receptive fields farther from the start of the tracing process".
  - Propagation speed "was low if curves were nearby and … increased if curves were far apart".
  - The data fit an "attentional growth-cone" model in which enhancement spreads "in multiple visual cortical areas with different receptive field sizes at a speed of approximately 50 ms per receptive field".
  - [PubMed](https://pubmed.ncbi.nlm.nih.gov/25456446/)
- Inspiration: the growth-cone idea derives from earlier curve-tracing models (McCormick & Jolicoeur 1994). Larger receptive fields in higher areas "could speed up the grouping process" when curves are far apart. Size invariance is explained by grouping at multiple hierarchy levels — [Jeurissen et al. 2016, eLife PDF](https://cdn.elifesciences.org/articles/14320/elife-14320-v1.pdf)

**Jeurissen, Self & Roelfsema (2016): extension to 2D regions**
- Citation: Jeurissen D, Self MW, Roelfsema PR (2016). "Serial grouping of 2D-image regions with object-based attention in humans." *eLife* 5:e14320. DOI 10.7554/eLife.14320 — [PDF](https://cdn.elifesciences.org/articles/14320/elife-14320-v1.pdf)
- Grouping was compared under pixel-by-pixel, growth-cone and filling-in models. The growth-cone model, in which speed depends on the size of homogeneous regions, explained the data best — [eLife PDF](https://cdn.elifesciences.org/articles/14320/elife-14320-v1.pdf)

**Jolicoeur, Ullman & MacKay (1986): curve-tracing psychophysics**
- Citation: Jolicoeur P, Ullman S, MacKay M (1986). "Curve tracing: a possible basic operation in the perception of spatial relations." *Memory & Cognition* 14:129–140. DOI 10.3758/BF03198373 — [DOI](https://doi.org/10.3758/BF03198373)
- Curve tracing is serial, with reaction time increasing linearly with traced length (as summarized by Brosch et al. 2015) — [Brosch et al., PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4619762/)

**Learned tracing networks (contrast)**
- Brosch T, Neumann H, Roelfsema PR (2015). "Reinforcement learning of linking and tracing contours in recurrent neural networks." *PLoS Computational Biology* 11:e1004489. DOI 10.1371/journal.pcbi.1004489
  - RELEARNN trains recurrent networks by reward.
  - "Without horizontal connections, our networks did not learn the task for contours longer than three pixels".
  - Contour linking was learned in ≈15,000 trials.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4619762/)
- Linsley D, Kim J, Veerabadran V, Serre T (2018). "Learning long-range spatial dependencies with horizontal gated-recurrent units." CCN 2018 / NeurIPS 2018. DOI 10.32470/CCN.2018.1116-0. A learned recurrent "hGRU" for the Pathfinder curve-tracing task; only bibliographic details were verified — [DOI](https://doi.org/10.32470/CCN.2018.1116-0)
- A 2024 bioRxiv preprint proposes a learned multiscale incremental-grouping network — [bioRxiv PDF](https://www.biorxiv.org/content/10.1101/2024.06.17.599272.full.pdf)

### Inferences
- **Proposed gating rule for a fixed-weight tracer.**
  - Seed a label at a vessel segment in the orientation score.
  - Propagate it with a fixed SE(2) kernel: forward convection plus small angular diffusion, as in a stochastic completion field.
  - Gate it multiplicatively by the feedforward vessel evidence, using the same unit-bias modulatory form as Weidenbacher & Neumann.
  - At a crossing, the label stays in its θ layer and continues straight. The crossing vessel occupies a different θ layer that is not reached unless angular diffusion is large.
  - At a bifurcation, the daughter branches leave at moderate angles. The label should be allowed to spread into them, which needs either larger angular diffusion or an explicit "junction-detected → spread to all branches" rule.
  - This asymmetry is exactly the bifurcation-versus-crossing distinction, and its threshold is set by the angular diffusion constant. This is untested synthesis.
- **Multiscale shortcut.** The growth-cone finding suggests using coarser scales for fast propagation where vessels are far from other vessels, and dropping to fine scales near crossings and close parallel vessels.

### Gaps
- No incremental-grouping or growth-cone model was found that explicitly specifies or evaluates different gating rules at crossings versus branch points. The growth-cone paper's full text, with model equations and receptive-field sizes per area, was not accessible.
- No quantitative evaluation of these tracing models on vessel images or junction benchmarks was found.

---

## 7. Compositional shape/junction filters inspired by V4: COSFIRE (Azzopardi & Petkov)

### Takeaway
COSFIRE filters are configured, not trained by gradient descent, from a single user-chosen prototype. Configuration finds the oriented line or edge parts (Gabor responses) on concentric circles around a keypoint. The filter then takes a weighted geometric mean (soft AND) of blurred, shifted Gabor responses.

On DRIVE binary vessel maps, 6 rotation-, scale- and reflection-invariant filters detect bifurcations and crossings ("Y- or T-form bifurcations or cross overs") with 98.50% recall and 96.09% precision. The detector does not by itself separate bifurcations from crossings, but separate prototypes could be configured for each class. It is the most directly reusable fixed-weight junction detector found.

### Cited Findings
**Azzopardi & Petkov (2013): COSFIRE filters**
- Citation: Azzopardi G, Petkov N (2013). "Trainable COSFIRE filters for keypoint detection and pattern recognition." *IEEE TPAMI* 35(2):490–503. DOI 10.1109/TPAMI.2012.106 — [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf). Crossref lists the second author incorrectly; the paper is Azzopardi & Petkov.
- Inspiration: shape-selective V4 neurons selective for contour parts (Pasupathy & Connor). The blur growing with distance from the centre is motivated by V4 receptive-field size growing with eccentricity — [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf)
- Configuration:
  - Gabor responses g_{λ,θ} are normalized so the positive and negative parts each sum to 1, which makes the response to constant intensity zero. They are thresholded at fraction t1 of the maximum.
  - Along circles of radius ρ around the prototype point, the maximum over the (λ, θ) superposition is taken. Positions that are local maxima within an arc of π/8 become contour-part locations (ρ_i, φ_i).
  - At each location, every (λ, θ) with response > t2 = 0.75 of the maximum is kept. This gives the tuple set S_f = {(λ_i, θ_i, ρ_i, φ_i)}.
  - [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf)
- Response:
  - Each selected Gabor response is blurred with a Gaussian of σ = σ0 + αρ (a max-weighted blur over |x′|, |y′| ≤ 3σ) and shifted by ρ_i opposite to φ_i.
  - The output is the weighted geometric mean r = (Π s_i^{ω_i})^{1/Σω_i}, with ω_i = exp(−ρ_i² / 2σ̂²). The setting σ̂ = (−ρ_max² / (2 ln 0.5))^{1/2} gives weights from 1 at the centre to 0.5 at the periphery.
  - An optional threshold t3 can be applied.
  - α sets orientation tolerance: larger α gives broader tuning.
  - [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf)
- Invariances — [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf):
  - Rotation invariance takes the maximum over filters with θ_i + ψ and φ_i + ψ, for example 16 rotations.
  - Scale invariance takes the maximum over ρ_i·υ and λ_i·υ.
  - Reflection invariance takes the maximum with a mirrored tuple set.
  - Contrast-inversion invariance uses appropriate Gabor symmetry.
- Vessel application:
  - Input is a binary DRIVE segmentation, with vessel widths of 1–7 px.
  - The bank is symmetric (line) Gabor filters with 8 orientations and 5 wavelengths λ ∈ {4·(√2)^i, i = 0…4}, thresholded at t1 = 0.2. The prototype filter used ρ ∈ {0, 4, 10}.
  - On the training image (107 junctions), 4 filters gave 100% recall and 100% precision.
  - On 40 DRIVE images (>5,000 junctions), 4 filters gave R = 95.58% and P = 95.25%; 6 filters gave R = 98.50% and P = 96.09%. Each filter's t3 was set to the smallest value giving 100% precision on the training image, then offset by +0.05 or +0.07.
  - A single filter detected 4 junctions without invariance, 24 with rotation invariance, 34 with added scale invariance and 67 with added reflection invariance.
  - Earlier work [33] reported 95.82% recall.
  - [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf)
- Scope of "features": the ground-truth "features" are "Y- or T-form bifurcations or cross overs", i.e. bifurcations and crossings counted together — [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf)
- Cost: a 564×584 fundus image took under 45 s on a 3 GHz CPU with 6 fully invariant filters. MATLAB code is at matlabserver.cs.rug.nl — [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf)
- Limitations and notes from the authors — [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf):
  - The prototype is chosen by a user.
  - More ρ values make a filter more selective and less general.
  - t1 depends on contrast and noise.
  - Results were obtained on binary segmentations, not raw fundus images.
  - "In principle, all vascular bifurcations can be detected if a sufficient number of filters are configured".
- Related COSFIRE papers:
  - CBMS 2012 conference version: Azzopardi G, Petkov N. "Detection of retinal vascular bifurcations by rotation-, scale- and reflection-invariant COSFIRE filters." DOI 10.1109/CBMS.2012.6266338. It reports, with 5 filters, 98.57% recall at 95.37% precision on 40 DRIVE binary images with more than 5,000 bifurcations — [CBMS PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/26336/1/ConferencePaper_Detection%20of%20retinal%20vascular%20bifurcations%20by%20rotation-_scale-_and%20reflection-invariant%20COSFIRE%20filters.pdf); [DOI](https://doi.org/10.1109/CBMS.2012.6266338)
  - Azzopardi G, Petkov N (2013). "Automatic detection of vascular bifurcations in segmented retinal images using trainable COSFIRE filters." *Pattern Recognition Letters* 34(8):922–933. DOI 10.1016/j.patrec.2012.11.002 — [DOI](https://doi.org/10.1016/j.patrec.2012.11.002)
  - Azzopardi G, Strisciuglio N, Vento M, Petkov N (2015). "Trainable COSFIRE filters for vessel delineation with application to retinal images." *Medical Image Analysis* 19:46–57. DOI 10.1016/j.media.2014.08.002. This is B-COSFIRE, a bar-selective version for vessel segmentation — [DOI](https://doi.org/10.1016/j.media.2014.08.002)
  - Strisciuglio N, Azzopardi G, Vento M, Petkov N (2016). *Machine Vision and Applications* 27:1137–1149, DOI 10.1007/s00138-016-0781-7, on automatic selection of B-COSFIRE filters — [DOI](https://doi.org/10.1007/s00138-016-0781-7)
- The authors call the same design principle COSFIRE in engineering and CORF (Combination Of Receptive Fields) in neuroscience — [PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf)

### Inferences
- **Separate crossing and branch filters.** For crossing-versus-bifurcation classification, configure separate COSFIRE filters from crossing prototypes (4 arms in 2 collinear pairs) and bifurcation prototypes (3 arms). Use rotation and reflection invariance. Classify by which filter family wins after per-class t3 thresholds. Because the geometric mean is a soft AND, a crossing prototype will not respond fully to a 3-armed bifurcation (a missing arm drives the product toward 0). A bifurcation prototype, however, may respond to a subset of a crossing's arms. A winner-take-all or "explain-away" step, such as a crossing response suppressing bifurcation responses at the same site, is therefore needed. This is my inference.
- **COSFIRE as a fixed CNN layer.** COSFIRE maps exactly onto a fixed-weight network layer: Gabor bank → per-tuple blur (max-pooling with a Gaussian-weighted window) → spatial shift → log, weighted sum, exp (geometric mean) → max over rotations. Its parameters (σ0, α, t1, t2, t3, ρ set) are few and interpretable.

### Gaps
- The DRIVE numbers in Pattern Recognition Letters (2013) were not retrieved. The TPAMI and CBMS numbers differ slightly (98.50/96.09 against 98.57/95.37), which is consistent with different filter sets.
- The σ0 and α values used for the vessel application in TPAMI 2013 were not extracted. Values quoted elsewhere in the paper, such as σ0 = 0.67 and α = 0.04 for traffic signs, are application-specific.
- No COSFIRE paper was found that explicitly classifies bifurcation versus crossing with separate filter families and reports per-class accuracy.

---

## 8. Comparative evaluations on junction detection and classification benchmarks

### Takeaway
No common benchmark compares these neurally inspired models on junction detection or classification. Each is evaluated on its own data:
- ROC on synthetic or corner test images (Hansen & Neumann; Weidenbacher & Neumann);
- DRIVE binary maps (COSFIRE);
- HRF tracking models (Bekkers);
- IOSTAR patches (Abbasi-Sureshjani);
- BSDS figure-ground (Hu et al.).

On retinal vessels specifically, reported figures give a rough picture:
- Fixed-weight COSFIRE detection of junctions (class-agnostic): ~96–98% precision/recall on DRIVE binaries.
- Orientation-score tracking: ~96% correct crossings and ~76% correct bifurcations on HRF.
- A non-neural skeleton method: ~87–95% recall/precision per class.
- A trained CNN: only ~0.70–0.77 accuracy for detection and classification on colour fundus images.

These numbers are not directly comparable: the inputs (binary versus colour), datasets and metrics all differ.

### Cited Findings
- **Neural models against standard detectors:**
  - Hansen & Neumann (2002/2004): ROC superiority over Gaussian-curvature and structure-tensor detectors, and over a feedforward complex-cell model, on synthetic and camera images — [BMCV chapter](https://mijn.bsl.nl/chapter/10.1007/3-540-36181-2_2); [Neural Computation DOI](https://doi.org/10.1162/089976604773135087)
  - Weidenbacher & Neumann (2009): ROC superiority over Harris and over their own feedforward stage on a Smith & Brady-style corner test image with L/T/X ground truth. Detection only, not classification — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691604/)
  - Würtz & Lourens (1997/2000): a qualitative "compares favorably with known corner detectors" claim — [ICANN'97 PDF](https://www.ini.rub.de/upload/file/1470692858_650d4f644e49415163c7/rolf-icann97.pdf)
- **Retinal junctions, fixed-weight:**
  - COSFIRE: 6 filters, DRIVE binaries, R = 98.50%, P = 96.09% for bifurcations and crossings pooled — [TPAMI PDF](https://www.um.edu.mt/library/oar/bitstream/123456789/8375/1/PAMI2013.pdf)
  - Bekkers et al. (2014) ETOS, 9 HRF images: crossings 95.61%, bifurcations 76.12% correct — [arXiv](https://arxiv.org/abs/1212.3530)
  - Abbasi-Sureshjani et al. (2018): 76.02% correct grouping over 274 crossing, bifurcation and parallel-vessel patches — [arXiv](https://arxiv.org/abs/1608.08049)
- **Non-neural baseline:** Hamad H, Tegolo D, Valenti C (2014). "Automatic detection and classification of retinal vascular landmarks." *Image Analysis & Stereology* 33:189–200. DOI 10.5566/ias.1101
  - Pixels with ≥3 skeleton neighbours are detected, then classified.
  - On DRIVE and STARE, the average recall and precision were 95.4% and 87.1% for bifurcations, and 87.6% and 90.5% for crossovers.
  - [DOI](https://doi.org/10.5566/ias.1101)
- **Deep-learning contrast:** Pratt H, Williams BM, Ku JY, Vas C, McCann E, Al-Bander B, Zhao Y, Coenen F, Zheng Y (2017/2018). "Automatic detection and distinction of retinal vessel bifurcations and crossings in colour fundus photography." *Journal of Imaging* 4(1):4. DOI 10.3390/jimaging4010004
  - A CNN detects then classifies junctions.
  - Detection accuracy against three graders was 0.76, 0.76 and 0.77. Classification accuracies were "≥0.70".
  - It was tested on unseen IOSTAR images, with lower values in several configurations.
  - [DOI](https://doi.org/10.3390/jimaging4010004)
- **Non-neural natural-image junction benchmarks (contrast):**
  - Xia G-S, Delon J, Gousseau Y (2014). "Accurate junction detection and characterization in natural images." *IJCV* 106:31–56. DOI 10.1007/s11263-013-0640-1 — [DOI](https://doi.org/10.1007/s11263-013-0640-1)
  - Maire M, Arbeláez P, Fowlkes C, Malik J (2008). "Using contours to detect and localize junctions in natural images." *CVPR 2008*, DOI 10.1109/CVPR.2008.4587420 — [DOI](https://doi.org/10.1109/CVPR.2008.4587420)
  - Lindeberg T (1994). "Junction detection with automatic selection of detection scales and localization scales." *ICIP 1994*, DOI 10.1109/ICIP.1994.413244 — [DOI](https://doi.org/10.1109/ICIP.1994.413244)
- **Figure-ground benchmark of a grouping-cell model:** Hu et al. (2019), BSDS-500 contour F = 0.64 against humans 0.80; figure-ground accuracy 71.5% against humans 83.9% — [eNeuro PDF](https://www.eneuro.org/content/eneuro/6/3/ENEURO.0479-18.2019.full.pdf)
- **Related V1 contour-integration models (no junction evaluation found):**
  - Li Z (1998). "A neural model of contour integration in the primary visual cortex." *Neural Computation* 10:903–940, DOI 10.1162/089976698300017557 — [DOI](https://doi.org/10.1162/089976698300017557)
  - Yen SC, Finkel LH (1998). "Extraction of perceptually salient contours by striate cortical networks." *Vision Research* 38:719–741, DOI 10.1016/S0042-6989(97)00197-1 — [DOI](https://doi.org/10.1016/S0042-6989%2897%2900197-1)

### Inferences
- **The shared weak point.** Across fixed-weight vessel methods, crossings are reliably handled once orientations are disentangled (≈96% in Bekkers). The weak point is bifurcations near crossings and small-angle or near-orthogonal configurations, which appears in both Bekkers' E1 errors and Abbasi-Sureshjani's failure cases. A fixed-weight design should budget its complexity there. Examples are explicit "explain-away" competition between crossing and bifurcation hypotheses at the same site, and multiscale orientation scores to resolve close junctions.
- **The deep-learning gap.** The weak CNN numbers (Pratt et al.) relative to fixed-weight geometric methods suggest that, at least with small datasets, hand-designed orientation-score and COSFIRE pipelines are competitive or better for junction classification. The inputs differ, though, so this is suggestive only.

### Gaps
- No head-to-head benchmark was found that runs Heitger/Würtz keypoints, Hansen–Neumann circular variance, Weidenbacher T/L/X maps, tensor voting, COSFIRE and orientation-score methods on the same junction-classification data, such as DRIVE or IOSTAR junction annotations.
- No per-class (T/L/X/Y) classification accuracies were found for any of the cortical V1–V2 models.
- The IOSTAR and DRIVE junction annotations used by different groups may differ: COSFIRE used ground truth "defined by the authors". Cross-paper comparisons are therefore unreliable.
