# Neurophysiology of junction, corner, line-end, curvature and occlusion coding in primate visual cortex (for fixed-weight vessel-junction networks)

Source note: bibliographic metadata, abstracts and DOIs were retrieved from PubMed (NCBI E-utilities) unless another URL is given. Quantitative values come from PubMed abstracts unless marked "full text" (read from the PMC, eNeuro or PLoS page). Species and method tags: [Mq] macaque, [Hu] human, [Cat], [TS] tree shrew, [NWM] New World monkey (owl/squirrel monkey), [Model] computational. Cat, tree-shrew and model results are included only where they are foundational, and are flagged as such.

---

## 1. End-stopped / hypercomplex cells in V1 and V2: what do they respond to (line ends, corners, curvature), and how are they built?

### Takeaway
End-stopped ("hypercomplex") cells signal 2-D discontinuities: line ends, corners, short segments and high-curvature points. They occur in macaque V1 (mainly upper layers II/III and deep layers V/VI) and in more than half of V2 pale-stripe cells. Cat physiology and models show that end-stopping is formally a curvature detector: an excitatory oriented center minus excitation at offset "end zones" along the preferred axis. End-inhibition is generated partly by layer 6 → layer 4 projections (cat, inactivation) and partly by V2/V3 feedback (alert macaque, cooling). In macaque V1 the end-zone inhibition is selective for relative contrast sign, so end-stopped cells can tell junction contrast configurations apart (occlusion vs shadow or transparency) without explicit junction detectors.

### Cited Findings
**Classic descriptions**
- [Cat] Hubel & Wiesel introduced "hypercomplex" cells in cat areas 18/19. An optimally oriented line stops driving these cells when it extends beyond the activating region at one or both ends. Hubel & Wiesel proposed a hierarchical wiring: excitatory input from complex cell(s) at the RF center plus inhibitory input from complex cells whose fields sit at the end(s). They also described "higher-order hypercomplex" cells that respond to two orientations about 90° apart, which would make them corner-like. PubMed has no abstract for this paper, so this summary follows standard accounts of it — [Hubel DH & Wiesel TN (1965) "Receptive fields and functional architecture in two nonstriate visual areas (18 and 19) of the cat." J Neurophysiol 28:229–289](https://doi.org/10.1152/jn.1965.28.2.229)
- [Mq] Most macaque striate cells are simple, complex or hypercomplex. Layers II and upper III contain complex and hypercomplex cells, with simple cells "virtually absent". Layers V and VI contain mostly complex and hypercomplex cells. Simple cells sit deep in layer III and in IVA/IVB. Macaque RFs are smaller and more orientation-selective than cat RFs — [Hubel DH & Wiesel TN (1968) "Receptive fields and functional architecture of monkey striate cortex." J Physiol 195:215–243](https://doi.org/10.1113/jphysiol.1968.sp008455)

**Laminar origin and circuitry of end-inhibition**
- [Cat] Among simple cells, only those in layer 4 showed end-inhibition. Among standard complex cells, only those in layer 2+3 did. Layer-6 cells summated over very long slits (very long RFs) — [Gilbert CD (1977) "Laminar differences in receptive field properties of cells in cat primary visual cortex." J Physiol 268:391–421](https://doi.org/10.1113/jphysiol.1977.sp011863)
- [Cat] Inactivating layer 6 reversibly with GABA abolished end-inhibition in layer 4 and in layer 2+3 cells. Orientation and direction selectivity were unchanged. Interpretation: the long-RF layer 6 → layer 4 projection generates end-inhibition — [Bolz J & Gilbert CD (1986) "Generation of end-inhibition in the visual cortex via interlaminar connections." Nature 320:362–365](https://doi.org/10.1038/320362a0)
- [Mq] Area-summation tests fitted with a difference-of-Gaussians (DOG) model:
  - Summation extent was similar across layers except layer 6, which summed more extensively.
  - Length and width summation extents were about equal.
  - Surround suppression was strongest in layer 4B and weakest in layer 6.
  - A divisive-normalization model gave similar extents but a weaker estimated surround than the subtractive DOG — [Sceniak MP, Hawken MJ & Shapley R (2001) "Visual spatial characterization of macaque V1 neurons." J Neurophysiol 85:1873–1887](https://doi.org/10.1152/jn.2001.85.5.1873)
- [Mq] Spatial summation extent in V1 was on average 2.3× larger at low contrast, independently of surround inhibition. The preferred length (degree of end-stopping) therefore depends on contrast — [Sceniak MP, Ringach DL, Hawken MJ & Shapley R (1999) "Contrast's effect on spatial summation by macaque V1 neurons." Nat Neurosci 2:733–739](https://doi.org/10.1038/11197)
- [Mq, alert] Cooling V2 and V3 released V1 surround suppression (strong facilitation for large stimuli, at low and high contrast). "For extended contours, eliminating feedback had the effect of reducing end stopping." Effects were largest in strongly suppressed cells, and their timing matched the dynamics of surround suppression. Feedback is therefore one source of end-stopping in primate — [Nassi JJ, Lomber SG & Born RT (2013) "Corticocortical feedback contributes to surround suppression in V1 of the alert primate." J Neurosci 33:8504–8517](https://doi.org/10.1523/JNEUROSCI.5124-12.2013)
- [Mq] Spatial layout of the surround:
  - A surround at the preferred orientation always suppressed. An orthogonal surround suppressed less and sometimes facilitated.
  - Iso-oriented (parallel) surround patches suppressed most at the RF **ends**. Orthogonal surround patches suppressed most on the **flanks**.
  - At low center contrast, orthogonal surrounds became relatively more suppressive.
  - In effect, end-stopping is the end-zone part of an orientation-specific surround — [Cavanaugh JR, Bair W & Movshon JA (2002) "Selectivity and spatial distribution of signals from the receptive field surround in macaque V1 neurons." J Neurophysiol 88:2547–2556](https://doi.org/10.1152/jn.00693.2001)
- [Cat] Most suppressive surrounds were spatially asymmetric and came from a localized region, with a slight bias toward the RF end zones — [Walker GA, Ohzawa I & Freeman RD (1999) "Asymmetric suppression outside the classical receptive field of the visual cortex." J Neurosci 19:10536–10553](https://doi.org/10.1523/JNEUROSCI.19-23-10536.1999)

**End-stopping ≡ curvature selectivity**
- [Cat + Model] Model: the difference between two simple-cell responses produces end-stopping, and the difference varies in proportion to curvature. Physiology in cat area 17: end-stopped cells were curvature-selective and non-end-stopped cells were not; some were selective for the sign of curvature. Zucker's argument: plain orientation measurements give strong, wrong responses when an RF spans two distinct curves, and curvature estimates fix this — [Dobbins A, Zucker SW & Cynader MS (1987) "Endstopped neurons in the visual cortex as a substrate for calculating curvature." Nature 329:438–441](https://doi.org/10.1038/329438a0)
- [Model] Even-symmetric end-stopped models respond to thin curves over a range of orientations and curvatures, regardless of curvature sign. Odd-symmetric versions respond to thin and thick curves and are sign-selective for curved lines but not curved edges. The paper also analyzes responses at curve singularities — [Dobbins A, Zucker SW & Cynader MS (1989) "Endstopping and curvature." Vision Res 29:1371–1387](https://doi.org/10.1016/0042-6989(89)90193-4)
- [Cat] Cat area 17 had three response classes for curves and chevrons (angles):
  - Class 1: optimal for straight lines; similar responses to chevrons and curves.
  - Class 2: responds to all curvatures.
  - Class 3: responds only to large curvatures; about 2/3 to both signs and about 1/3 to one sign. Class 3 cells responded *differently* to chevrons than to curves.
  - The classes differed in length-response curves and in orientation bandwidth, and were distributed differently across layers — [Versavel M, Orban GA & Lagae L (1990) "Responses of visual cortical neurons to curved stimuli and chevrons." Vision Res 30:235–248](https://doi.org/10.1016/0042-6989(90)90039-n)
- [Mq] Utah-array recordings from V1, V4 and PIT in 6 monkeys. In every area, end-stopped sites tended to prefer curved over straight contours. Population curvature preference shifted along the hierarchy: V1 sites preferred straight Gabors, V4 sites preferred curves, and many PIT sites preferred curvature concave relative to fixation — [Ponce CR, Hartmann TS & Livingstone MS (2017) "End-stopping predicts curvature tuning along the ventral stream." J Neurosci 37:648–659](https://doi.org/10.1523/JNEUROSCI.2507-16.2016)

**Line ends, corners and occlusion cues**
- [Mq] Neurons sensitive to occlusion cues (dark and light line ends, corners):
  - They were **end-stopped** and occurred at similar frequencies in V1 and V2. In V1 they concentrated in the upper layers; in V2 they were found in all laminae and all cytochrome-oxidase stripes.
  - They responded to short bars and edges of optimal length and to stimuli *terminating* in the RF (line ends, corners).
  - About half signaled the **direction** of the termination, and about 60% were selective for particular **types** of termination.
  - Neurons signaling occluding (illusory) contours were most common in V2 pale stripes and rare in V1 — [Heider B, Meskenaite V & Peterhans E (2000) "Anatomy and physiology of a neural mechanism defining depth order and contrast polarity at illusory contours." Eur J Neurosci 12:4117–4130](https://doi.org/10.1046/j.1460-9568.2000.00293.x)
- [Mq/NWM] V2 pale-stripe cells are orientation-selective and "more than half of them are end-stopped". Pale stripes receive input from V1 interblobs (parvocellular stream) — [Hubel DH & Livingstone MS (1987) "Segregation of form, color, and stereopsis in primate area 18." J Neurosci 7:3378–3415](https://doi.org/10.1523/JNEUROSCI.07-11-03378.1987)
- [Mq] Contrast sign at end zones:
  - In end-stopped V1 cells, end-zone inhibition is "highly selective for the relative sign of contrast between the central activating stimulus and stimuli presented at the end zones".
  - In length-summing cells, end-zone *facilitation* is not sign-selective.
  - The authors argue that occlusion and shadows have distinct sign-of-contrast relations at edge-crossing junctions, and that this property "obviat[es] the need for explicit junction detectors" — [Yazdanbakhsh A & Livingstone MS (2006) "End stopping in V1 is sensitive to contrast." Nat Neurosci 9:697–702](https://doi.org/10.1038/nn1693)
- [Mq] End-stopped V1 neurons signal motion direction independently of contour orientation by responding only to the endpoints of long contours. "The time course of end-stopping is similar to the time course of motion integration by MT neurons" (no ms figure in the abstract) — [Pack CC, Livingstone MS, Duffy KR & Born RT (2003) "End-stopping and the aperture problem: two-dimensional motion signals in macaque V1." Neuron 39:671–680](https://doi.org/10.1016/s0896-6273(03)00439-2)

**Fixed-weight model directly built on these cells**
- [Model] The Heitger et al. pipeline:
  1. Even- and odd-symmetric oriented filters (S-operators).
  2. A local-energy combination (C-operator).
  3. "Differentiation" of the C maps *along* the preferred orientation, giving single and double end-stopped operators.
  4. Local maxima ("key-points") of the combined end-stopped activity.
  - S/C operators represent 1-D features (edges, lines). End-stopped key-points make 2-D features explicit: line ends, corners and strong curvature. Tested on grey-value images — [Heitger F, Rosenthaler L, von der Heydt R, Peterhans E & Kübler O (1992) "Simulation of neural contour mechanisms: from simple to end-stopped cells." Vision Res 32:963–981](https://doi.org/10.1016/0042-6989(92)90039-l)

### Inferences
- **Design primitive.** A single-ended end-stopped unit is C-energy at (x, θ) minus C-energy at (x + L·u_θ, θ). Its sign of offset (+u or −u) encodes the *direction of termination*, as in Heider et al. 2000. A double-ended unit subtracts both sides. Following Heitger 1992, this gives a fixed, differentiable line-end/corner/curvature map.
- **Junction signatures from end-stopped (ES) responses.** This is my own synthesis; it is not tested physiologically.
  - At an **X-crossing**, both lines continue through the point, so ES units on either orientation see continuing energy in both end zones and stay quiet.
  - At a **T** (or a Y branch), the stem terminates on the bar. A single-ended ES unit tuned to the stem fires, with its "open" side pointing away from the bar. The bar's own ES units stay quiet.
  - At an **L**, both arms terminate, giving two co-localized single-ended ES responses with orthogonal orientations and opposite termination directions.
  - Caveat: real end zones are orientation-tuned (strongest for iso-oriented stimuli at the ends; Cavanaugh 2002). An orthogonal line crossing the end zone (the bar of a T) therefore inhibits only weakly. This is what lets the stem's ES unit fire at a T even though contrast is present in its end zone.
- **Occlusion vs transparency.** Biological end-zone inhibition is computed separately per relative contrast sign (Yazdanbakhsh & Livingstone 2006). This argues for computing end-zone inhibition separately for ON and OFF (or odd-filter-sign) channels. In vessel images (dark vessels on a bright background), a transparent overlap is darker than either vessel and keeps the contrast sign along each vessel. An occluding vessel instead replaces the occluded vessel's contrast. Polarity-specific end zones could separate these cases.
- **Contrast normalization.** Summation length grows about 2.3× at low contrast (Sceniak 1999), and divisive and subtractive surround models fit similarly (Sceniak 2001). A fixed ES operator should therefore include divisive contrast normalization, so that faint, thin vessels are not fragmented into spurious "line ends".

### Gaps
- I did not find the latency of end-stopping onset in macaque V1, in ms after response onset. Pack et al. 2003 give only a qualitative comparison with MT integration dynamics.
- No primate test was found that is equivalent to Bolz & Gilbert's layer-6 inactivation. In primate, Nassi et al. 2013 implicate feedback. The relative contribution of intracortical (layer 6 → 4) and feedback circuits to primate end-stopping is unresolved.
- The Hubel & Wiesel 1965 wiring scheme and "higher-order hypercomplex" description come from standard accounts of the paper; PubMed has no abstract and the full text was not read this session.
- Direct curvature-tuning evidence for end-stopped cells is mostly cat (Dobbins 1987; Versavel 1990). For macaque, Ponce et al. 2017 give array-site correlations rather than single-cell curvature-sign tuning.

---

## 2. Corner-, angle- and junction-selective neurons (V1, V2, V4, IT): is there selectivity for T- vs L- vs X-junctions?

### Takeaway
Selectivity for combinations of orientations already appears in macaque V1: two-photon imaging shows many superficial-layer neurons preferring corners, junctions and curvature to their components, plus a separate population of "plaid detectors" with cross-angle selectivity. In V2, angle-selective cells are common (36% of superficial V2 cells were selective for wide angles), and about one-third of V2 cells are selective for complex shapes or orientation combinations. In V4, most neurons prefer curves or corners to bars; they are tuned to curvature × object-centered position and clustered into separate curve and corner domains. Modelling of V2 angle cells suggests a **linear combination of two "half-line" subunits plus suppression of non-preferred components and gain control**, not a dedicated AND detector. **I found no single-unit study that directly compares T- vs L- vs X- vs Y-junction selectivity.** The closest evidence is: V2/V4 responses to "intersecting lines" (Hegdé & Van Essen); V1 plaid/cross-angle detectors (Guan et al.); orientation-discontinuity detectors (Sillito et al.); and T-junction-dependent suppression in V4 (Bushnell et al., see Q3).

### Cited Findings
**V1**
- [Mq, two-photon, awake] A "large percentage" of superficial-layer V1 neurons "responded more strongly to complex patterns, such as corners, junctions, and curvature, than to their oriented line or edge components". The authors propose that single V1 neurons detect local high-order patterns. The full text (with percentages per pattern class) was not accessible this session — [Tang S, Lee TS, Li M, Zhang Y, Xu Y, Liu F, Teo B & Jiang H (2018) "Complex pattern selectivity in macaque primary visual cortex revealed by large-scale two-photon imaging." Curr Biol 28:38–48.e3](https://doi.org/10.1016/j.cub.2017.11.039)
- [Mq, two-photon, awake, superficial V1] Responses to plaids (two superimposed gratings, an X-like texture):
  - Many *non-orientation-tuned* neurons responded weakly to gratings but strongly to plaids, often with plaid-orientation and **cross-angle selectivity**.
  - About 94% of orientation-tuned neurons showed cross-orientation inhibition regardless of relative contrast. Only about 8% showed plaid facilitation at off-peak orientations.
  - About 95% of plaid neurons were direction-insensitive ("plaid pattern detectors").
  - The authors conclude that plaid-responding and grating-responding neurons are separate subpopulations — [Guan SC, Zhang SH, Zhang YC, Tang SM & Yu C (2020) "Plaid detectors in macaque V1 revealed by two-photon calcium imaging." Curr Biol 30:934–940.e3](https://doi.org/10.1016/j.cub.2020.01.005)
- [Cat + Mq per MeSH] Orientation-discontinuity cells:
  - In patch-suppressed V1 cells, a surround at a markedly different (e.g. orthogonal) orientation enhanced the response through disinhibition and active facilitation, producing "supra-optimal" responses.
  - Some cells responded well when center and surround orientations were *swapped*, with neither component effective alone.
  - Simultaneously recorded cells with orthogonal RFs fired in a correlated way.
  - The authors propose a mechanism for "focal orientation discontinuities such as junctions or corners" — [Sillito AM, Grieve KL, Jones HE, Cudeiro J & Davis J (1995) "Visual cortical mechanisms detecting focal orientation discontinuities." Nature 378:492–496](https://doi.org/10.1038/378492a0)
- [Cat — not primate] Local connectivity and position in the orientation map:
  - Short-range connections (from cross-correlation) were graded with distance, "largely radially symmetrical and relatively independent of orientation preferences".
  - The contextual influence of flanking elements varied systematically with the neuron's position in the orientation map, i.e. near pinwheels vs within iso-orientation domains.
  - The authors conclude local circuitry "could endow neurons with a graded specialization for processing angular visual features such as corners and T junctions", possibly forming its own map linked to the orientation map — [Das A & Gilbert CD (1999) "Topography of contextual modulations mediated by short-range interactions in primary visual cortex." Nature 399:655–661](https://doi.org/10.1038/21371). Commentary: [Eysel U (1999) "Turning a corner in vision research." Nature 399:641–644](https://doi.org/10.1038/21329)
- [Mq] About 15% of cells in each of LGN, V1 and V2 responded significantly more to concentric than to parallel gratings. Many cells responded to hyperbolic or radial gratings, but few preferred them. Complex, non-oriented, low-aspect-ratio cells responded best to non-Cartesian gratings — [Mahon LE & De Valois RL (2001) "Cartesian and non-Cartesian responses in LGN, V1, and V2 cells." Vis Neurosci 18:973–981](https://pubmed.ncbi.nlm.nih.gov/12020088/)

**V2**
- [Mq, alert] Angle selectivity in superficial V2:
  - Angle stimuli were two half-lines, each longer than the classical RF and drawn in 1 of 12 directions, meeting at the RF center.
  - Of 114 superficial-layer V2 neurons, 91 responded selectively to angles. 41 (36.0%) were selective for **wide angles (60°–150°)**, responding differently from straight lines or sharp (30°) angles.
  - About one-quarter were highly selective for one angle.
  - Half-line preferences were "more or less consistent with one or two components of the optimal angle".
  - Responses to the optimal angle were indistinguishable from responses to the optimal single half-line, so these cells are "unlikely to be specific angle detectors" — [Ito M & Komatsu H (2004) "Representation of angles embedded within contour stimuli in area V2 of macaque monkeys." J Neurosci 24:3313–3324](https://doi.org/10.1523/JNEUROSCI.4364-03.2004)
- [Mq + Model] A linear–non-linear cascade model with a few half-line-selective subunits replicated angle selectivity in most V2 neurons:
  - Selectivity comes from a **linear combination** of responses to individual half-line components, sharpened by **suppression of non-preferred half-lines**, with **gain control by the number of components**. There is no specific facilitatory interaction between the two preferred components.
  - Acute-angle selectivity comes from one preferred component plus weak responses to the other — [Ito M & Goda N (2011) "Mechanisms underlying the representation of angles embedded within contour stimuli in area V2 of macaque monkeys." Eur J Neurosci 33:130–142](https://doi.org/10.1111/j.1460-9568.2010.07489.x)
- [Mq] Orientation tuning measured at several positions inside the classical RF:
  - 70% of V2 neurons had similar orientation tuning throughout the RF.
  - Many others had subregions tuned to different orientations, "most commonly about 90 degrees apart" (a corner- or L-like RF structure).
  - About one-third showed inhibitory two-position interactions that made them selective for combinations of orientations — [Anzai A, Peng X & Van Essen DC (2007) "Neurons in monkey visual area V2 encode combinations of orientations." Nat Neurosci 10:1313–1321](https://doi.org/10.1038/nn1975)
- [Mq, alert] With 128 grating and geometric line stimuli (bars, sinusoids, angles, arcs, circles, **intersecting lines**, hyperbolic and polar gratings), most V2 cells responded well to some complex stimuli. In many, the best complex stimulus beat the best bar or sinusoid, and about one-third showed significant differential responses to complex shape characteristics — [Hegdé J & Van Essen DC (2000) "Selectivity for complex shapes in primate visual area V2." J Neurosci 20:RC61](https://doi.org/10.1523/JNEUROSCI.20-05-j0001.2000)
- [Mq] V2 cells responded better on average to gratings but were more selective for contour stimuli (angles, intersections, arcs, circles). The first 2 and 5 principal components explained 69% and 85% of population response variance — [Hegdé J & Van Essen DC (2003) "Strategies of shape representation in macaque visual area V2." Vis Neurosci 20:313–328](https://doi.org/10.1017/s0952523803203102)
- [Mq] V1 also showed complex-shape selectivity in some cells. V2 vs V4 differences "do not parallel the stepwise organization of the anatomical hierarchy" — [Hegdé J & Van Essen DC (2007) "A comparative study of shape representation in macaque visual areas V2 and V4." Cereb Cortex 17:1100–1116](https://doi.org/10.1093/cercor/bhl020)
- [Mq, anesthetized] Texture (orientation-discontinuity) boundaries:
  - Transient V2 neurons responded robustly to texture boundaries at both 20 and 40 ms frame durations, and more strongly when the boundary was parallel to the preferred orientation.
  - V1 responded only at 40 ms frames, and independently of boundary orientation.
  - The authors conclude V2 boundary signals arise independently of V1 contextual modulation — [Schmid AM, Purpura KP & Victor JD (2014) "Responses to orientation discontinuities in V1 and V2: physiological dissociations and functional implications." J Neurosci 34:3559–3578](https://doi.org/10.1523/JNEUROSCI.2293-13.2014)

**V4**
- [Mq, awake] Of 152 V4 cells tested with contour features (angles and curves), most responded better to contour features than to edges or bars, and about one-third were systematically tuned for them. Many were tuned to contour-feature *orientation* (angles and curves "pointing" in a direction), with a strong **convex** bias — [Pasupathy A & Connor CE (1999) "Responses to contour features in macaque area V4." J Neurophysiol 82:2490–2502](https://doi.org/10.1152/jn.1999.82.5.2490)
- [Mq] 109 V4 cells were tested with 366 shapes built from convex and concave boundary elements. Responses were best described by Gaussian tuning on a **curvature × angular position** (object-centered) domain, e.g. "concave curvature at the right". Many cells were tuned to compound configurations such as "a convex angle adjacent to a concave curve". This fit beat edge- or axis-orientation models — [Pasupathy A & Connor CE (2001) "Shape representation in area V4: position-specific tuning for boundary conformation." J Neurophysiol 86:2505–2519](https://doi.org/10.1152/jn.2001.86.5.2505)
- [Mq] Summing each cell's tuning peak weighted by its response gave a population surface with 3–8 peaks at major boundary features, enough to approximately reconstruct the shape — [Pasupathy A & Connor CE (2002) "Population coding of shape in area V4." Nat Neurosci 5:1332–1338](https://doi.org/10.1038/nn972)
- [Mq] Early V4 responses carried individual contour-orientation information. Tuning for multi-orientation (curved) contours developed gradually over about 50 ms, supporting a synthesis from simpler orientation signals — [Yau JM, Pasupathy A, Brincat SL & Connor CE (2013) "Curvature processing dynamics in macaque area V4." Cereb Cortex 23:198–209](https://doi.org/10.1093/cercor/bhs004)
- [Mq] V4 neurons tuned to highly curved shapes showed limited translation invariance and local variation of orientation preference at a fine spatial scale. Straight-preferring cells had homogeneous fine-scale orientation maps. Both patterns fit a simple **orientation-pooling model**: curvature tuning comes from pooling heterogeneous orientation signals from earlier areas — [Nandy AS, Sharpee TO, Reynolds JH & Mitchell JF (2013) "The fine structure of shape tuning in area V4." Neuron 78:1102–1115](https://doi.org/10.1016/j.neuron.2013.04.016)
- [Mq] About 70% of V4 neurons encoded shape size-invariantly; "normalized" curvature (scale-free) explained responses better than absolute curvature — [El-Shamayleh Y & Pasupathy A (2016) "Contour curvature as an invariant code for objects in visual area V4." J Neurosci 36:5532–5543](https://doi.org/10.1523/JNEUROSCI.4139-15.2016)
- [Mq, two-photon, awake, dorsal V4; full text read] Curve and corner domains:
  - Stimuli: bars, smooth curves, corners (separation angles 45°, 90°, 135°), non-smooth "Π-shapes" and Cartesian/concentric/radial gratings. **No T, X or Y junctions were tested.**
  - 64.7% (346/535) of neurons preferred curves and corners to single bars; 1.5% preferred bars.
  - Curve-preferring and corner-preferring neurons clustered into separate domains about 300 µm across.
  - Smoothness was the cardinal difference between curve and corner selectivity (curve/corner index vs curve/Π index, R = 0.72).
  - 91.6% were tuned to the orientation of their preferred curve or corner — [Jiang R, Andolina IM, Li M & Tang S (2021) "Clustered functional domains for curves and corners in cortical area V4." eLife 10:e63798](https://doi.org/10.7554/eLife.63798) (full text: [PMC8175081](https://pmc.ncbi.nlm.nih.gov/articles/PMC8175081/))

**IT / higher**
- [Mq] Using a reduction method (simplifying complex objects to each cell's "critical feature"), cells responding maximally to complex object features were found in V4 and posterior IT as well as anterior IT. V4 and posterior IT cells had small RFs and also responded somewhat to simple features — [Kobatake E & Tanaka K (1994) "Neuronal selectivities to complex object features in the ventral visual pathway of the macaque cerebral cortex." J Neurophysiol 71:856–867](https://doi.org/10.1152/jn.1994.71.3.856)
- [Mq] Posterior IT (TEO/posterior TE) neurons integrate multiple contour fragments (straight and curved edge fragments) through both linear and nonlinear mechanisms — [Brincat SL & Connor CE (2004) "Underlying principles of visual shape selectivity in posterior inferotemporal cortex." Nat Neurosci 7:880–886](https://doi.org/10.1038/nn1278)

### Inferences
- **A "ray basis" for junction classification.** V2 angle cells behave like sums of half-line (single-ended, termination-direction-coded) subunits meeting at a vertex (Ito & Komatsu 2004; Ito & Goda 2011). By extension, all junction types can be coded from a fixed set of ~12–16 ray detectors centered on a candidate vertex:
  - L / corner: 2 rays, not collinear.
  - T: 3 rays, two of them collinear.
  - Y / arrow: 3 rays, none collinear (arrow: one angle >180°).
  - X: 4 rays forming two collinear pairs.
  - Line end: 1 ray.
  - Continuation: 2 collinear rays.
  
  The biological "linear sum + suppression of non-preferred rays + divisive gain control by ray count" rule gives a fixed-weight recipe. A deterministic classifier would also need explicit ray-count and collinearity tests, because biological angle cells respond as strongly to their best single half-line (a distributed code, not a label).
- **A separate X channel.** Guan et al. 2020 show that most oriented V1 cells are suppressed by a superimposed second orientation (cross-orientation inhibition), while a distinct non-oriented population fires selectively to co-occurring orientations with cross-angle selectivity. This argues for a dedicated "co-occurrence/crossing" detector, separate from the oriented line channels, rather than relying on oriented cells to signal crossings.
- **Smoothness splits curves from corners.** In V4, curve and corner selectivity are anatomically segregated and split on smoothness (Jiang 2021). For vessels, a smooth high-curvature bend (tortuosity) should be coded separately from a sharp angle (a likely junction, or a vessel occluded at a corner). The curve/Π comparison suggests a fixed "smoothness" test: are orientation changes distributed or concentrated at one point?

### Gaps
- **No primary single-unit study was found that directly contrasts T- vs L- vs X- vs Y-junction selectivity** in macaque V1, V2, V4 or IT. A targeted search did not locate one. Absence of evidence here does not mean such cells do not exist.
- The full stimulus breakdown and percentages of Tang et al. 2018 (which pattern classes — corners, crosses, T-like, curvature — were preferred, and by what fraction) could not be retrieved (PDF host returned HTTP 503).
- Das & Gilbert 1999 (pinwheel-dependent corner/T-junction specialization) is **cat** data. I found no equivalent primate test.

---

## 3. Border ownership and occlusion: how do T-junctions influence border-ownership signals, what is their latency, and is the mechanism feedback or horizontal?

### Takeaway
Border-ownership (BO, side-of-figure) modulation is found in >50% of V2 and V4 neurons and 18% of top-layer V1 neurons. It emerges <25 ms after response onset (about 60–70 ms after stimulus onset in V2) and is nearly independent of figure size up to 21°. Local edge signals (contrast polarity, stereo depth order) are present almost immediately (<13 ms after onset), while the context-dependent BO signal is delayed about 30 ms. Its latency grows far less with cortical distance than horizontal-fiber conduction predicts, and distant figure fragments act without extra delay. The von der Heydt lab therefore favors fast **feedback from "grouping cells"** in a higher area. This is an inference by elimination, and an intra-areal (horizontal) model (Zhaoping 2005) remains a competing account. Occlusion cues (line terminations/T-junctions, corners) are represented by end-stopped cells in V1/V2. V2/V3 neurons use them to signal figure-ground direction independently of contrast polarity, and in V4, T-junction geometry suppresses responses to "accidental" contours of occluded shapes. Local cues alone give only weak, transient BO signals.

### Cited Findings
**Core BO physiology**
- [Mq, awake] Border-ownership coding in V1, V2 and V4:
  - The same local edge **or line** was presented as part of different figures. Responses were significantly modulated by side-of-figure in **>50% of V2 and V4 neurons** and **18% of top-layer V1 neurons**.
  - Most BO cells also coded contrast polarity. Another 20% of V2/V4 neurons and 48% of top-layer V1 neurons coded polarity only.
  - BO differences emerged "<25 msec after the response onset" and were nearly independent of figure size up to the 21° display limit.
  - With complex displays (e.g. overlapping figures, where T-junctions are present), some neurons gave invariant BO signals and others failed for some displays, but reversals were rare.
  - The authors conclude that the short latencies and incomplete cue invariance mean BO is "generated within the visual cortex rather than projected down from higher levels" — [Zhou H, Friedman HS & von der Heydt R (2000) "Coding of border ownership in monkey visual cortex." J Neurosci 20:6594–6611](https://doi.org/10.1523/JNEUROSCI.20-17-06594.2000)
- [Mq] V2 neurons combine binocular stereo depth order with Gestalt configuration cues. The "near" side of a cell's preferred 3-D edge generally coincides with its preferred side-of-figure in 2-D displays. Gestalt factors can enhance or null stereo information — [Qiu FT & von der Heydt R (2005) "Figure and ground in the visual cortex: V2 combines stereoscopic cues with gestalt rules." Neuron 47:155–166](https://doi.org/10.1016/j.neuron.2005.05.028)
- [Mq, V1/V2] Latency comparisons:
  - Local edge definition (contrast polarity, stereo depth order) emerged "immediately (<13 ms) after the edge onset response". The context-dependent BO signal was "delayed by about 30 ms".
  - When context was placed farther away, latencies increased with cortical distance, but "much less than predicted by the horizontal propagation hypothesis" given known horizontal conduction velocities.
  - Horizontal propagation alone is "highly unlikely"; back-projections from extrastriate areas are "plausible" — [Sugihara T, Qiu FT & von der Heydt R (2011) "The speed of context integration in the visual cortex." J Neurophysiol 106:374–385](https://doi.org/10.1152/jn.00928.2010)
- [Mq, V1/V2] Fragmented rectangles:
  - Surround fragments produced **facilitation on the preferred BO side and suppression on the non-preferred side**, with about 80% of tested locations contributing.
  - Far fragments acted even without nearer fragments and without extra delay.
  - The results favor an antagonistic surround produced by **reentrant signals from a higher-level area** — [Zhang NR & von der Heydt R (2010) "Analysis of the context integration mechanisms underlying figure-ground organization in the visual cortex." J Neurosci 30:6482–6496](https://doi.org/10.1523/JNEUROSCI.5168-09.2010)
- [Mq, V2, natural scenes; full text read] 140 V2 neurons in 3 monkeys:
  - 63% (88/140) were BO-selective for standard squares and 46% (65/140) for full natural scenes. Preferences were 94% consistent between the two tests.
  - Local "patches" alone gave about 2% of square-test BO strength (n.s.); context alone gave about 35%.
  - Latencies: response onset about 44 ms; full-image BO about 60 ms; context influence about 73 ms; square BO 62 ms (3° figure) vs 71 ms (8° figure).
  - Junctions and high-curvature points were excluded from later test points. The paper does **not** quantify T-junction contributions.
  - The authors' grouping-cell framework takes inputs from end-stopped cells responding to T-junctions (indicating occlusion direction) and favors compact, cocircular edges.
  - They argue "by elimination" (intracortical propagation too slow; IT too late at about 100–130 ms) that BO requires back-projections to V1/V2 from unidentified higher-level mechanisms with fast white-matter conduction — [Williford JR & von der Heydt R (2016) "Figure-ground organization in visual cortex for natural scenes." eNeuro 3(6):ENEURO.0127-16.2016](https://doi.org/10.1523/ENEURO.0127-16.2016) (full text: [eneuro.org](https://www.eneuro.org/content/3/6/eneuro.0127-16.2016.full))
- [Mq, V2/V3] A smaller fraction of cells than previously reported were consistent across the artificial BO stimulus battery. Consistent cells *as a population* signaled side-of-figure for faces, occluding faces and, **at longer latencies**, illusory-contour stimuli (Mooney faces, faces without local edges). Local edges or outlines alone did not always evoke BO signals — [Hesse JK & Tsao DY (2016) "Consistency of border-ownership cells across artificial stimuli, natural stimuli, and stimuli with ambiguous contours." J Neurosci 36:11338–11349](https://doi.org/10.1523/JNEUROSCI.1857-16.2016)
- [Mq] Spike synchrony was elevated only between neurons inferred, from their BO preferences, to share a grouping circuit. The elevation was weak relative to random fluctuations, so binding appears to be coded by feedback modulation of response *strength* rather than by synchrony — [Martin AB & von der Heydt R (2015) "Spike synchrony reveals emergence of proto-objects in visual cortex." J Neurosci 35:6860–6870](https://doi.org/10.1523/JNEUROSCI.3590-14.2015)
- [Mq, V2] Attentional enhancement was stronger when the attended figure was on the neuron's preferred BO side. For a border between two overlapping figures, enhancement was generally on the preferred BO side — [Qiu FT, Sugihara T & von der Heydt R (2007) "Figure-ground mechanisms provide structure for selective attention." Nat Neurosci 10:1492–1499](https://doi.org/10.1038/nn1989)
- [Review] Evidence that BO reflects "proto-objects": reversal with perceived object structure, BO-specific attentional enhancement, persistence, and remapping across saccades. These are explained by "grouping circuits that sum contour feature signals in search of objectness, and, via recurrent projections, enhance the corresponding low-level feature signals" — [von der Heydt R (2015) "Figure-ground organization and the emergence of proto-objects in the visual cortex." Front Psychol 6:1695](https://doi.org/10.3389/fpsyg.2015.01695)

**Models (feedback vs horizontal) — flagged as contested**
- [Model] Craft et al. criticize lateral-propagation models: horizontal propagation would make BO delays grow with figure size, contradicting the data. They propose dedicated contour-grouping (G) cells that pool edge signals and feed back to BO cells, which also gives an interface for attention — [Craft E, Schütze H, Niebur E & von der Heydt R (2007) "A neural model of figure-ground organization." J Neurophysiol 97:4310–4326](https://doi.org/10.1152/jn.00203.2007)
- [Model] Contrasting view: a V2 model with local classical RFs and **finite-range intra-areal interactions** generates BO signals "without requiring any top-down mechanism or external explicit labels for figures, T junctions, or corners" — [Zhaoping L (2005) "Border ownership from intracortical interactions in visual area V2." Neuron 47:143–153](https://doi.org/10.1016/j.neuron.2005.04.005)

**Occlusion cues (T-junctions, line terminations) in V1/V2/V3/V4**
- [Mq, awake, V1/V2/V3/V3A] The stimuli mimicked occlusion: a uniform rectangle overlying a grating of opposite contrast, with figure-ground direction defined by the **terminating grating lines** (T-junction-like occlusion cues).
  - Most neurons signaled figure-ground direction independently of contrast polarity, and these were found **only in prestriate cortex**.
  - Striate neurons often failed to respond or were polarity-selective.
  - The model explaining this also explains illusory-contour responses — [Baumann R, van der Zwan R & Peterhans E (1997) "Figure-ground segregation at contours: a neural mechanism in the visual cortex of the alert monkey." Eur J Neurosci 9:1290–1303](https://doi.org/10.1111/j.1460-9568.1997.tb01484.x)
- [Mq] End-stopped neurons detecting line ends and corners (occlusion cues) occur in V1 and V2. About half signal termination direction and about 60% are termination-type selective (details in Q1) — [Heider, Meskenaite & Peterhans (2000) Eur J Neurosci 12:4117](https://doi.org/10.1046/j.1460-9568.2000.00293.x)
- [Mq, V4] Responses to preferred contours were **suppressed when adjoining context rendered them "accidental"** (partial occlusion at the boundary with the occluder). Suppression reversed when a small gap broke the occlusion percept. Controls indicated dependence on "contour geometry at **T-junctions**", not on local color/luminance, proximity or spatial frequency. The authors suggest this as a first step of segmentation in the ventral stream — [Bushnell BN, Harding PJ, Kosai Y & Pasupathy A (2011) "Partial occlusion modulates contour-based shape encoding in primate area V4." J Neurosci 31:4012–4024](https://doi.org/10.1523/JNEUROSCI.4766-10.2011)
- [Mq, V1/V2] V2 contour-completion facilitation:
  - **Amodal completion:** facilitation by collinear lines outside the RF was **blocked by orthogonal lines intersecting the contour** (a crossing) but **recovered when the orthogonal line was placed in the near depth plane** (an occluder).
  - V2 cells also showed modal completion and surface depth capture.
  - These effects were more common in V2 than V1 — [Bakin JS, Nakayama K & Gilbert CD (2000) "Visual responses in monkey areas V1 and V2 to three-dimensional surface configurations." J Neurosci 20:8188–8198](https://doi.org/10.1523/JNEUROSCI.20-21-08188.2000)
- [Mq, V1] Some orientation-selective V1 cells stopped responding when a bar was partly occluded by a small patch, and also when the patch had uncrossed disparity (behind the bar). They **resumed responding when the patch had crossed disparity** (in front, i.e. occluding) — [Sugita Y (1999) "Grouping of image fragments in primary visual cortex." Nature 401:269–272](https://doi.org/10.1038/45785)
- [Mq, V1] Figure-ground response enhancement for texture-defined figures had a latency of 30–40 ms relative to response onset and covered the whole figure — [Lamme VAF (1995) "The neurophysiology of figure-ground segregation in primary visual cortex." J Neurosci 15:1605–1615](https://doi.org/10.1523/JNEUROSCI.15-02-01605.1995)
- [Mq, V1 laminar] Laminar timing in a figure-ground task:
  - Visual response started about 40 ms after stimulus onset in layers 4 and 6.
  - Boundary detection appeared about 70 ms after onset in upper layer 4 and the superficial layers, consistent with horizontal connections.
  - About 30 ms later, synaptic input arrived in layers 1, 2 and 5 (feedback-recipient layers), filling in the whole figure with enhanced activity — [Self MW, van Kerkoerle T, Supèr H & Roelfsema PR (2013) "Distinct roles of the cortical layers of area V1 in figure-ground segregation." Curr Biol 23:2121–2129](https://doi.org/10.1016/j.cub.2013.09.013)
- [Hu, fMRI adaptation] BO selectivity in human V2 was "robust and reliable across subjects" and "largely dependent on attention" — [Fang F, Boyaci H & Kersten D (2009) "Border ownership selectivity in human early visual cortex and its modulation by attention." J Neurosci 29:460–465](https://doi.org/10.1523/JNEUROSCI.4628-08.2009)

### Inferences
- **Two-stage, two-speed design.** Stage 1, fast and local: edge polarity, depth and line-end/T-junction cues, available within about 13 ms of edge onset (Sugihara 2011). Stage 2, context: BO and occlusion assignment delayed about 15–30 ms, pooling distant contour evidence through a higher-level "grouping cell" layer that feeds back multiplicatively (Craft 2007; Zhang & von der Heydt 2010; Williford & von der Heydt 2016). In a fixed-weight network, grouping cells can be implemented as large annular or cocircular pooling kernels over edge/ES maps, with feedback gain applied to the matching BO-tagged edge units. This is effectively one recurrent iteration, with no long horizontal diffusion needed.
- **Crossing vs occlusion vs branch in vessel images.** Bakin et al. 2000 and Sugita 1999 show the biological rule: a line interrupted by another line is completed *only* if the interrupter is interpreted as nearer (an occluder); a plain 2-D crossing blocks completion. In 2-D vessel projections (fundus photographs, angiograms), depth cues are mostly absent. The rule would then need replacing with configural cues:
  - **Occlusion crossing:** the occluded vessel's two edges terminate at the occluder's edges, forming paired T-junctions, and the stems are collinear across the occluder (amodal continuation).
  - **Transparent crossing:** an X-junction with the darker overlap preserving contrast sign along both vessels (see Q1 inference on polarity-specific end zones).
  - **Branch point:** a Y or T without a collinear partner across the junction.
  
  This is a design proposal, not a tested neural mechanism.
- **The V4 accidental-contour rule** (Bushnell 2011): suppress curvature/corner evidence at T-junction points, because the corner formed where an occluded vessel meets an occluder's edge is "accidental". This keeps vessel shape descriptors (tortuosity, bifurcation angle) from being contaminated by crossings.

### Gaps
- No single-unit study was found that systematically manipulates **T-junctions on thin line stimuli** (as opposed to filled figures or gratings) and measures BO. Zhou et al. 2000 used lines as well as edges, but the abstract does not report T-junction-specific manipulations.
- The feedback vs horizontal question is **contested**:
  - Pro feedback: von der Heydt lab, by elimination from latency and size data.
  - Pro intra-areal: Zhaoping 2005.
  - Zhou et al. 2000 originally concluded BO is generated "within the visual cortex" rather than projected from higher levels, which is not contradictory but shows the "higher-level area" is unidentified.
- The latency of BO signals specifically driven by T-junction cues (as opposed to global shape) was not found.

---

## 4. Illusory/subjective contours and line-end signals in V2 (and V1), and their relevance to amodal completion behind occluders

### Takeaway
About 32–44% of orientation-selective V2 neurons signal illusory contours: either "lines of pattern discontinuity" formed by aligned line ends of abutting gratings, or contours bridging gaps of 1–3.5° between notched inducers. Responses exceed linear-summation predictions and are abolished by thin "closing" lines. In the original reports V1 almost never did this (1/60 and 1/26 neurons). The proposed mechanism pools **end-stopped responses oriented orthogonal to the contour** (line-end detectors). Whether V1 signals illusory contours is **contested** (Grosof 1993 yes; Ramsden 2001 negative or de-emphasized signaling; Lee & Nguyen 2001 weak and later than V2). Illusory-contour signals appear in V2 superficial layers about 70 ms after onset and in V1 about 100 ms after onset. Human EEG localizes the earliest illusory-contour effect to LOC at 88–100 ms, and modal and amodal completion share a 140 ms stage. V4 represents real and illusory contours in the same orientation domains.

### Cited Findings
- [Mq, alert] Area-18 (V2) cells responded to illusory-contour figures "as if the contours were formed by real lines or edges". Modifications that weakened the percept reduced responses. Area-17 cells did not respond — [von der Heydt R, Peterhans E & Baumgartner G (1984) "Illusory contours and cortical neuron responses." Science 224:1260–1262](https://doi.org/10.1126/science.6539501)
- [Mq, alert] Abutting gratings (a border defined only by aligned line ends of two gratings offset by half a cycle):
  - In V2, **45/103 (44%)** neurons signaled the anomalous contour's orientation; 16 did so without signaling the inducing lines.
  - In V1, only 1/60 orientation-selective neurons did.
  - Responses needed a minimum number of inducing lines but depended little on spacing, and were weaker for oblique inducers, with a Zöllner-like orientation bias.
  - Proposed model: V2 sums two parallel paths, one for edges/lines and one "that defines anomalous contours by **pooling signals from end-stopped receptive fields oriented mainly orthogonal to the contour**" — [von der Heydt R & Peterhans E (1989) "Mechanisms of contour perception in monkey visual cortex. I. Lines of pattern discontinuity." J Neurosci 9:1731–1748](https://doi.org/10.1523/JNEUROSCI.09-05-01731.1989)
- [Mq, alert] Contours bridging gaps (moving notch pairs mimicking an occluding dark bar):
  - **23/72 (32%)** V2 neurons responded with the RF in the gap. Responses were reduced or abolished when the notches were closed by thin lines.
  - Neurons within 4° of the fovea signaled contours bridging gaps of 1°–3.5°.
  - Responses exceeded linear-summation and summation-to-threshold predictions.
  - Only 1/26 V1 neurons responded comparably (narrow gap only).
  - Explained by "corners excit[ing] end-stopped fields orthogonal to the contour whose signals are pooled in the contour neurons" — [Peterhans E & von der Heydt R (1989) "Mechanisms of contour perception in monkey visual cortex. II. Contours bridging gaps." J Neurosci 9:1749–1763](https://doi.org/10.1523/JNEUROSCI.09-05-01749.1989)
- [Mq] Occluding (illusory) contour neurons are most frequent in V2 **pale CO stripes** and found in all V2 laminae, but rare in V1 (upper layers only) — [Heider, Meskenaite & Peterhans (2000) Eur J Neurosci 12:4117](https://doi.org/10.1046/j.1460-9568.2000.00293.x)
- **Contested — V1:** [Mq] With different stimuli, illusory-contour responses were found in "about half the neurons studied in V1" — [Grosof DH, Shapley RM & Hawken MJ (1993) "Macaque V1 neurons can signal 'illusory' contours." Nature 365:550–552](https://doi.org/10.1038/365550a0)
- **Contested — V1:** [Mq, optical imaging + units] V1 shows an orientation-specific response to abutting-line-grating illusory contours that is **inversely related** to the real orientation map, i.e. illusory orientation is "negatively signaled or de-emphasized in V1". This is "not due merely to presence of line ends" — [Ramsden BM, Hung CP & Roe AW (2001) "Real and illusory contour processing in area V1 of the primate: a cortical balancing act." Cereb Cortex 11:648–665](https://doi.org/10.1093/cercor/11.7.648)
- [Mq, static Kanizsa figures] V1 illusory-contour signals were weaker than in V2 but significant, especially in superficial layers. Population onset: V1 superficial **100 ms**, V1 deep 120–190 ms; V2 superficial **70 ms**, V2 deep 95 ms. The authors infer intercortical (V2→V1) interaction — [Lee TS & Nguyen M (2001) "Dynamics of subjective contour formation in the early visual cortex." PNAS 98:1907–1911](https://doi.org/10.1073/pnas.98.4.1907)
- [Mq, optical imaging + units] V4 orientation domains for real and illusory contours overlapped completely. Orientation domains in V1/V2 mainly encoded the local inducing features — [Pan Y, Chen M, Yin J, An X, Zhang X, Lu Y, Gong H, Li W & Wang W (2012) "Equivalent representation of real and illusory contours in macaque V4." J Neurosci 32:6760–6770](https://doi.org/10.1523/JNEUROSCI.6140-11.2012)
- [Mq] Amodal completion of contours behind occluders in V2 depends on the depth order of the interrupting line (crossing blocks, near occluder restores; see Q3) — [Bakin, Nakayama & Gilbert (2000) J Neurosci 20:8188](https://doi.org/10.1523/JNEUROSCI.20-21-08188.2000); V1 analogue with disparity-defined occluders — [Sugita (1999) Nature 401:269](https://doi.org/10.1038/45785)
- [Mq] Occlusion-cue (line-termination) stimuli drive polarity-invariant figure-ground signals in prestriate cortex, explained by the same model as illusory contours — [Baumann, van der Zwan & Peterhans (1997) Eur J Neurosci 9:1290](https://doi.org/10.1111/j.1460-9568.1997.tb01484.x)
- [Mq] BO population signals for illusory-contour stimuli (Mooney faces) appear at longer latencies than for real edges — [Hesse & Tsao (2016) J Neurosci 36:11338](https://doi.org/10.1523/JNEUROSCI.1857-16.2016)
- [Hu, high-density EEG + source analysis + fMRI] For centrally presented illusory contours, the earliest effect (88–100 ms) localized to bilateral LOC, 40 ms after visual response onset; it shifted to about 120 ms for lateral presentation. The authors propose that V1/V2 illusory-contour sensitivity reflects feedback from LOC — [Murray MM, Wylie GR, Higgins BA, Javitt DC, Schroeder CE & Foxe JJ (2002) "The spatiotemporal dynamics of illusory contour processing." J Neurosci 22:5055–5073](https://doi.org/10.1523/JNEUROSCI.22-12-05055.2002)
- [Hu, EEG] Modal and amodal completion share a common initial mechanism at about 140 ms (response-strength modulation in LOC and parietal areas). They diverge at about 240 ms, when amodal completion relies on continued strong responses — [Murray MM, Foxe DM, Javitt DC & Foxe JJ (2004) "Setting boundaries: brain dynamics of modal and amodal illusory shape completion in humans." J Neurosci 24:6898–6903](https://doi.org/10.1523/JNEUROSCI.1996-04.2004)
- [Hu, fMRI adaptation with brief masked exposure] The early visual-cortex representation of occluded shapes evolves over a few hundred ms, from an image-based (mosaic) representation to the amodally completed one — [Rauschenberger R, Liu T, Slotnick SD & Yantis S (2006) "Temporally unfolding neural representation of pictorial occlusion." Psychol Sci 17:358–364](https://doi.org/10.1111/j.1467-9280.2006.01711.x)

### Inferences
- **Line-end alignment detector.** The V2 mechanism suggests a fixed-weight "occluding-border" unit: sum the single-ended ES responses along a line orthogonal to their orientation, with a minimum count of aligned terminations. When several vessel stems all terminate on one straight path, this flags an occluding structure or an image boundary (e.g. the optic disc rim, or a tool/catheter edge) rather than a set of independent vessel ends.
- **Gap bridging limits.** Bridging spans of 1–3.5° near the fovea (Peterhans & von der Heydt 1989) are several times the RF size. Contour completion across gaps (stenoses, occlusions, imaging dropouts) can therefore use a collinear bipole kernel extending several RF lengths. Thin "closing lines" abolish completion, so a fixed design should suppress completion when a terminating cross-line (a T-bar) closes the gap.
- **Timing.** Illusory/amodal signals appear in V2 before V1 (70 vs 100 ms) and in LOC by about 90–140 ms. A network mimicking this would put completion in a second-level stage feeding back to the line map, rather than in the first filtering stage.

### Gaps
- No physiological data were found on illusory or amodal completion specifically for **thin crossing lines** (one line passing behind another), apart from Bakin 2000 and Sugita 1999 (bars with disparity-defined occluders).
- The V1 illusory-contour question is unresolved; the conflicting primary sources are listed above.

---

## 5. Contextual modulation and horizontal connections in V1: collinear facilitation, iso-orientation surround suppression, cross-orientation effects, topography near pinwheels, and what V1 does at crossings

### Takeaway
V1 contextual modulation has the layout of an "association field". Collinear, co-oriented flankers facilitate (42% of superficial complex cells in alert macaque), and the facilitation is located at the RF **ends**. Parallel or flanking stimuli on the RF **sides** inhibit. Iso-orientation surrounds suppress strongly (94% of cells, mean 67% reduction), while orthogonal surrounds suppress less and sometimes facilitate. At a **crossing**:
- An orthogonal line placed between two collinear segments **abolishes** collinear facilitation, both perceptually and in V1.
- A superimposed orthogonal orientation suppresses ~94% of oriented superficial V1 cells (cross-orientation inhibition).
- A separate non-oriented population responds to the plaid (X) configuration.

Horizontal connections link co-oriented, co-axially aligned columns: shown for tree shrew, with axis anisotropy in New World monkeys. **Macaque anisotropy is contested.** Their extent matches the summation field. Feedback from V2 and beyond is needed to explain the far surround and fast suppression.

### Cited Findings
**Collinear facilitation and spatial layout**
- [Hu + Mq, alert, superficial complex cells] Human contrast detection improved 40% with a second suprathreshold collinear bar. In V1, **42% of complex cells** showed facilitation from a second bar outside the classical RF, with similar dependence on separation, collinearity and relative orientation. "**Both effects were eliminated by an orthogonal line between the two iso-oriented lines.**" Surround inhibition from random lines could be removed by making a few elements collinear with the target — [Kapadia MK, Ito M, Gilbert CD & Westheimer G (1995) "Improvement in visual sensitivity by changes in local context: parallel studies in human observers and in V1 of alert monkeys." Neuron 15:843–856](https://doi.org/10.1016/0896-6273(95)90175-2)
- [Mq + Hu] Opposing contextual interactions are spatially segregated: excitatory interactions along the RF **ends** (collinear axis) and inhibitory interactions strongest along the **orthogonal axis** (flanks), matching human psychophysics — [Kapadia MK, Westheimer G & Gilbert CD (2000) "Spatial distribution of contextual interactions in primary visual cortex and in visual perception." J Neurophysiol 84:2048–2062](https://doi.org/10.1152/jn.2000.84.4.2048)
- [Cat — foundational] Facilitation occurs preferentially when a **near-threshold** center is flanked by higher-contrast collinear elements. Both collinear and orthogonal flanks *reduce* responses to high-contrast centers — [Polat U, Mizobe K, Pettet MW, Kasamatsu T & Norcia AM (1998) "Collinear stimuli regulate visual responses depending on cell's contrast threshold." Nature 391:580–584](https://doi.org/10.1038/35372)
- [Mq, task] V1 responses correlate with perceptual contour saliency; single-neuron ROC discrimination matched behavior; responses were weaker for undetected or unattended contours — [Li W, Piëch V & Gilbert CD (2006) "Contour saliency in primary visual cortex." Neuron 50:951–962](https://doi.org/10.1016/j.neuron.2006.04.035)
- [Mq, simultaneous V1+V4] Global-contour information in clutter emerged first in V4 and about **40 ms later in V1**, then developed in parallel in both areas. This supports incremental integration (feedforward assembly plus feedback disambiguation) — [Chen M, Yan Y, Gong X, Gilbert CD, Liang H & Li W (2014) "Incremental integration of global contours through interplay between visual cortical areas." Neuron 82:682–694](https://doi.org/10.1016/j.neuron.2014.03.023)
- [Hu + Mq, fMRI adaptation] Collinear-contour selectivity is present in early visual areas, depending on RF size, as well as in higher shape areas — [Kourtzi Z, Tolias AS, Altmann CF, Augath M & Logothetis NK (2003) "Integration of local features into global shapes: monkey and human fMRI studies." Neuron 37:333–346](https://doi.org/10.1016/s0896-6273(02)01174-1)
- [Review] Top-down inputs select among components of the association field, so the same V1 neuron can take different functional roles depending on task — [Gilbert CD & Li W (2013) "Top-down influences on visual processing." Nat Rev Neurosci 14:350–363](https://doi.org/10.1038/nrn3476)

**Horizontal connection anatomy (with species flags and conflicts)**
- [TS — not primate] Beyond 500 µm from the injection site, horizontal connections show both modular (iso-orientation) specificity and **axis specificity**: longer and denser along the visuotopic axis matching the preferred orientation. Within 500 µm, connections are much less specific and reach all orientations. Summary: connections link "co-oriented, co-axially aligned receptive fields" — [Bosking WH, Zhang Y, Schofield B & Fitzpatrick D (1997) "Orientation selectivity and the arrangement of horizontal connections in tree shrew striate cortex." J Neurosci 17:2112–2127](https://doi.org/10.1523/JNEUROSCI.17-06-02112.1997)
- [TS] Layer 2/3 (not layer 4) neurons show elongated RFs, length summation and axis- and orientation-specific facilitation matching horizontal-connection layout. Collinear Gabor arrays evoked stronger population responses than non-collinear arrays, *without* filling in between elements — [Chisum HJ, Mooser F & Fitzpatrick D (2003) "Emergent properties of layer 2/3 neurons reflect the collinear arrangement of horizontal connections in tree shrew visual cortex." J Neurosci 23:2947–2960](https://doi.org/10.1523/JNEUROSCI.23-07-02947.2003)
- [NWM: owl and squirrel monkeys] Layer-3 axons spread **anisotropically**, elongated along the preferred orientation of the injection site. Near the fovea, most terminals fall within or just outside the classical RF's cortical map — [Sincich LC & Blasdel GG (2001) "Oriented axon projections in primary visual cortex of the monkey." J Neurosci 21:4416–4426](https://doi.org/10.1523/JNEUROSCI.21-12-04416.2001)
- **Conflict** — [Mq] Monosynaptic V1 horizontal connections match the **summation field** and contrast-dependent changes in its size, but "cannot fully account for the dimensions of the surround field". Feedback from extrastriate cortex matches the full center-surround range. In macaque, V1 horizontal connections are **isotropic**, while feedback covers anisotropic regions and "may play a role in contour completion" — [Angelucci A, Levitt JB, Walton EJS, Hupé JM, Bullier J & Lund JS (2002) "Circuits for local and global signal integration in primary visual cortex." J Neurosci 22:8633–8646](https://doi.org/10.1523/JNEUROSCI.22-19-08633.2002)
- **Conflict** — [Mq, GFP adenovirus labeling] V1 intrinsic connections and V2→V1 feedback both span regions up to 8× the classical RF. Intrinsic connections are an **order of magnitude denser** and **link similarly oriented domains**; V2→V1 feedback shows **no orientation specificity**. The authors favor intrinsic connections as the contour-integration substrate — [Stettler DD, Das A, Bennett J & Gilbert CD (2002) "Lateral connectivity and contextual interactions in macaque primary visual cortex." Neuron 36:739–750](https://doi.org/10.1016/s0896-6273(02)01029-2)
- [Cat] Short-range local connections are radially symmetric and largely orientation-independent. Contextual effects vary with position in the orientation map (pinwheels vs domains), suggesting specialization for corners and T-junctions near pinwheels (Q2) — [Das & Gilbert (1999) Nature 399:655](https://doi.org/10.1038/21371)

**Surround suppression strength, tuning and timing**
- [Mq] 94% (99/105) of V1 cells across all layers were suppressed by stimuli exceeding the classical RF (mean reduction 67%; 43% of cells >70%). Classical surround suppression ("CSS") occurred in 37% of cells and center-gated suppression in 63% — [Jones HE, Grieve KL, Wang W & Sillito AM (2001) "Surround suppression in primate V1." J Neurophysiol 86:2011–2028](https://doi.org/10.1152/jn.2001.86.4.2011)
- [Mq] Iso-oriented surrounds suppress most at the RF ends; orthogonal surrounds suppress most on the flanks (Q1) — [Cavanaugh, Bair & Movshon (2002) J Neurophysiol 88:2547](https://doi.org/10.1152/jn.00693.2001)
- [Mq] Timing of surround suppression:
  - Strong suppression arrived on average **30 ms earlier** than weak suppression, and sometimes faster than the excitatory classical-RF response.
  - Nearby surrounds gave sustained suppression; distant surrounds gave transient suppression.
  - Orientation-tuned suppression propagated across **6–8 mm of cortex at about 1 m/s**, "considerably faster than expected for horizontal cortical connections". The authors invoke feedback from cells with large RFs — [Bair W, Cavanaugh JR & Movshon JA (2003) "Time course and time-distance relationships for surround suppression in macaque V1 neurons." J Neurosci 23:7690–7701](https://doi.org/10.1523/JNEUROSCI.23-20-07690.2003)
- [Mq] V2 RFs and surrounds are twice as large as in V1. In both areas surrounds are about 5× the RF; the far surround can exceed 12.5° radius (mean 5.5° in V1, 9.2° in V2). As in V1, V2 horizontal connections match the RF center, and far surrounds likely derive from feedback — [Shushruth S, Ichida JM, Levitt JB & Angelucci A (2009) "Comparison of spatial summation properties of neurons in macaque V1 and V2." J Neurophysiol 102:2069–2083](https://doi.org/10.1152/jn.00512.2009)
- [Mq] Inactivating MT showed that feedback amplifies responses inside the classical RF, enhances background-evoked surround suppression, and matters most for low-salience stimuli — [Hupé JM, James AC, Payne BR, Lomber SG, Girard P & Bullier J (1998) "Cortical feedback improves discrimination between figure and background by V1, V2 and V3 neurons." Nature 394:784–787](https://doi.org/10.1038/29537)

**What V1 does where two orientations co-occur (crossings)**
- [Mq] An orthogonal line between two collinear bars **eliminates** collinear facilitation in V1 and in perception — [Kapadia et al. (1995) Neuron 15:843](https://doi.org/10.1016/0896-6273(95)90175-2)
- [Mq] In V2 amodal completion, a crossing orthogonal line blocks facilitation unless it is in the near depth plane (an occluder) — [Bakin et al. (2000) J Neurosci 20:8188](https://doi.org/10.1523/JNEUROSCI.20-21-08188.2000)
- [Mq] ~94% of oriented superficial V1 cells show cross-orientation inhibition to plaids; a separate non-oriented population responds selectively to plaids, with cross-angle tuning — [Guan et al. (2020) Curr Biol 30:934](https://doi.org/10.1016/j.cub.2020.01.005)
- [Cat + Mq] Orientation-discontinuity (center ≠ surround orientation) configurations give enhanced or "supra-optimal" responses in some patch-suppressed cells, proposed as junction or corner detectors — [Sillito et al. (1995) Nature 378:492](https://doi.org/10.1038/378492a0)

### Inferences
- **Association-field kernel for fixed weights.** Excitation along the collinear axis at the RF ends, contrast-gated so it is strongest for near-threshold (faint) segments (Polat 1998). Inhibition on the flanks and from iso-oriented stimuli beyond the summation field (Kapadia 2000; Cavanaugh 2002). This maps directly onto a "bipole" lateral kernel for vessel enhancement. Spatial extent: about the summation field for horizontal connections (Angelucci 2002), and up to ~5× RF for feedback-mediated far surround (Shushruth 2009).
- **The crossing problem.** In biology, a 2-D orthogonal crossing *breaks* collinear facilitation (Kapadia 1995) and cross-orientation inhibition suppresses both oriented channels (Guan 2020). Perceptual continuity through a crossing therefore probably depends on:
  - (a) a separate co-occurrence (plaid) channel that labels the location as a crossing;
  - (b) depth or occlusion interpretation (Bakin 2000);
  - (c) feedback from higher areas, where contour signals lead V1 by about 40 ms (Chen 2014).
  
  For a vessel network, one design option is orientation-channel-specific divisive normalization at crossings: normalize *across* orientations only partially, so each orientation's collinear chain survives the crossing, plus an explicit X-detector that gates (disables) cross-channel suppression where two strong orientations co-occur. This is a proposal.
- **Isotropic vs oriented lateral wiring.** Macaque horizontal-connection anisotropy is disputed (isotropic: Angelucci 2002; orientation-specific but axis unclear: Stettler 2002; anisotropic in NWM: Sincich 2001; strongly co-axial in tree shrew: Bosking 1997). A design need not mimic one species. For tracing through crossings, co-axial, co-oriented connectivity is functionally required (see Q6).

### Gaps
- No primate single-unit study was found that measures V1 responses to a **thin-line X-crossing** placed in the RF (as opposed to gratings, plaids or flanker paradigms), or that compares X-crossing with T- or Y-configurations under matched conditions.
- Das & Gilbert's pinwheel-topography result is cat. I did not find whether macaque pinwheel-center neurons respond preferentially to corners or junctions.
- Polat et al. 1998 is cat. The primate collinear-facilitation data used here are Kapadia et al. 1995/2000 (alert macaque).

---

## 6. Curve tracing and object-based attention spreading along curves in V1: how does enhancement propagate, especially through crossings vs branch points?

### Takeaway
During mental curve tracing, macaque V1 neurons whose RFs lie on the **target** curve respond more strongly than those on a distractor curve. The enhancement holds **even when the two curves cross** (Roelfsema et al. 1998), so the label propagates through an X-crossing along the target's own continuation. The enhancement is absent from the initial transient; it appears after a delay (about 150 ms per a later summary) and spreads gradually from the cued end. Spread speed is about 50 ms per receptive field: slower where curves are close together and faster where they are far apart. This is explained by an "attentional growth cone" spreading across multiple areas with different RF sizes. Human tracing time grows with curve length. Mechanistically, Roelfsema proposes **incremental grouping**: response enhancement spread through horizontal and feedback connections, contrasted with fast feedforward "base grouping". **No neurophysiology was found that compares propagation through crossings vs through branch points (Y/T).**

### Cited Findings
- [Mq, V1] In monkeys trained on a curve-tracing task, "neuronal responses to the various segments of a target curve were simultaneously enhanced relative to responses evoked by a distractor curve, **even if the two curves crossed each other**" — [Roelfsema PR, Lamme VAF & Spekreijse H (1998) "Object-based attention in the primary visual cortex of the macaque monkey." Nature 395:376–381](https://doi.org/10.1038/26475)
- [Mq, V1] Spread dynamics:
  - The target curve was traced mentally while a distractor was ignored.
  - Enhancement was **delayed for RFs farther from the start** of tracing, which allowed propagation speed to be measured.
  - Speed was **low when curves were nearby and increased when they were far apart**.
  - Explained by an "attentional growth-cone" model in which enhancement spreads "in multiple visual cortical areas with different receptive field sizes at a speed of **approximately 50 ms per receptive field**" — [Pooresmaeili A & Roelfsema PR (2014) "A growth-cone model for the spread of object-based attention during contour grouping." Curr Biol 24:2869–2877](https://doi.org/10.1016/j.cub.2014.10.007)
- [Review/Model; full text read] Summary of the curve-tracing physiology:
  - Enhancement "does not appear in the initial transient response" and emerges after **~150 ms**.
  - It starts at the cued location and spreads gradually, at about 50 ms per RF.
  - It "cannot spread to blank image locations in between curves".
  - Model implementation: **modulatory horizontal connections** that affect only units already receiving feedforward excitation, which stops spread into the background, together with balanced excitation and disynaptic inhibition. Networks without horizontal linking connections could not learn tracing beyond 3-pixel lines.
  - The paper does not address crossings — [Brosch T, Neumann H & Roelfsema PR (2015) "Reinforcement learning of linking and tracing contours in recurrent neural networks." PLoS Comput Biol 11(10):e1004489](https://doi.org/10.1371/journal.pcbi.1004489)
- [Review] Two grouping mechanisms:
  - **Base-grouping**: fast, coded by feedforward selectivity of neurons tuned to feature conjunctions.
  - **Incremental grouping**: slower, flexible, implemented by enhancing responses of neurons coding features bound in perception through **horizontal and feedback connections**. The enhancement corresponds to attention spreading over the object — [Roelfsema PR (2006) "Cortical algorithms for perceptual grouping." Annu Rev Neurosci 29:203–227](https://doi.org/10.1146/annurev.neuro.29.051605.112939)
- [Review] Ullman's "visual routines": elemental operations (contour grouping, search, texture segregation) implemented by firing-rate modulation that groups responses into coherent objects and passes information between operations — [Roelfsema PR, Lamme VAF & Spekreijse H (2000) "The implementation of visual routines." Vision Res 40:1385–1411](https://doi.org/10.1016/s0042-6989(00)00004-3)
- [Hu, psychophysics] Attention spreads gradually from previously labeled contour segments to segments that are "colinear and connected to them"; grouping completes when the whole curve is attended — [Houtkamp R, Spekreijse H & Roelfsema PR (2003) "A gradual spread of attention during mental curve tracing." Percept Psychophys 65:1136–1144](https://doi.org/10.3758/bf03194840)
- [Hu, psychophysics] Curve tracing proposed as a basic visual operation. Response time increases with distance along the curve; the PubMed record has no abstract — [Jolicoeur P, Ullman S & Mackay M (1986) "Curve tracing: a possible basic operation in the perception of spatial relations." Mem Cognit 14:129–140](https://doi.org/10.3758/bf03198373)
- [Mq, V1] Attentional response modulation spreads **automatically** to image elements outside the attentional focus when they are bound to the attended stimulus by Gestalt criteria — [Wannig A, Stanisor L & Roelfsema PR (2011) "Automatic spread of attentional response modulation along Gestalt criteria in primary visual cortex." Nat Neurosci 14:1243–1244](https://doi.org/10.1038/nn.2910)
- [Mq, V1 laminar] Feedforward response at about 40 ms in layers 4 and 6; boundary signals at about 70 ms in superficial layers (horizontal); feedback-driven filling-in about 30 ms later through layers 1, 2 and 5 — [Self et al. (2013) Curr Biol 23:2121](https://doi.org/10.1016/j.cub.2013.09.013)
- [Mq, V1] For n/u shapes, a "proto-ground" region was first enhanced with the figure (about 100 ms) and then suppressed back to background level after about 115 ms. This occurred only in trained animals and predicted choice. Attention did not change proto-ground suppression — [Self MW, Jeurissen D, van Ham AF, van Vugt B, Poort J & Roelfsema PR (2019) "The segmentation of proto-objects in the monkey primary visual cortex." Curr Biol 29:1019–1029.e4](https://doi.org/10.1016/j.cub.2019.02.016)

### Inferences
- **Label propagation rule for vessel tracing.** Implement tracing as iterative multiplicative gating:
  - A "label" map L is updated as L ← L + g · (lateral kernel ⊛ L) ⊙ feedforward-drive.
  - Lateral linking only acts where feedforward vessel evidence exists (Brosch 2015's modulatory connections), so the label never leaks into background.
  - Iteration count corresponds to time; roughly one RF per ~50 ms step in biology (Pooresmaeili & Roelfsema 2014).
- **Crossings.** For the label to stay on the target through an X-crossing (Roelfsema 1998), the lateral kernel must be **orientation-specific and co-axial**: a unit labeled at orientation θ passes the label only to units of similar orientation along the θ axis. An isotropic or orientation-independent kernel would leak the label into the crossing vessel. The co-axial, iso-orientation horizontal-connection anatomy (Bosking 1997; Sincich 2001; Stettler 2002) is exactly the substrate needed. This is an inference; the papers do not test crossings mechanistically.
- **Branch points.** At a true bifurcation (Y), the target's continuation splits into two branches, each deviating in orientation from the parent. With a co-axial kernel of moderate angular tolerance (e.g. ±30–45°), the label spreads into both daughter branches, which is desired for vessel-tree labeling. At an X, the crossing vessel differs by a large angle and is excluded. The **angular tolerance of the linking kernel is the key parameter** separating branch-following from crossing-leakage. Houtkamp et al. 2003's "colinear and connected" rule implies that both cues matter. These numbers are a design suggestion, not measured values.
- **Crowding slows propagation.** Pooresmaeili & Roelfsema 2014 found slower spread where curves are near each other; the growth cone shrinks to finer-scale (smaller-RF) representations. A multiscale implementation would propagate on coarse scales where vessels are isolated and drop to fine scales near crossings and parallel vessel pairs.

### Gaps
- **No primary neurophysiology was found that compares label propagation through X-crossings vs Y-branch points**, or tracing along branching (tree-like) structures in V1.
- The precise onset latency of tracing enhancement in Roelfsema et al. 1998 was not verified from the primary paper. The "~150 ms" figure comes from Brosch et al. 2015's summary.
- Crossing geometry (crossing angle, contrast) effects on V1 label propagation were not found.

---

## 7. Human neuroimaging of junction / vertex processing (fMRI, EEG, MEG)

### Takeaway
Human behavior and fMRI indicate that **contour-junction statistics** (junction types L, T, X, Y, arrow and their angles) are the critical cue for scene categorization from line drawings:
- Scrambling junctions (shifting contours so they no longer meet) drastically reduces decoding of scene category in PPA and OPA but **does not affect early visual cortex decoding**.
- Searchlight analyses show length and curvature statistics dominate early visual areas, and junction statistics dominate scene-selective areas.

For objects, LOC represents perceived shape (parts and medial-axis relations) rather than local vertices or lines (fMRI adaptation). Classic behavior shows that deleting contour at vertices impairs recognition far more than deleting midsegments. Human V2 shows attention-dependent border-ownership selectivity. EEG localizes illusory and amodal completion to LOC at 90–140 ms. **No EEG/MEG study isolating responses to specific junction types (T vs L vs X) was found.**

### Cited Findings
- [Hu, fMRI] Scene categories could be decoded from line drawings as well as from photographs, from V1 through PPA and RSC; in PPA and RSC, error patterns for drawings and photographs were very similar. Behaviorally, removing **long** contours hurt categorization more than removing short ones — [Walther DB, Chai B, Caddigan E, Beck DM & Fei-Fei L (2011) "Simple line drawings suffice for functional MRI decoding of natural scene categories." PNAS 108:9661–9666](https://doi.org/10.1073/pnas.1015666108)
- [Hu, behavior + computation] Contour properties (orientation, length, curvature) and **junction properties (types and angles)** were extracted from line drawings. All carried category information computationally, but only **junctions and curvature** predicted human error patterns. Perturbing junctions by randomly shifting contours significantly reduced accuracy. The authors note "These properties correspond to the visual features represented in area V2" — [Walther DB & Shen D (2014) "Nonaccidental properties underlie human categorization of complex natural scenes." Psychol Sci 25:851–860](https://doi.org/10.1177/0956797613512662)
- [Hu, fMRI MVPA] Line drawings were manipulated to disrupt orientation statistics or junction statistics:
  - "Disruption of junctions but not orientations led to a drastic decrease in decoding accuracy in the PPA and OPA". Early visual-cortex decoding "was unaffected by either image manipulation".
  - The correlation between neural and behavioral error patterns depended on intact junctions.
  - Searchlight: "Statistics of contour length and curvature dominate neural representations of scene categories in early visual areas and contour junctions in high-level scene-selective brain regions" — [Choo H & Walther DB (2016) "Contour junctions underlie neural representations of scene categories in high-level human visual cortex." NeuroImage 135:32–44](https://doi.org/10.1016/j.neuroimage.2016.04.021)
- [Hu, behavior — classic; abstract not retrieved this session] Recognition-by-components: nonaccidental properties (collinearity, curvilinearity, symmetry, parallelism, cotermination) and junction types (L, fork/Y, arrow, T) are proposed as the basis of geon recovery. Line drawings with contour deleted at **vertices** (concavities and cusps) are far harder to recognize than drawings with equal amounts deleted at midsegments — [Biederman I (1987) "Recognition-by-components: a theory of human image understanding." Psychol Rev 94:115–147](https://doi.org/10.1037/0033-295X.94.2.115)
- [Hu, fMRI adaptation] With complementary contour-deleted images, "virtually all the adaptation in LOC [especially pFs] could be attributed to repetition of the parts, almost none to the repetition of local image features, such as **lines or vertices**" — [Hayworth KJ & Biederman I (2006) "Neural evidence for intermediate representations in object recognition." Vision Res 46:4024–4031](https://doi.org/10.1016/j.visres.2006.07.015)
- [Hu, fMRI adaptation] LOC adapted when perceived shape was the same but contours differed, and not when contours were the same but perceived shape differed (figure-ground reversal). LOC therefore represents perceived shape, not simple image features — [Kourtzi Z & Kanwisher N (2001) "Representation of perceived object shape by the human lateral occipital complex." Science 293:1506–1509](https://doi.org/10.1126/science.1061133)
- [Hu, fMRI MVPA] By V3, medial-axis structure (relations between parts) was classified better than orientation, indicating a change in shape representation beyond early areas — [Lescroart MD & Biederman I (2013) "Cortical representation of medial axis structure." Cereb Cortex 23:629–637](https://doi.org/10.1093/cercor/bhs046)
- [Hu, fMRI] Border-ownership selectivity in V2 is attention-dependent (Q3) — [Fang, Boyaci & Kersten (2009) J Neurosci 29:460](https://doi.org/10.1523/JNEUROSCI.4628-08.2009)
- [Hu, EEG/fMRI] Illusory-contour effects at 88–100 ms in LOC (Murray 2002). Modal and amodal completion share a 140 ms stage and diverge at 240 ms (Murray 2004). Amodal completion in early visual cortex unfolds over a few hundred ms (Rauschenberger 2006). Full citations in Q4.
- [Hu + Mq, fMRI] Collinear-contour selectivity is present in early and higher visual areas — [Kourtzi et al. (2003) Neuron 37:333](https://doi.org/10.1016/s0896-6273(02)01174-1)

### Inferences
- **Early visual cortex is junction-insensitive at the voxel-pattern level.** In humans, junction statistics are critical for categorization, yet voxel patterns in early visual cortex are unaffected by junction disruption (Choo & Walther 2016). This suggests that explicit, population-level junction coding emerges at intermediate or high levels, while V1/V2 single-unit junction sensitivity (Q2) is too sparse or local to dominate fMRI patterns. For network design: junction labels plausibly sit at a stage after orientation, end-stopping and curvature maps (a V2/V4-like layer), not in the first filter bank.
- **Vertices carry the junction-type information.** The vertex-deletion effect (Biederman 1987) implies that junction-type information is concentrated at vertices. Junction detection in vessel images should therefore preserve fine resolution at candidate vertices, not blur them away with large smoothing kernels.

### Gaps
- **No EEG or MEG study that isolates neural responses to specific junction types (T vs L vs X vs Y) in line drawings was found** in this session's searches. A partial hit (an L-junction-pattern study reporting a small V1 effect) could not be identified well enough to cite.
- No human fMRI study was found that contrasts crossing (X) vs branching (Y) configurations.

---

## 8. Cross-cutting: what quantitative receptive-field, connectivity and timing parameters are reported (for fixed-weight network design)?

### Takeaway
The data support a 3-phase timeline in macaque V1/V2:
1. **Feedforward drive**: about 40–45 ms after stimulus onset.
2. **Local context / boundary signals**: about 60–70 ms (local edge polarity and depth within ~13 ms of edge onset; boundary signals at ~70 ms; BO ~10–30 ms after response onset).
3. **Feedback-driven grouping**: object-level enhancement, filling-in and curve tracing at roughly 100–150+ ms; tracing spreads about 50 ms per RF.

Horizontal fibers conduct slowly (activity spreads at 0.1–0.25 m/s). Inter-areal feedforward and feedback axons are fast (~3.5 m/s). This is why fast, distance-insensitive context effects (BO, far-surround suppression) are attributed to feedback. Horizontal connections span roughly the summation field. Feedback spans the full surround, about 5× the RF in V1 and V2.

### Cited Findings
- [Mq] Feedforward and feedback axons between V1 and V2 have comparable fast conduction velocities, **~3.5 m/s** — [Girard P, Hupé JM & Bullier J (2001) "Feedforward and feedback connections between areas V1 and V2 of the monkey have similar rapid conduction velocities." J Neurophysiol 85:1328–1331](https://doi.org/10.1152/jn.2001.85.3.1328)
- [Mq, real-time optical imaging] Activity from a small stimulus spread across an area ≥10× larger in upper layers, at **100–250 µm/ms** (0.1–0.25 m/s). Near the V1/V2 border the space constant was ~2.7 mm parallel to the border and ~1.5 mm perpendicular. All surround stimuli suppressed the center response, with iso-orientation suppression sometimes stronger than orthogonal — [Grinvald A, Lieke EE, Frostig RD & Hildesheim R (1994) "Cortical point-spread function and long-range lateral interactions revealed by real-time optical imaging of macaque monkey primary visual cortex." J Neurosci 14:2545–2568](https://doi.org/10.1523/JNEUROSCI.14-05-02545.1994)
- [Cat — foundational] Synaptic integration fields (intracellular) are much larger than spiking RFs. Subthreshold response latency increases with distance from the RF center, consistent with slowly conducting horizontal axons — [Bringuier V, Chavane F, Glaeser L & Frégnac Y (1999) "Horizontal propagation of visual activity in the synaptic integration field of area 17 neurons." Science 283:695–699](https://doi.org/10.1126/science.283.5402.695)
- [Mq] Orientation-tuned surround suppression propagates across 6–8 mm of cortex at **~1 m/s**, too fast for horizontal fibers; strong suppression arrives ~30 ms earlier than weak suppression — [Bair, Cavanaugh & Movshon (2003) J Neurosci 23:7690](https://doi.org/10.1523/JNEUROSCI.23-20-07690.2003)
- [Mq] Spatial scales:
  - Horizontal connections match the **summation field**; feedback matches the **full surround**.
  - Macaque horizontal connections are isotropic; feedback is anisotropic — [Angelucci et al. (2002) J Neurosci 22:8633](https://doi.org/10.1523/JNEUROSCI.22-19-08633.2002)
  - Surrounds are ~5× the RF; far surround averages 5.5° (V1) and 9.2° (V2); V2 RFs are 2× V1 — [Shushruth et al. (2009) J Neurophysiol 102:2069](https://doi.org/10.1152/jn.00512.2009)
  - Intrinsic V1 connections and V2 feedback both reach ~8× the classical RF; intrinsic connections are ~10× denser and orientation-specific; feedback is not orientation-specific — [Stettler et al. (2002) Neuron 36:739](https://doi.org/10.1016/s0896-6273(02)01029-2)
- [Mq] Summation extent is 2.3× larger at low contrast — [Sceniak et al. (1999) Nat Neurosci 2:733](https://doi.org/10.1038/11197). Surround suppression is strongest in layer 4B and weakest in layer 6 — [Sceniak et al. (2001) J Neurophysiol 85:1873](https://doi.org/10.1152/jn.2001.85.5.1873)
- [Mq] Timing benchmarks:
  - V1 feedforward response ~40 ms (layers 4 and 6); boundary signals ~70 ms; feedback filling-in ~100 ms — [Self et al. (2013)](https://doi.org/10.1016/j.cub.2013.09.013)
  - Figure-ground enhancement 30–40 ms after response onset — [Lamme (1995)](https://doi.org/10.1523/JNEUROSCI.15-02-01605.1995)
  - BO <25 ms after response onset — [Zhou et al. (2000)](https://doi.org/10.1523/JNEUROSCI.20-17-06594.2000)
  - Local edge definition <13 ms vs BO context ~30 ms after edge onset — [Sugihara et al. (2011)](https://doi.org/10.1152/jn.00928.2010)
  - V2 natural-scene BO at 60 ms vs onset 44 ms; context at 73 ms — [Williford & von der Heydt (2016)](https://doi.org/10.1523/ENEURO.0127-16.2016)
  - Illusory contours: V2 70 ms, V1 100 ms (superficial) — [Lee & Nguyen (2001)](https://doi.org/10.1073/pnas.98.4.1907)
  - V4 curvature tuning emerges over ~50 ms — [Yau et al. (2013)](https://doi.org/10.1093/cercor/bhs004)
  - Global contour information in V4 ~40 ms before V1 — [Chen et al. (2014)](https://doi.org/10.1016/j.neuron.2014.03.023)
  - Curve-tracing spread ~50 ms per RF — [Pooresmaeili & Roelfsema (2014)](https://doi.org/10.1016/j.cub.2014.10.007)
  - Tracing enhancement onset ~150 ms (secondary summary) — [Brosch et al. (2015)](https://doi.org/10.1371/journal.pcbi.1004489)
  - Proto-ground suppression after ~115 ms — [Self et al. (2019)](https://doi.org/10.1016/j.cub.2019.02.016)
  - Human EEG: illusory contours in LOC at 88–100 ms — [Murray et al. (2002)](https://doi.org/10.1523/JNEUROSCI.22-12-05055.2002); modal/amodal completion common stage 140 ms, divergence 240 ms — [Murray et al. (2004)](https://doi.org/10.1523/JNEUROSCI.1996-04.2004)

### Inferences
- **Suggested fixed-weight architecture** (a synthesis; each stage is anchored to the cited physiology, but the overall design is a proposal):
  1. **Stage 1 (V1 simple/complex, "~40 ms")**: even/odd oriented filters at multiple scales, energy (C) units, and separate ON/OFF polarity channels. Divisive contrast normalization pooled over orientation (cross-orientation inhibition), with partial normalization so crossings keep both orientations.
  2. **Stage 2 (V1/V2 end-stopped and curvature, "~50–70 ms")**: single- and double-ended end-stopped operators (Heitger 1992) with polarity-specific end zones (Yazdanbakhsh & Livingstone 2006), plus curvature from ES differences (Dobbins 1987). A non-oriented co-occurrence (plaid/X) detector (Guan 2020). Collinear association-field lateral kernel (Kapadia 2000), spanning about the summation field.
  3. **Stage 3 (V2/V4 junction and angle units, "~70–120 ms")**: a ray (half-line) basis around keypoints, combined linearly with suppression of non-preferred rays and divisive gain by ray count (Ito & Goda 2011). This classifies end / L / T / Y / X / arrow. Curve-vs-corner smoothness split (Jiang 2021). Line-end-alignment "occluding border" units (von der Heydt & Peterhans 1989).
  4. **Stage 4 (grouping-cell feedback, "~100–150+ ms")**: large cocircular pooling (G cells) feeding back multiplicatively to assign border ownership and occlusion (Craft 2007). An iterative, orientation-specific, co-axial, feedforward-gated label propagation for vessel tracing (Roelfsema 1998; Pooresmaeili & Roelfsema 2014; Brosch 2015), with angular tolerance tuned so labels follow branches but not crossings.
- **Timing as iteration counts.** Horizontal spread at 0.1–0.25 m/s versus feedback at ~3.5 m/s implies that, in a discrete-time network, a few feedback iterations should carry long-range context. Lateral (horizontal) kernels should only be iterated for short-range linking. This matches the biological finding that BO latency is nearly size-independent.

### Gaps
- No primate measurements were found of **horizontal-connection angular tolerance** (the orientation difference over which co-axial linking still operates), which is a key design parameter for branch vs crossing discrimination.
- No direct primate measurement was found of end-stopping latency or of junction-type-specific response latencies.
