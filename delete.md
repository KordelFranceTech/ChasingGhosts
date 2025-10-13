Authors’ Response to Reviews of
Survey - Readme
NeurIPS 2025: Position Papers Track , 566
RC: Reviewers’ Comment , AR: Authors’ Response, □Manuscript Text

1. Section 2 (Adjuducation) - Texfile
1.0. Reviewer #1 - Score: 8

> It seems, however, to have been written in a rush. Perhaps something about possible uses of AI with olfaction could be added to the introduction.  

Thank you very much for your valuable suggestion. We assure you that the authors put a lot of thoughtful time into the paper. While we did include a substantial amount of olfaction use cases in the supplementary section, perhaps this could have been better contextualized earlier in the paper. We have noted your comment for improvement to the manuscript.  

---

> Also, citation of related work is missing at times, such as in line 168. There is a space missing in line 114.  

Thank you for your suggestion. The field of olfaction is still quite niche, so citations were appropriately added where necessary. Upon examination of line 168, we do not feel that sentence warranted a citation because it was an inference based on the consolidated cited facts above. We agree with the reviewer's suggestion that a space should be placed in line 114.  

**Way Forward:** We have noted your suggestion to keep a closer eye on citations and formatting.  

---

> Lines 268 to 278 are crucial for the paper and outline the recommendations, but they are under section 2.4 - *The Case for Olfaction*. Maybe this could be moved so it gets more attention?  
This is valuable insight. We will place it in earlier sections as suggested.  

---

> The paper mentions that the human olfactory system processes information episodically, in "brief, irregular bursts" due to the nature of turbulent plumes. Do you think this should inform the generation/collection of Olfaction Data in any way?  
Excellent question! Yes, this should inform the generation/collection of olfaction data. We mention that olfactory data may benefit from combinatorial encoding in section 2.4, specifically on line 154, and how this should inspire how one thinks about constructing both a data standard and consequential dataset. We also mention the sparse temporal nature of olfactory signals on line 161, which we advise should influence long-form capture of olfaction data.  


1.1. Reviewer #2 - Score: 6
> The paper asserts that olfaction is essential for embodied AI but does not critically examine scenarios
where smell might offer minimal benefit over well-established modalities like vision or audition.

We respectfully disagree with the statement. Our claim is notthat olfaction is universally superior, but that it
is essential in specific embodied settings while offering limited marginal value in others. We state these limits
explicitly:

(1)Section 2.2 (lines 121–147): We first discuss the bandwidth of vision, audition, and olfaction. Then we
note that current state of olfactory sensors does not contain the bandwidth capabilities of its counterparts and
therefore may not be useful in robotics and AI scenarios requiring rapid, high-fidelity scene readouts.

(2)Section 4: We discuss limitations of our position and in the field of machine olfaction in general that
include the likely need for adaptive learning methods (lines 353 - 371), nuances surrounding manufacturing
olfaction hardware (lines 377-382), and the slow sampling speeds of olfaction sensors (lines 380-382). Each
of these points amount support for why smell might offer minimal benefit over well-established modalities.

> The paper outlines data standard definition, annotation protocols, consortium formation and benchmark
creation but provides no actionable details such as timelines or milestones; governance models or funding
mechanisms for the proposed consortium; or concrete procedures for defining and calibrating annotation
taxonomies.

(1) Interesting question! Our choice to omit timelines, milestones, governance, and funding details was delib-
erate. Through this position paper, we aimed to set an agenda rather than prescribe a single implementation
pathway. Also, pre-selecting mechanisms risks hard-coding our biases and constraining community input.

(2) The lack of progress in machine olfaction thus far has partially been attributed to inappropriate timeline
and budget proposals surrounding the complexity of olfactory sensing itself, and our preliminary peer reviews
encouraged us to leave hard numbers out as a result. We highlight existing taxonomies and data standards that
exist such as the current work in time series (lines 241 and 263), principal odor maps (lines 195, 230, 294),
and combinatorial coding (line 154) that should all be taken in to consideration to inspire the community.
We also note their shortcomings and critiques in order to foster more discussion from the AI/ML research
community for improvements.


1.2. Reviewer #3 - Score: 4
> The difference between olfaction as (a) the chemosensory hardware problem (detecting the chemical and
issuing a signal to the brain) and (b) the subjective cognitive experience problem (interpreting the signal,
predicting how molecule in a given concentration smells to humans) is not clearly discussed. The paper
would be stronger if the distinction were clearer - or if the paper re-focused on one of the areas.

(1) We respectfully disagree that our manuscript fails to distinguish between (a) chemosensory hardware
and (b) subjective olfactory experience. The intent of the paper is to facilitate discussion on points for why
olfaction should or should not be included as a primary modality for embodied intelligence, not to argue
(a) how olfaction works biologically which is already a well established argument in the life sciences [6]
[8] [9] [3] [2] [4] (refer Section 2.1: Scientific Understanding of Olfaction ) or (b) re-establish at-length
arguments well articulated in the life sciences for smell objectivity [7] [1] [5] (refer Section 2.3: Objectivity
and Appendix C).
(2) We strongly disagree with the statement that re-focusing on one of (a) or (b) would make the paper stronger
as we re-iterate that our position is that the slow progress and inequity in machine olfaction—especially within
artificial intelligence—can be attributed to five systemic gaps (which includes both (a) and (b)). Narrowing
the scope to only one would undercut our central thesis that addressing even a subset of these gaps—via
foundational research, infrastructure, and cross-disciplinary collaboration—can materially accelerate the field
and bring artificial olfaction closer to parity with vision, audition, and language for embodied intelligence.
While page limits preclude a deeper discussion on chemosensory biology, as the reviewer suggested, we will
incorporate additional content in the Supplementary material to further delineate (a) hardware pipelines and
(b) perceptual modeling.

> The paper claims olfaction research is overlooked, but it is unclear if this point is properly supported.
E.g., the fact that few papers on the topic are published on arxiv could be simply sociological - this
research is more likely to appear in chemistry, electrical engineering, neuroscience journals. It is also
not clear how comprehensive semantic scholar source publication set is. How about bioarxiv? On similar
point, electric nose research has a very long history and while it is mentioned, it feels like the paper
understates the amount of work and findings done in the area - at least, I did not feel I got a comprehensive
enough review of it. A cursory look suggests ML in olfaction is also a reasonably active research area:
https://www.science.org/doi/10.1126/science.aal2014

(1) To clarify, our manuscript does not claim that olfaction research is broadly overlooked. Our claim is
narrower: artificial olfaction within AI and robotics is comparatively underrepresented relative to vision,
audition, and language (see Section 2.1: The Scientific Understanding of Olfaction ). In fact, part of our
argument for the position in the paper is to leverage all of the excellent research done within biological
olfaction to help advance artificial olfaction to a similar magnitude that biological vision, audition, taction,
and speech has help advance their machine analogs. In Section 2.2: Olfaction Data Standard , we highlight this
and elaborate on how AI researchers do not have to start from scratch - but leverage foundational work within
the life sciences to catalyze artificial olfactory developments using knowledge about sensory bandwidths,
encoding schemes, and processing mechanisms.
(2) The paper reviewer cites aligns with our position and discusses about a small dataset from 2017 that
was established for machine olfaction techniques, but unfortunately falls under the same critique for human
objectivity as [5], [1] from Kim, et al. in [7] (refer Section 2.3: Objectivity in our paper). The cited paper
from the reviewer, while in a reputable journal, has also only been cited 220 times - a very small number for a
dataset attempting to define an AI-oriented benchmark in olfaction in comparison to, for example, ImageNet
which has been cited 88,803 times (directly but many fold more through ImageNet pre-trained models such as ResNet) at the time of this writing. So this further complements our argument that artificial olfaction is an
under-served topic in computer science, AI, and robotics.


> The proposal are vague / lack details.

We respectfully disagree with the characterization that our proposal is vague. As a position-track submission,
our goal is to articulate a system-level agenda for machine olfaction in embodied AI—framing priorities and
pathways—rather than to present a single, fully specified solution. The paper outlines concrete directions
spanning data standards, benchmarking, hardware–software integration, and cross-disciplinary infrastructure.
If there are particular areas the reviewer found under-specified, we would welcome that guidance so we can
address it directly.

> The chemosensory side of olfaction problem seems to lie substantially in the hardware limitations - it is not
easy to detect minuscule concentration of many different chemicals at once. Then, should this paper in the
part where it talks about that hardware side of olfaction even be addressed to AI research community?
Perhaps, it is better fitted for material scientists or electrical engineers?

We feel that more careful observance of the paper by the reviewer would have clarified this question as the
contents for its answer exist in the paper. We agree that today’s olfactory sensing challenges are tightly
coupled to hardware. However, we argue this makes the topic squarely relevant to the AI/robotics community:
data characteristics are shaped by sensor physics, and algorithmic advances can both diagnose and compensate
hardware limitations while informing sensor co-design. Our stance parallels the ImageNet-era inflection
in vision (circa 2009): the aim is not to address materials or EE audiences exclusively, but to catalyze
cross-disciplinary work that elevates olfaction to a first-class modality for embodied AI.
Accordingly, while hardware advances are essential, the field advances best when by AI/ML practitioners
collaborate with materials scientists and electrical engineers (lines 377–385). We therefore call for shared data
standards, drift-aware benchmarks, and learning protocols that enable rigorous co-design. This collaborative
path is the most direct route to robust olfactory systems the AI/ML community can adopt with confidence.

> What exactly should the benchmarks and the data sets that you call for look like? Perhaps, you could
provide idealized examples?

AR: Excellent question. A complete specification is beyond this survey’s scope and merits a dedicated paper;
nonetheless, we outline below idealized exemplars of datasets and benchmarks.

**Datasets**
(1) Static multimodal scenes: Synchronized RGB(D)/3D of indoor scenes + raw olfactory streams near
candidate sources; labels for source identity, location, concentration; stepped concentrations.
(2) Dynamic 4D archives: Mobile runs with time-stamped sensor data, robot trajectory, environment geometry,
and controlled source emissions. Controls: standardized calibration, drift logs, ppb sensitivity when relevant;
for subjective labels, multi-rater consensus.

**Benchmarks**
(1) Foundational perception: odorant detection + concentration estimation. Metrics: mAP/F1, RMSE; drift
robustness via cross-day/sensor splits.
(2) Static scene grounding: localize emitting object and olfactory-visual QA. Metrics: 2D/3D localization
error, mAP@IoU, QA accuracy.
(3) Embodied tasks: source localization/tracking, olfaction-aided navigation, event reasoning. Metrics:
time-to-source, success/SPL, ADE/FDE, temporal AP.


2. Section 3 (Review Response) - Texfile
2.1. Reviewer 4FTk — Score: 8
Thoughtful, supportive feedback that mostly targets exposition (organization, citation hygiene).
2.1.1 Thoughtfulness
The reviewer raised a substantive question about how turbulent, episodic plume dynamics should shape data
generation and capture strategies indicating close reading and an intent for meaningful engagement.
2.1.2 Support for Our Position
With a score of 8 and suggestions to surface use cases and relocate recommendations for visibility, the review
reinforces our central claim.
2.1.3 Focus: Technical Aspects vs. Position
Most comments address organization and scholarly apparatus (where recommendations appear, when to cite,
formatting). The episodic-sensing question is technical and conforms with our claims about sparsity, irregular
bursts, and combinatorial encoding.
2.1.4 Level of Gatekeeping
None observed.
2.1.5 Planned Revisions (Actions)
•Elevate recommendations: Move the guidance currently in Section 2.4 (lines 268–278) to an earlier
section to foreground actionable takeaways.
•Surface use cases in the intro: Add representative application scenarios up front to reduce reliance on
supplementary material.
5

2.2. Reviewer rnGR — Score: 6
Thoughtful and supportive. The review focuses more on scope/implementation details than disputing our
thesis.
2.2.1 Thoughtfulness
The comments probe real pain points for a nascent field: value boundaries, roadmap clarity, and annotation
governance. The request to name explicit tasks and provide taxonomy procedures is constructive.Some
expectations (e.g., detailed funding/governance mechanisms) are more aligned with a program proposal than
a position paper.
2.2.2 Support for Our Position
•Supportive elements: The review tacitly accepts the premise that olfaction can matter in embodied AI
by asking where it adds most value. It encourages sharpening boundaries and operational detail rather
than rejecting the thesis.
•Points of tension: Suggests we have not sufficiently contrasted scenarios where olfaction offers
minimal benefit. We respectfully disagree with the statement. Our claim is notthat olfaction is
universally superior, but that it is essential in specific embodied settings while offering limited marginal
value in others. We state these limits explicitly: (1) Section 2.2 (lines 121–147): We first discuss the
bandwidth of vision, audition, and olfaction. Then we note that current state of olfactory sensors does
not contain the bandwidth capabilities of its counterparts and therefore may not be useful in robotics
and AI scenarios requiring rapid, high-fidelity scene readouts. (2) Section 4: We discuss limitations of
our position and in the field of machine olfaction in general that include the likely need for adaptive
learning methods (lines 353 - 371), nuances surrounding manufacturing olfaction hardware (lines
377-382), and the slow sampling speeds of olfaction sensors (lines 380-382). Each of these points
amount support for why smell might offer minimal benefit over well-established modalities.
2.2.3 Focus: Technical Aspects vs. Position
Review leans toward implementation framing of the position (standards, governance, taxonomies) rather than
core scientific validity. Concrete task mapping and explicit annotation procedures (including bias handling)
are legitimate technical extensions that improve reproducibility and adoption.
2.2.4 Level of Gatekeeping
Low. The review requests clarity and concreteness; it does not impose out-of-scope experiments or dismiss
the modality a priori .
6

2.3. Reviewer TLRz — Score: 4
2.3.1 Thoughtfulness
Expectations for a comprehensive cross-discipline survey and fully specified benchmarks nudge beyond
typical position-track constraints. Reviewer’s suggestions to re-focus our position derails the perspective on
the role of AI/ robotics research in machine olfaction we intend to present.
2.3.2 Support for Our Position
•Partial alignment: By asking for where and how olfaction should be integrated (and for what tasks),
the review implicitly accepts the modality’s potential.
•Points of tension:
–We respectfully disagree that our manuscript fails to distinguish between (a) chemosensory
hardware and (b) subjective olfactory experience. The intent of the paper is to facilitate discussion
on points for why olfaction should or should not be included as a primary modality for embodied
intelligence, not to argue (a) how olfaction works biologically which is already a well established
argument in the life sciences [ 6] [8] [9] [3] [2] [4] (refer Section 2.1: Scientific Understanding
of Olfaction ) or (b) re-establish at-length arguments well articulated in the life sciences for smell
objectivity [7] [1] [5] (refer Section 2.3: Objectivity andAppendix C ).
–We strongly disagree with the statement that re-focusing on one of (a) or (b) would make the
paper stronger as we re-iterate that our position is that the slow progress and inequity in machine
olfaction—especially within artificial intelligence—can be attributed to five systemic gaps (which
includes both (a) and (b)). Narrowing the scope to only one would undercut our central thesis
that addressing even a subset of these gaps—via foundational research, infrastructure, and cross-
disciplinary collaboration—can materially accelerate the field and bring artificial olfaction closer
to parity with vision, audition, and language for embodied intelligence. While page limits preclude
a deeper discussion on chemosensory biology, as the reviewer suggested, we will incorporate
additional content in the Supplementary material to further delineate (a) hardware pipelines and
(b) perceptual modeling.
–Our manuscript does not claim that olfaction research is broadly overlooked. Our claim is
narrower: artificial olfaction within AI and robotics is comparatively underrepresented relative to
vision, audition, and language (see Section 2.1: The Scientific Understanding of Olfaction ). In
fact, part of our argument for the position in the paper is to leverage all of the excellent research
done within biological olfaction to help advance artificial olfaction to a similar magnitude that
biological vision, audition, taction, and speech has help advance their machine analogs. In Section
2.2: Olfaction Data Standard , we highlight this and elaborate on how AI researchers do not
have to start from scratch - but leverage foundational work within the life sciences to catalyze
artificial olfactory developments using knowledge about sensory bandwidths, encoding schemes,
and processing mechanisms.
–We respectfully disagree with the characterization that our proposal is vague. As a position-track
submission, our goal is to articulate a system-level agenda for machine olfaction in embodied
AI—framing priorities and pathways—rather than to present a single, fully specified solution. The
paper outlines concrete directions spanning data standards, benchmarking, hardware–software
integration, and cross-disciplinary infrastructure. If there are particular areas the reviewer found
under-specified, we would welcome that guidance so we can address it directly.
7

–We agree that today’s olfactory sensing challenges are tightly coupled to hardware. However, we
argue this makes the topic squarely relevant to the AI/robotics community: data characteristics are
shaped by sensor physics, and algorithmic advances can both diagnose and compensate hardware
limitations while informing sensor co-design. Our stance parallels the ImageNet-era inflection
in vision (circa 2009): the aim is not to address materials or EE audiences exclusively, but to
catalyze cross-disciplinary work that elevates olfaction to a first-class modality for embodied AI.
2.3.3 Focus: Technical Aspects vs. Position
Skew: Heavier on literature breadth and implementation concreteness than on disputing the central thesis.
Useful technical asks: Exemplars of datasets/benchmarks and explicit interfaces between sensor limits and
learning protocols.
2.3.4 Level of Gatekeeping
Incorporating reviewer’s suggestions derail the focus of the position we intend to present.
8

References
[1]Leffingwell & Associates. Pmp 2001 - database of perfumery materials and performance. http:
//www.leffingwell.com/bacispmp.htm . Accessed: 2025-03-08.
[2]Eric Block, Seogjoo Jang, Hiroaki Matsunami, Sivakumar Sekharan, Bérénice Dethier, Mehmed Z
Ertem, Sivaji Gundala, Yi Pan, Shengju Li, Zhen Li, Stephene N Lodge, Mehmet Ozbil, Huihong Jiang,
Sonia F Penalba, Victor S Batista, and Hanyi Zhuang. Implausibility of the vibrational theory of olfaction.
Proceedings of the National Academy of Sciences of the United States of America , 112(21):E2766–E2774,
May 2015. Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov’t; Research Support,
U.S. Gov’t, Non-P.H.S.
[3]A. E. Bourgeois and Joanne O. Bourgeois. Theories of olfaction: A review. Revista Interamericana de
Psicología/Interamerican Journal of Psychology , 4(1), Jul. 2017.
[4]Jennifer C Brookes, Filio Hartoutsiou, AP Horsfield, and AM Stoneham. Could humans recognize odor
by phonon assisted tunneling? Physical review letters , 98(3):038101, 2007.
[5]The Good Scents Company. The good scents company information system. http://www.
thegoodscentscompany.com/ . Accessed: 2025-03-08.
[6]GM Dyson. Some aspects of the vibration theory of odor. Perfumery and essential oil record , 19(456-459),
1928.
[7]Chuntae Kim, Kyung Kwan Lee, Moon Sung Kang, Dong-Myeong Shin, Jin-Woo Oh, Chang-Soo Lee,
and Dong-Wook Han. Artificial olfactory sensor technology that mimics the olfactory mechanism: a
comprehensive review. Biomaterials Research , 26(1):40, Aug 2022.
[8]G. Malcolm Dyson. The scientific basis of odour. Journal of the Society of Chemical Industry , 57(28):647–
651, 1938.
[9]Luca Turin, Simon Gane, Dimitris Georganakis, Klio Maniati, and Efthimios M. C. Skoulakis. Plausibility
of the vibrational theory of olfaction. Proceedings of the National Academy of Sciences , 112(25):E3154–
E3154, 2015.
9




> "The paper asserts that olfaction is essential for embodied AI but does not critically examine scenarios where smell might offer minimal benefit over well-established modalities like vision or audition."

We respectfully disagree with the statement. Our claim is not that olfaction is universally superior, but that it is essential in specific embodied settings while offering limited marginal value in others. We state these limits explicitly in Section 2.2 (lines 121–147) and Section 4 (lines 353 - 371 and 377-382). Each of the points established in these lines amount support for why smell might offer minimal benefit over well-established modalities.

> "The paper outlines data standard definition, annotation protocols, consortium formation and benchmark creation but provides no actionable details such as timelines or milestones; governance models or funding mechanisms for the proposed consortium; or concrete procedures for defining and calibrating annotation taxonomies."

(1) Interesting question! Our choice to omit timelines, milestones, governance, and funding details was deliberate. Through this position paper, we aimed to set an agenda rather than prescribe a single implementation pathway. Also, pre-selecting mechanisms risks hard-coding our biases and constraining community input.

(2) The lack of progress in machine olfaction thus far has partially been attributed to inappropriate timeline and budget proposals surrounding the complexity of olfactory sensing itself, and our preliminary peer reviews encouraged us to leave hard numbers out as a result. We highlight existing taxonomies and data standards that exist such as the current work in time series (lines 241 and 263), principal odor maps (lines 195, 230, 294), and combinatorial coding (line 154) that should all be taken in to consideration to inspire the community. We also note their shortcomings and critiques in order to foster more discussion from the AI/ML research community for improvements.




--


We genuinely appreciate the comments from Reviewer rnGR. However, we feel that they may have been viewing the position paper through a lens that reflects the requirements of the main track where explicit solutions are provided for hypotheses. While we provide general solutions to support both sides of our position, some details were intentionally left out in order to maintain the true spirit of the position track by not biasing the community on what the correct solution *should* be. In addition, some of the questions raised by this reviewer we feel were answered within several sections of the paper, so we worry that the paper was reviewed in a rush.




--








> "The difference between..."
We respectfully disagree that our manuscript fails to distinguish between (a) and (b). The intent of the paper is to facilitate discussion on why olfaction should or should not be included as a primary modality for embodied AI, not to argue (a) how olfaction works biologically (Section 2.1) or (b) objectivities around smell (Section 2.3), both of which are already well established arguments in the life sciences.

> "The paper claims olfaction research..."
(1) To clarify, our manuscript does not claim that olfaction research is broadly overlooked. Our claim is narrower: artificial olfaction within AI and robotics is comparatively underrepresented relative to vision, audition, and language (Section 2.1).  
(2) The paper the reviewer cites aligns with our position but unfortunately falls under the same community critique for human objectivity (Section 2.3). It has also only been cited 220 times - a very small number for a dataset attempting to define an AI-oriented benchmark in olfaction in comparison to, for example, ImageNet which has 88,803 citations at the time of this writing.

> "The proposal are vague / lack details."
We respectfully disagree with this characterization. As a position-track submission, our goal is to articulate a system-level agenda for machine olfaction in embodied AI rather than to present a fully specified solution. If there are particular areas the reviewer found under specified, we welcome guidance so we can address it directly.

> "The chemosensory side..."
We feel that more careful observance of the paper by the reviewer would have clarified this question as the contents for its answer exist in the paper. We argue this makes the topic squarely relevant to the AI community (lines 377–385).

> "What exactly should the benchmarks..."
Excellent question. A complete specification is beyond this survey’s scope and merits a dedicated paper; nonetheless, we outline idealized exemplars of these in (Sections 2.4-2.5).



We appreciate the comments by Reviewer 3. However, we worry that this review was done in haste, as several of the questions raised by the reviewer were thoroughly addressed in the paper. In addition, we strongly feel that the paper may not have been viewed in the context of an AI/ML conference as much of the critique from this reviewer was with respect to olfactory research in biology and chemistry, but not necessarily to its application in CS, AI, or robotics. 

We respectfully disagree that our manuscript claims olfaction research is overlooked. Our claim is narrower: artificial olfaction within AI and robotics is comparatively underrepresented relative to vision, audition, and language (see Section 2.1). In fact, part of our argument for the position in the paper is to leverage all of the excellent research done within biological olfaction to help advance artificial olfaction to a similar magnitude that biological vision, audition, taction, and speech has help advance their machine analogs. In Section 2.2, we highlight this and elaborate on how AI researchers do not have to start from scratch - but leverage foundational work within the life sciences to catalyze artificial olfactory developments. The intent of the paper is to facilitate discussion on points for why olfaction should or should not be included as a primary modality for embodied AI, not to argue how olfaction works biologically which is already a well established argument in the life sciences (refer Section 2.1) or re-establish at-length arguments well articulated in the life sciences for smell objectivity (refer Section 2.3 and Appendix C ).

As a position-track submission, our goal is to articulate a system-level agenda for machine olfaction in embodied AI—framing priorities and pathways—rather than to present a single, fully specified solution. If there are particular areas the reviewer found under-specified, we would welcome that guidance so we can address it directly.


---

Thoughtful, supportive feedback that mostly targets exposition (organization, citation hygiene).

**Thoughtfulness:**
The reviewer raised a substantive question about how turbulent, episodic plume dynamics should shape data
generation and capture strategies indicating close reading and an intent for meaningful engagement.

**Support for Our Position:**
With a score of 8 and suggestions to surface use cases and relocate recommendations for visibility, the review
reinforces our central claim.

**Focus: Technical Aspects vs. Position:**
Most comments address organization and scholarly apparatus (where recommendations appear, when to cite,
formatting). The episodic-sensing question is technical and conforms with our claims about sparsity, irregular
bursts, and combinatorial encoding.

**Level of Gatekeeping:**
None observed.

**Planned Revisions:**
- Elevate recommendations: Move the guidance currently in Section 2.4 (lines 268–278) to an earlier
section to foreground actionable takeaways.
- Surface use cases in the intro: Add representative application scenarios up front to reduce reliance on
supplementary material.


--
Thoughtful and supportive. The review focuses more on scope/implementation details than disputing our
thesis.

**Thoughtfulness:**
The comments probe real pain points for a nascent field: value boundaries, roadmap clarity, and governance. The request to name explicit tasks and provide taxonomy procedures is constructive. Some expectations (e.g. detailed funding/governance mechanisms) are more aligned with a program proposal than a position paper.

**Support for Our Position:**
- Supportive elements: The review tacitly accepts the premise that olfaction can matter in embodied AI by asking where it adds most value. It encourages sharpening boundaries and operational detail rather than rejecting the thesis.
- Points of tension: Suggests we have not sufficiently contrasted scenarios where olfaction offers minimal benefit. We respectfully disagree with the statement. Our claim is not that olfaction is universally superior, but that it is essential in specific embodied settings while offering limited marginal value in others. We state these limits explicitly: (1) We discuss the bandwidth of all modalities, and then note that current state of olfactory sensors does not contain the bandwidth capabilities of its counterparts (lines 121–147). (2) We discuss limitations of our position that include the likely need for adaptive learning methods (lines 353 - 371), nuances around manufacturing olfaction hardware (lines 377-382), and the slow sampling speeds of olfaction sensors (lines 380 382).

**Focus: Technical Aspects vs. Position:**
Review leans toward implementation framing of the position (standards, governance, taxonomies) rather than
core scientific validity. Concrete task mapping and explicit annotation procedures (including bias handling)
are legitimate technical extensions that improve reproducibility and adoption.

**Level of Gatekeeping:**
Low. The review requests clarity and concreteness; it does not impose out-of-scope experiments or dismiss
the modality a priori.
















**Thoughtfulness:**
Expectations for a comprehensive cross-discipline survey and fully specified benchmarks lie beyond position-track constraints. Reviewer’s suggestions to re-focus our position derails the perspective on the role of AI/robotics research in machine olfaction we intend to present.

**Support for Our Position:**
- Partial alignment: By asking for where and how olfaction should be integrated (and for what tasks), the review implicitly accepts the modality’s potential.
- Points of tension:
(1) We respectfully disagree that our manuscript fails to distinguish between (a) chemosensory hardware and (b) subjective olfactory experience. The intent of the paper is to facilitate discussion on points for why olfaction should or should not be included as a primary modality for embodied intelligence, not to argue (a) how olfaction works biologically which is already a well established argument in the life sciences (refer Section 2.1) or (b) re-establish at-length arguments well articulated in the life sciences for smell objectivity (refer Section 2.3 and Appendix C).

(2) Our manuscript does not claim that olfaction research is broadly overlooked. Our claim is narrower: artificial olfaction within AI and robotics is comparatively underrepresented relative to other modalities (see Section 2.1). In fact, part of our argument for the position in the paper is to leverage all of the excellent research done within biological olfaction to help advance artificial olfaction to a similar magnitude that biological vision, audition, taction, and speech has helped advance their machine analogs. 

**Focus: Technical Aspects vs. Position:**
Skew: Heavier on literature breadth and implementation than on disputing the central thesis. Useful technical asks: Exemplars of datasets/benchmarks and explicit interfaces between sensor limits and learning protocols.

**Level of Gatekeeping:**
Incorporating reviewer’s suggestions derail the focus of the position we intend to present.